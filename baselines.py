import pandas as pd
import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


from sklearn.linear_model import LassoCV

import skccm
from skccm.utilities import train_test_split, exp_weight, corrcoef
from sklearn import neighbors


from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
from tigramite import data_processing as pp

from pcgce import PCGCE

from lingam_master.lingam.var_lingam import VARLiNGAM

from subprocess import Popen, PIPE
import os
import glob


def granger_lasso(data, tau_max=5, sig_level=0.05, cv=5):
    '''
    Granger causality test for multi-dimensional time series
    Parameters:
    -----------
    data - input data (nxd)
    maxlag: maximum time lag
    cv: number of cross validation folds for lasso regression
    Returns:
    ----------
    coeff: coefficient matrix [A_1, A_2, ..], where A_k is the dxd causal matrix for the k-th time lag. The ij-th entry
    in A_k represents the causal influence from j-th variable to the i-th variable.
    '''

    n, dim = data.shape
    # stack data to form one-vs-all regression
    Y = data.values[tau_max:]
    X = np.hstack([data.values[tau_max - k:-k] for k in range(1, tau_max + 1)])

    lasso_cv = LassoCV(cv=cv)
    coeff = np.zeros((dim, dim * tau_max))
    # Consider one variable after the other as target
    for i in range(dim):
        lasso_cv.fit(X, Y[:, i])
        coeff[i] = lasso_cv.coef_

    names = data.columns
    dataset = pd.DataFrame(np.zeros((len(names), len(names)), dtype=int), columns=names, index=names)
    # g = nx.DiGraph()
    # g.add_nodes_from(names)
    for i in range(dim):
        for l in range(tau_max):
            for j in range(dim):
                # if abs(coeff[i, j+l*dim]) > sig_level:
                if abs(coeff[i, j + l * dim]) > sig_level:
                    # g.add_edge(names[j], names[i])
                    dataset[names[i]].loc[names[j]] = 2

    for c in dataset.columns:
        for r in dataset.index:
            if dataset[r].loc[c] == 2:
                if dataset[c].loc[r] == 0:
                    dataset[c].loc[r] = 1
            if r == c:
                dataset.loc[r, c] = 1

    list_nodes = []
    for col in dataset.columns:
        list_nodes.append(GraphNode(col))
    causal_graph = GeneralGraph(list_nodes)
    for col_i in dataset.columns:
        for col_j in dataset.columns:
            if (dataset[col_j].loc[col_i] == 2) and (dataset[col_i].loc[col_j] == 2):
                causal_graph.add_edge(Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
            elif dataset[col_j].loc[col_i] == 2:
                causal_graph.add_edge(Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))

    return causal_graph


def ccm(data, tau_max=1, sig_level=0.05, embed_max=5):
    def shuffle(data_ij, tau_ij):
        data_suffle = data_ij.copy()
        personids = data_suffle.index.values.tolist()
        vector_i = data_suffle[data_suffle.columns[0]].values.tolist()
        vector_j = data_suffle[data_suffle.columns[1]].values.tolist()
        block_size = tau_ij + 1
        samples_i = []
        samples_j = []
        sampels_id = []
        block_idx = list(range(int(len(vector_i)/block_size)+1))
        num_blocks = len(block_idx)
        for idx in range(num_blocks):
            chosen_idx = np.random.choice(block_idx, size=1, replace=False)[0]
            block_idx.remove(chosen_idx)
            start = chosen_idx * block_size
            end = start + block_size - 1
            # np.random.randint(len(vector_i), size=num_samples)
            samples_i = samples_i + vector_i[start:end]
            samples_j = samples_j + vector_j[start:end]
            sampels_id = sampels_id + personids[start:end]
        diff_idx = list(set(personids) - set(sampels_id))
        for idx in diff_idx:
            samples_i.append(data_suffle[data_suffle.columns[0]].loc[idx])
            samples_j.append(data_suffle[data_suffle.columns[1]].loc[idx])
            sampels_id.append(idx)
        data_suffle[data_suffle.columns[0]] = samples_i
        data_suffle[data_suffle.columns[1]] = samples_j
        data_suffle.index = sampels_id
        return data_suffle

    def shuffle_test(data_ij, stat_i, stat_j, tau_ij, embed_ij, lib_lens_ij):
        min_list_i = []
        max_list_i = []
        min_list_j = []
        max_list_j = []
        for perm in range(100):
            data_suffle = shuffle(data_ij, tau_ij)
            ei = skccm.Embed(data_suffle[data_suffle.columns[0]].values)
            ej = skccm.Embed(data_suffle[data_suffle.columns[1]].values)
            X1 = ei.embed_vectors_1d(tau_ij, embed_ij)
            X2 = ej.embed_vectors_1d(tau_ij, embed_ij)
            x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)
            CCM = skccm.CCM()  # initiate the class
            CCM.fit(x1tr, x2tr)
            x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens_ij)
            sc1, sc2 = CCM.score()
            min_list_i.append(sc1[0])
            max_list_i.append(sc1[-1])
            min_list_j.append(sc2[0])
            max_list_j.append(sc2[-1])

        # diff_i = (stat_i[-1] - stat_i[0])
        # diff_j = (stat_j[-1] - stat_j[0])
        # p1 = sum([1 for min_i, max_i in zip(min_list_i, max_list_i) if (max_i - min_i) > diff_i])/100
        # p2 = sum([1 for min_j, max_j in zip(min_list_j, max_list_j) if (max_j - min_j) > diff_j])/100
        p1 = [1 for min_i, max_i in zip(min_list_i, max_list_i) if (abs(max_i) < abs(min_i))]
        p1 = sum(p1)/len(min_list_i)
        p2 = [1 for min_j, max_j in zip(min_list_j, max_list_j) if (abs(max_j) < abs(min_j))]
        p2 = sum(p2)/len(min_list_j)
        return p1, p2

    list_nodes = []
    for col in data.columns:
        list_nodes.append(GraphNode(col))
    causal_graph = GeneralGraph(list_nodes)
    summary_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]), columns=data.columns,
                                  index=data.columns)
    for col_i in data.columns:
        next_columns = list(data.columns[list(data.columns).index(col_i) + 1:])
        for col_j in next_columns:
            if col_i != col_j:
                # for tau in range(tau_max):
                # tunning lag: tau
                ei = skccm.Embed(data[[col_i, col_j]])
                mi = ei.df_mutual_information(tau_max)
                tau_i = np.argmin(mi[col_i]) + 1
                tau_j = np.argmin(mi[col_j]) + 1
                tau = min(tau_i, tau_j)

                # tunning embed: tau
                score_1 = []
                score_2 = []
                for embed in range(1, embed_max + 1):
                    ei = skccm.Embed(data[col_i].values)
                    ej = skccm.Embed(data[col_j].values)
                    X1 = ei.embed_vectors_1d(tau, embed)
                    X2 = ej.embed_vectors_1d(tau, embed)
                    x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)
                    x1_p = np.empty(x1te.shape)
                    x2_p = np.empty(x2te.shape)
                    near_neighs = x1tr.shape[1] + 1
                    knn1 = neighbors.KNeighborsRegressor(near_neighs)
                    knn2 = neighbors.KNeighborsRegressor(near_neighs)
                    knn1.fit(x1tr, x1tr)
                    knn2.fit(x2tr, x2tr)
                    dist1, ind1 = knn1.kneighbors(x1te)
                    dist2, ind2 = knn2.kneighbors(x2te)
                    for j in range(x1tr.shape[1]):
                        W1 = exp_weight(dist1)
                        W2 = exp_weight(dist2)
                        # flip the weights and indices
                        x1_p[:, j] = np.sum(x1tr[ind2, j] * W2, axis=1)
                        x2_p[:, j] = np.sum(x2tr[ind1, j] * W1, axis=1)
                    num_preds = x1tr.shape[1]
                    sc1 = np.empty(num_preds)
                    sc2 = np.empty(num_preds)
                    for ii in range(num_preds):
                        p1 = x1_p[:, ii]
                        p2 = x2_p[:, ii]
                        sc1[ii] = corrcoef(p1, x1te[:, ii])
                        sc2[ii] = corrcoef(p2, x2te[:, ii])
                    score_1.append(np.mean(sc1))
                    score_2.append(np.mean(sc2))
                embed_i = np.argmax(score_1) + 1
                embed_j = np.argmax(score_2) + 1
                embed = max(embed_i, embed_j)

                print("Tunned hyperparameters: tau=" + str(tau) + ", embed="+str(embed))

                # Start CCM
                ei = skccm.Embed(data[col_i].values)
                ej = skccm.Embed(data[col_j].values)

                X1 = ei.embed_vectors_1d(tau, embed)
                X2 = ej.embed_vectors_1d(tau, embed)
                x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)
                CCM = skccm.CCM()  # initiate the class
                # library lengths to test
                len_tr = len(x1tr)
                step = len(x1tr)
                lib_lens = np.arange(10, len_tr, len_tr / step, dtype='int')
                # test causation
                CCM.fit(x1tr, x2tr)
                x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)
                sc1, sc2 = CCM.score(how='corrcoef')

                # import matplotlib.pyplot as plt
                # plt.plot(sc1)
                # plt.plot(sc2)
                # plt.show()

                diff_i = (sc1[-1] - sc1[0])
                diff_j = (sc2[-1] - sc2[0])
                if diff_i > sig_level:
                    summary_matrix[col_j].loc[col_i] = 1
                if diff_j > sig_level:
                    summary_matrix[col_i].loc[col_j] = 1

                # pval_i, pval_j = shuffle_test(data[[col_i, col_j]], sc1, sc2, tau, embed, lib_lens)
                # if pval_i < sig_level:
                #     summary_matrix[col_j].loc[col_i] = 1
                # if pval_j < sig_level:
                #     summary_matrix[col_j].loc[col_i] = 1
                # print(col_i, col_j, pval_i, pval_j)

    for col_i in data.columns:
        for col_j in data.columns:
            if (summary_matrix[col_j].loc[col_i] == 1) and (summary_matrix[col_i].loc[col_j] == 1):
                if (not causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j))) and \
                        (not causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i))):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
            elif summary_matrix[col_j].loc[col_i] == 1:
                if not causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j)):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))
            elif summary_matrix[col_i].loc[col_j] == 1:
                if not causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i)):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_j), GraphNode(col_i), Endpoint.TAIL, Endpoint.ARROW))
    return causal_graph


def pcmciplus(data, tau_max=1, sig_level=0.05):
    dataframe = pp.DataFrame(data.values,
                             datatime=np.arange(len(data)),
                             var_names=data.columns)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    output = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=sig_level)


    summary_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]), columns=data.columns,
                                  index=data.columns)
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            for t in range(0, tau_max + 1):
                # print(i,j,t, output["graph"][i, j, t])
                if output["graph"][i, j, t] == '-->':
                    summary_matrix[data.columns[j]].loc[data.columns[i]] = 1
                elif output["graph"][i, j, t] == '<--':
                    summary_matrix[data.columns[i]].loc[data.columns[j]] = 1

    list_nodes = []
    for col in data.columns:
        list_nodes.append(GraphNode(col))
    causal_graph = GeneralGraph(list_nodes)

    for col_i in data.columns:
        for col_j in data.columns:
            if (summary_matrix[col_j].loc[col_i] == 1) and (summary_matrix[col_i].loc[col_j] == 1):
                if (not causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j))) and \
                        (not causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i))):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
            elif summary_matrix[col_j].loc[col_i] == 1:
                if not causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j)):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))
            elif summary_matrix[col_i].loc[col_j] == 1:
                if not causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i)):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_j), GraphNode(col_i), Endpoint.TAIL, Endpoint.ARROW))

    return causal_graph


def pcgce(data, tau_max=1, sig_level=0.05):
    pc = PCGCE(data, sig_lev=sig_level, lag_max=tau_max, verbose=False)
    output = pc.fit()


    ghat = pc.graph.to_summary()

    summary_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]), columns=data.columns,
                                  index=data.columns)
    for edge in ghat.edges:
        col_i = edge[0]
        col_j = edge[1]
        summary_matrix[col_j].loc[col_i] = 1

    list_nodes = []
    for col in data.columns:
        list_nodes.append(GraphNode(col))
    causal_graph = GeneralGraph(list_nodes)

    for col_i in data.columns:
        for col_j in data.columns:
            if (summary_matrix[col_j].loc[col_i] == 1) and (summary_matrix[col_i].loc[col_j] == 1):
                if (not causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j))) and \
                        (not causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i))):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
            elif summary_matrix[col_j].loc[col_i] == 1:
                if not causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j)):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))
            elif summary_matrix[col_i].loc[col_j] == 1:
                if not causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i)):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_j), GraphNode(col_i), Endpoint.TAIL, Endpoint.ARROW))

    return causal_graph


def varlingam(data, tau_max=1, sig_level=0.05):
    min_causal_effect = sig_level
    split_by_causal_effect_sign = True

    model = VARLiNGAM(lags=tau_max, criterion='bic', prune=True)
    model.fit(data)

    m = model._adjacency_matrices
    am = np.concatenate([*m], axis=1)

    # dag = np.abs(am) > min_causal_effect
    dag = np.abs(am) != 0

    if split_by_causal_effect_sign:
        direction = np.array(np.where(dag))
        signs = np.zeros_like(dag).astype('int64')
        for i, j in direction.T:
            signs[i][j] = np.sign(am[i][j]).astype('int64')
        dag = signs

    dag = np.abs(dag)
    names = data.columns
    res_dict = dict()

    # for col in data.columns:
    #     list_nodes.append(GraphNode(col))
    # causal_graph = GeneralGraph(list_nodes)
    summary_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]), columns=data.columns,
                                  index=data.columns)
    for e in range(dag.shape[0]):
        res_dict[names[e]] = []
    for c in range(dag.shape[0]):
        for te in range(dag.shape[1]):
            if dag[c][te] == 1:
                e = te%dag.shape[0]
                t = te//dag.shape[0]
                res_dict[names[e]].append((names[c], -t))
                # if not causal_graph.is_parent_of(list_nodes[c], list_nodes[e]):
                #     causal_graph.add_edge(Edge(list_nodes[c], list_nodes[e], Endpoint.ARROW, Endpoint.TAIL))
                summary_matrix[names[c]].loc[names[e]] = 1

    list_nodes = []
    for col in data.columns:
        list_nodes.append(GraphNode(col))
    causal_graph = GeneralGraph(list_nodes)

    for col_i in data.columns:
        for col_j in data.columns:
            if (summary_matrix[col_j].loc[col_i] == 1) and (summary_matrix[col_i].loc[col_j] == 1):
                if (not causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j))) and \
                        (not causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i))):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
            elif summary_matrix[col_j].loc[col_i] == 1:
                if not causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j)):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))
            elif summary_matrix[col_i].loc[col_j] == 1:
                if not causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i)):
                    causal_graph.add_edge(
                        Edge(GraphNode(col_j), GraphNode(col_i), Endpoint.TAIL, Endpoint.ARROW))

    return causal_graph


def run_timino_from_r(arg_list):
    def clear_args(dir_path):
        files = glob.glob(dir_path + '/args/*')
        for f in files:
            os.remove(f)

    def clear_results(dir_path):
        files = glob.glob(dir_path + '/results/*')
        for f in files:
            os.remove(f)

    # Remove all arguments from directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    script = dir_path + "/timino.R"
    clear_args(dir_path)
    clear_results(dir_path)
    r_arg_list = []
    # COMMAND WITH ARGUMENTS
    for a in arg_list:
        if isinstance(a[0], pd.DataFrame):
            a[0].to_csv(dir_path + "/args/"+a[1]+".csv", index=False)
            r_arg_list.append(dir_path + "/args/" + a[1] + ".csv")
        if isinstance(a[0], int):
            f = open(dir_path + "/args/"+a[1]+".txt", "w")
            f.write(str(a[0]))
            f.close()
            r_arg_list.append(dir_path + "/args/" + a[1] + ".txt")
        if isinstance(a[0], float):
            f = open(dir_path + "/args/"+a[1]+".txt", "w")
            f.write(str(a[0]))
            f.close()
            r_arg_list.append(dir_path + "/args/" + a[1] + ".txt")

    r_arg_list.append(dir_path)
    cmd = ["Rscript", script] + r_arg_list

    p = Popen(cmd, cwd="./", stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # Return R output or error
    output, error = p.communicate()
    # print(output)
    if p.returncode == 0:
        print('R Done')
        g_df = pd.read_csv(dir_path + "/results/result.csv", header=0, index_col=0)
        # g_df = g_df.transpose()
        print(g_df)
        list_nodes = []
        for col in g_df.columns:
            list_nodes.append(GraphNode(col))
        causal_graph = GeneralGraph(list_nodes)
        for col_i in g_df.columns:
            for col_j in g_df.columns:
                if (g_df[col_j].loc[col_i] == 2) and (g_df[col_i].loc[col_j] == 2):
                    causal_graph.add_edge(Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
                elif g_df[col_j].loc[col_i] == 2:
                    causal_graph.add_edge(Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))
        return causal_graph
    else:
        print('R Error:\n {0}'.format(error))
        exit(0)
