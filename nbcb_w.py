import random
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression as lr

from tigramite.pcmci import PCMCI
from pcmci_with_bk import PCMCI as PCMCIbk
from tigramite.independence_tests import ParCorr
from tigramite import data_processing as pp

from subprocess import Popen, PIPE
import os
import glob

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from lingam_master.lingam.var_lingam import VARLiNGAM

from cbnb_w import CBNBw

import matplotlib.pyplot as plt


def clear_args(dir_path):
    files = glob.glob(dir_path + '/args/*')
    for f in files:
        os.remove(f)


def clear_results(dir_path):
    files = glob.glob(dir_path + '/results/*')
    for f in files:
        os.remove(f)


def process_data(data, nlags):
    nodes_to_temporal_nodes = dict()
    temporal_nodes = []
    for node in data.columns:
        nodes_to_temporal_nodes[node] = []
        for gamma in range(nlags + 1):
            if gamma == 0:
                temporal_node = str(node) + "_t"
                nodes_to_temporal_nodes[node].append(temporal_node)
                temporal_nodes.append(temporal_node)
            else:
                temporal_node = str(node) + "_t_" + str(gamma)
                nodes_to_temporal_nodes[node].append(temporal_node)
                temporal_nodes.append(temporal_node)

    new_data = pd.DataFrame()
    for gamma in range(0, nlags + 1):
        shifteddata = data.shift(periods=-nlags + gamma)

        new_columns = []
        for node in data.columns:
            new_columns.append(nodes_to_temporal_nodes[node][gamma])
        shifteddata.columns = new_columns
        new_data = pd.concat([new_data, shifteddata], axis=1, join="outer")
    new_data.dropna(axis=0, inplace=True)
    return new_data, nodes_to_temporal_nodes, temporal_nodes


def run_varlingam(data, tau_max):
    # temporal_data, col_to_temporal_col, _ = process_data(data, nlags)
    model = VARLiNGAM(lags=tau_max, criterion='bic', prune=False)
    model.fit(data)
    order = model.causal_order_

    order = [data.columns[i] for i in order]
    order.reverse()

    order_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]), columns=data.columns, index=data.columns, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    order_matrix[col_j].loc[col_i] = 2
                    order_matrix[col_i].loc[col_j] = 1
    return order_matrix


def get_dependence_and_significance(x, e, indtest="linear"):
    e = e.reshape(-1, 1)
    dim_x = x.shape[1]
    dim_e = e.shape[1]
    a = np.concatenate((x, e), axis=1)
    xe = np.array([0] * dim_x + [1] * dim_e)

    if indtest == "linear":
        test = ParCorr(significance='analytic')
        statval = test.get_dependence_measure(a, xe)
        pval = test.get_shuffle_significance(a, xe, statval)
    else:
        pval, statval = 0, 0
        exit(0)
    return pval, statval


def get_prediction(X, y, model="linear"):
    if model == "linear":
        reg = lr().fit(X, y)
        yhat = reg.predict(X)
    else:
        yhat = 0
        exit(0)
    return yhat


def run_timino(data, nlags, model="linear", indtest="linear"):
    order = []
    temporal_data, col_to_temporal_col, _ = process_data(data, nlags)
    list_col = list(data.columns.copy())
    while len(list_col) > 1:
        list_pval = []
        list_statval = []
        temporal_cols = []
        for col_j in list_col:
            temporal_cols = temporal_cols + col_to_temporal_col[col_j]
        for col_i in list_col:
            temporal_col_i = col_to_temporal_col[col_i][0]
            print(temporal_col_i, temporal_cols)
            temporal_data_temp = temporal_data[temporal_cols].copy()
            X = temporal_data_temp.drop(temporal_col_i, inplace=False, axis=1).values
            y = temporal_data_temp[temporal_col_i].values
            yhat = get_prediction(X, y, model=model)
            err = y - yhat
            # plt.plot(err, X[:, 0], ".")
            # plt.show()
            pval, statval = get_dependence_and_significance(X, err, indtest=indtest)
            list_pval.append(pval)
            list_statval.append(statval)
        if len(set(list_pval)) == 1:
            tmp = min(list_statval)
            index = list_statval.index(tmp)
        else:
            tmp = max(list_pval)
            index = list_pval.index(tmp)
        col_index = list_col[index]
        order.append(col_index)
        list_col.remove(col_index)
    order.append(list_col[0])

    order_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]), columns=data.columns, index=data.columns, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    order_matrix[col_j].loc[col_i] = 2
                    order_matrix[col_i].loc[col_j] = 1
    return order_matrix


def run_timino_from_r(arg_list):
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
        # print(g_df)
        # g_df = g_df.transpose()
        return g_df
    else:
        print('R Error:\n {0}'.format(error))
        exit(0)


class NBCBw:
    def __init__(self, data, tau_max, sig_level, model="linear",  indtest="linear", cond_indtest="linear"):
        """
        :param extra_background_knowledge_list:
        """
        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.model =model
        self.indtest = indtest
        self.cond_indtest = cond_indtest

        self.causal_order = []
        self.graph = []

        self.forbidden_orientation = []

        self.window_causal_graph_dict = dict()
        list_nodes = []
        for col in data.columns:
            self.window_causal_graph_dict[col] = []
            list_nodes.append(GraphNode(col))
        self.causal_graph = GeneralGraph(list_nodes)

    def noise_based(self):
        # for col_i in self.data.columns:
        #     1
        self.causal_order = run_varlingam(self.data, self.tau_max)
        # order_df = run_timino(self.data, self.tau_max, model=self.model, indtest=self.indtest)
        # self.causal_order = run_timino_from_r([[self.data, "data"], [0.00, "alpha"], [self.tau_max, "nlags"]])

        list_columns = list(self.causal_order.columns)
        for col_i in list_columns:
            for col_j in list_columns:
                if (self.causal_order[col_j].loc[col_i] == 2) and (self.causal_order[col_i].loc[col_j] == 1):
                    index_i = list_columns.index(col_i)
                    index_j = list_columns.index(col_j)
                    self.forbidden_orientation.append((index_j, index_i))

    def constraint_based(self, bk=True):
        dataframe = pp.DataFrame(self.data.values,
                                 datatime=np.arange(len(self.data)),
                                 var_names=self.data.columns)
        parcorr = ParCorr(significance='analytic')
        if bk:
            pcmci = PCMCIbk(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
            output = pcmci.run_pcmciplus(tau_min=0, tau_max=self.tau_max, pc_alpha=self.sig_level,
                                         forbidden_orientation=self.forbidden_orientation)
        else:
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
            output = pcmci.run_pcmciplus(tau_min=0, tau_max=self.tau_max, pc_alpha=self.sig_level)

        # all_parents = pcmci.all_parents
        # print(output["graph"])
        summary_matrix = pd.DataFrame(np.zeros([self.data.shape[1], self.data.shape[1]]), columns=self.data.columns,
                                      index=self.data.columns)
        for i in range(len(self.data.columns)):
            for j in range(len(self.data.columns)):
                for t in range(0, self.tau_max + 1):
                    # if bk:
                    #     print(i,j,t, output["graph"][i, j, t])
                    if output["graph"][i, j, t] == '-->':
                        if (self.data.columns[i], -t) not in self.window_causal_graph_dict[self.data.columns[j]]:
                            self.window_causal_graph_dict[self.data.columns[j]].append((self.data.columns[i], -t))
                            summary_matrix[self.data.columns[j]].loc[self.data.columns[i]] = 1
                    elif output["graph"][i, j, t] == '<--':
                        if (self.data.columns[j], -t) not in self.window_causal_graph_dict[self.data.columns[i]]:
                            self.window_causal_graph_dict[self.data.columns[i]].append((self.data.columns[j], -t))
                            summary_matrix[self.data.columns[i]].loc[self.data.columns[j]] = 1
                    elif output["graph"][i, j, t] == "o-o":
                        self.window_causal_graph_dict[self.data.columns[j]].append((self.data.columns[i], -t))
                        # summary_matrix[self.data.columns[i]].loc[self.data.columns[j]] = 1
                        print("##################################################################!!!!!!!!!!!!!", i, j, t)
                    elif output["graph"][i, j, t] == "x-x":
                        self.window_causal_graph_dict[self.data.columns[j]].append((self.data.columns[i], -t))
                        # summary_matrix[self.data.columns[i]].loc[self.data.columns[j]] = 1
                        print("!!!!!!!!!!!!!##################################################################", i, j, t)

        for col_i in self.data.columns:
            for col_j in self.data.columns:
                if (summary_matrix[col_j].loc[col_i] == 1) and (summary_matrix[col_i].loc[col_j] == 1):
                    if (not self.causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j))) and \
                            (not self.causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i))):
                        self.causal_graph.add_edge(
                            Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
                elif summary_matrix[col_j].loc[col_i] == 1:
                    if not self.causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j)):
                        self.causal_graph.add_edge(
                            Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))
                elif summary_matrix[col_i].loc[col_j] == 1:
                    if not self.causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i)):
                        self.causal_graph.add_edge(
                            Edge(GraphNode(col_j), GraphNode(col_i), Endpoint.TAIL, Endpoint.ARROW))

    def run(self):
        print("######## Running Noise-based step ########")
        self.noise_based()
        print("######## Running Constraint-based step ########")
        self.constraint_based()


def uniform_with_gap(min_value=-1, max_value=1, min_gap=-0.5, max_gap=0.5):
    while True:
        r = random.uniform(min_value, max_value)
        if min_gap>r or max_gap<r:
            break
    return r


def v_structure_generator(T=1000, seed=0, verbose=False):
    T = T+2
    random.seed(seed)
    ax = uniform_with_gap()
    ay = uniform_with_gap()
    aw = uniform_with_gap()
    axw = uniform_with_gap()
    ayw = uniform_with_gap()
    bx = 0.3
    by = 0.3
    bw = 0.3

    if verbose:
        print("V-structure: 0 -> 2 <- 1")
        print(ax, bx, ay,by, aw, bw, axw, ayw)
    epsx = np.random.uniform(0, 1, T)
    epsy = np.random.uniform(0, 1, T)
    epsw = np.random.uniform(0, 1, T)

    x = np.zeros([T])
    y = np.zeros([T])
    w = np.zeros([T])

    x[0] = bx * epsx[0]
    y[0] = by * epsy[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + by * epsy[1]
    x[2] = ax * x[1] + bx * epsx[2]
    y[2] = ay * y[1] + by * epsy[2]
    w[2] = aw * w[1] + axw * x[2] + ayw * y[2] + bw * epsw[2]
    for i in range(3, T):
        x[i] = ax * x[i - 1] - 0.8 * y[i] + bx * epsx[i]
        y[i] = ay * y[i - 1] + by * epsy[i]
        w[i] = aw * w[i - 1] + axw * x[i] + ayw * y[i] + bw * epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0,1]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series


if __name__ == '__main__':
    for k in range(20):
        print("############### Inter " +str(k))
        param_data = v_structure_generator()
        # print(param_data)

        res = run_timino(param_data, 4)
        print(res)

        nbcb = NBCBc(param_data, 4, 0.05)
        nbcb.run()
        print(nbcb.causal_order)
        print(nbcb.causal_graph)
        print(nbcb.window_causal_graph_dict)

        nbcb2 = NBCBc(param_data, 4, 0.05)
        nbcb2.constraint_based(bk=False)
        print(nbcb2.window_causal_graph_dict)
        # print(nbcb.causal_graph)

        cbnb = CBNBc(param_data, 4, 0.05)
        cbnb.run()
        # print(cbnb.causal_graph)
        print(cbnb.window_causal_graph_dict)

        print(nbcb.window_causal_graph_dict == nbcb2.window_causal_graph_dict,
              nbcb2.window_causal_graph_dict == cbnb.window_causal_graph_dict,
              nbcb.window_causal_graph_dict == cbnb.window_causal_graph_dict)

