import random

import networkx as nx
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression as lr

# from tigramite.pcmci import PCMCI
# from pcmci_with_bk import PCMCI as PCMCIbk
# from tigramite.independence_tests import ParCorr
# from tigramite import data_processing as pp
from pcgce import CITCE


from subprocess import Popen, PIPE
import os
import glob

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from lingam_master.lingam.var_lingam import VARLiNGAM

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
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    dim_x = x.shape[1]
    dim_e = e.shape[1]
    a = np.concatenate((x, e), axis=1)
    xe = np.array([0] * dim_x + [1] * dim_e)

    if indtest == "linear":
        test = ParCorr(significance='analytic')
        statval = test.get_dependence_measure(a, xe)
        # pval = test.get_shuffle_significance(a, xe, statval)
        pval = test.get_analytic_significance(value=statval, T=a.shape[0], dim=a.shape[1], xyz=xe)
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


def run_timino2(list_targets, list_parents, data, nlags, model="linear", indtest="linear"):
    sub_temporal_data, col_to_temporal_col, temporal_nodes = process_data(data[list_targets + list_parents], nlags)

    list_temporal_target = []
    for node in list_targets:
        list_temporal_target.append(col_to_temporal_col[node][0])
    list_temporal_parents = list(set(temporal_nodes) - set(list_temporal_target))

    order = []
    list_targets_saved = list_targets.copy()
    # list_parents = list(set(list(data.columns)) - set(list_targets))
    while len(list_temporal_target) > 1:
        list_pval = []
        list_statval = []
        temporal_cols = list_temporal_target.copy()
        temporal_cols = temporal_cols + list_temporal_parents
        for temporal_col_i in list_temporal_target:
            temporal_data_temp = sub_temporal_data[temporal_cols].copy()
            X = temporal_data_temp.drop(temporal_col_i, inplace=False, axis=1).values
            y = temporal_data_temp[temporal_col_i].values
            yhat = get_prediction(X, y, model=model)
            err = y - yhat
            pval, statval = get_dependence_and_significance(X, err, indtest=indtest)
            list_pval.append(pval)
            list_statval.append(statval)
        if len(set(list_pval)) == 1:
            tmp = min(list_statval)
            index = list_statval.index(tmp)
        else:
            tmp = max(list_pval)
            index = list_pval.index(tmp)
        temporal_col_index = list_temporal_target[index]
        col_index = list_targets[index]
        list_temporal_target.remove(temporal_col_index)
        order.append(col_index)
        list_targets.remove(col_index)
    order.append(list_targets[0])

    order_matrix = pd.DataFrame(np.zeros([len(list_targets_saved), len(list_targets_saved)]), columns=list_targets_saved, index=list_targets_saved, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    order_matrix[col_j].loc[col_i] = 2
                    order_matrix[col_i].loc[col_j] = 1
    return order_matrix


def run_timino(list_targets, list_parents, data, nlags, model="linear", indtest="linear"):
    order = []
    list_targets_saved = list_targets.copy()
    # list_parents = list(set(list(data.columns)) - set(list_targets))
    while len(list_targets) > 1:
        list_pval = []
        list_statval = []
        temporal_cols = list_targets.copy()
        temporal_cols = temporal_cols + list_parents
        for temporal_col_i in list_targets:
            temporal_data_temp = data[temporal_cols].copy()
            X = temporal_data_temp.drop(temporal_col_i, inplace=False, axis=1).values
            y = temporal_data_temp[temporal_col_i].values
            yhat = get_prediction(X, y, model=model)
            err = y - yhat
            pval, statval = get_dependence_and_significance(X, err, indtest=indtest)
            list_pval.append(pval)
            list_statval.append(statval)
        if len(set(list_pval)) == 1:
            tmp = min(list_statval)
            index = list_statval.index(tmp)
        else:
            tmp = max(list_pval)
            index = list_pval.index(tmp)
        col_index = list_targets[index]
        order.append(col_index)
        list_targets.remove(col_index)
    order.append(list_targets[0])

    order_matrix = pd.DataFrame(np.zeros([len(list_targets_saved), len(list_targets_saved)]), columns=list_targets_saved, index=list_targets_saved, dtype=int)
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


class CBNBe:
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
        self.window_causal_graph = None
        list_nodes = []
        for col in data.columns:
            self.window_causal_graph_dict[col] = []
            list_nodes.append(GraphNode(col))
        self.causal_graph = GeneralGraph(list_nodes)

    def constraint_based(self):
        pcgce = CITCE(self.data, sig_lev=self.sig_level, lag_max=self.tau_max)
        pcgce.skeleton_initialize()
        pcgce.find_sep_set()

        print(pcgce.graph.ghat.edges)

        col_names = list(self.data.columns)
        self.window_causal_graph = np.full([len(col_names), len(col_names), self.tau_max + 1], "---")
        map_names_nodes_inv = dict()
        nodes = list(pcgce.graph.map_names_nodes.keys())
        for node in nodes:
            for node_t in pcgce.graph.map_names_nodes[node]:
                map_names_nodes_inv[node_t] = node
        for edge in pcgce.graph.ghat.edges:
            node_0 = map_names_nodes_inv[edge[0]]
            node_1 = map_names_nodes_inv[edge[1]]
            i = col_names.index(node_0)
            j = col_names.index(node_1)
            if edge[0] == pcgce.graph.map_names_nodes[node_0][0]:
                for t in range(1, self.tau_max + 1):
                    self.window_causal_graph[i, j, t] = "-->"
                    # self.window_causal_graph[j, i, t] = "<--"
            elif edge[0] == pcgce.graph.map_names_nodes[node_0][1]:
                self.window_causal_graph[i, j, 0] = "o-o"
                self.window_causal_graph[j, i, 0] = "o-o"
            else:
                print("something is wrong")
                exit(0)

    def find_cycle_groups(self, ):
        instantaneous_nodes = []
        instantaneous_graph = nx.Graph()
        for i in range(len(self.data.columns)):
            for j in range(len(self.data.columns)):
                t = 0
                if (self.window_causal_graph[i, j, t] == "o-o") or (self.window_causal_graph[i, j, t] == "x-x") or \
                        (self.window_causal_graph[i, j, t] == "-->") or (self.window_causal_graph[i, j, t] == "<--"):
                    instantaneous_graph.add_edge(self.data.columns[i], self.data.columns[j])
                    if self.data.columns[i] not in instantaneous_nodes:
                        instantaneous_nodes.append(self.data.columns[i])
        list_cycles = nx.cycle_basis(instantaneous_graph)

        # create cycle groups
        cycle_groups = dict()
        idx = 0
        for i in range(len(list_cycles)):
            l1 = list_cycles[i]
            test_inclusion = True
            for k in cycle_groups.keys():
                for e1 in l1:
                    if e1 not in cycle_groups[k]:
                        test_inclusion = False
            if (not test_inclusion) or (len(cycle_groups.keys()) == 0):
                cycle_groups[idx] = l1
                idx = idx + 1
                for j in range(i + 1, len(list_cycles)):
                    l2 = list_cycles[j]
                    if l1 != l2:
                        if len(list(set(cycle_groups[idx - 1]).intersection(l2))) >= 2:
                            cycle_groups[idx - 1] = cycle_groups[idx - 1] + list(set(l2) - set(cycle_groups[idx - 1]))

        # adding edges that do not belong to any cycles
        for edge in instantaneous_graph.edges:
            if len(list_cycles) > 0:
                for cycle in list_cycles:
                    if (edge[0] not in cycle) or (edge[1] not in cycle):
                        if list(edge) not in list_cycles:
                            list_cycles.append(list(edge))
                            cycle_groups[idx] = list(edge)
                            idx = idx + 1
            else:
                list_cycles.append(list(edge))
                cycle_groups[idx] = list(edge)
                idx = idx + 1
        return cycle_groups, list_cycles, instantaneous_nodes

    # def noise_based(self):
    #     cycle_groups, list_cycles, instantaneous_nodes = self.find_cycle_groups()
    #
    #     if len(instantaneous_nodes)>1:
    #         print(instantaneous_nodes)
    #         sub_temporal_data, col_to_temporal_col, temporal_nodes = process_data(self.data[instantaneous_nodes], self.tau_max)
    #         temporal_instantaneous_nodes = []
    #         for k in col_to_temporal_col.keys():
    #             temporal_instantaneous_nodes.append(col_to_temporal_col[k][0])
    #         non_instantaneous_nodes = list(set(temporal_nodes) - set(temporal_instantaneous_nodes))
    #         causal_order = run_timino(temporal_instantaneous_nodes, non_instantaneous_nodes, sub_temporal_data, self.tau_max)
    #         # causal_order = run_timino_from_r([[sub_data, "data"], [0.00, "alpha"], [self.tau_max, "nlags"]])
    #
    #         for i in range(len(self.data.columns)):
    #             for j in range(len(self.data.columns)):
    #                 t = 0
    #                 if (self.window_causal_graph[i, j, t] == "o-o") or (self.window_causal_graph[i, j, t] == "x-x"):
    #                     tempora_node_i = col_to_temporal_col[self.data.columns[i]][0]
    #                     tempora_node_j = col_to_temporal_col[self.data.columns[j]][0]
    #                     if causal_order[tempora_node_j].loc[tempora_node_i] == 2 and \
    #                                 (causal_order[tempora_node_i].loc[tempora_node_j] == 1):
    #                         self.window_causal_graph[i, j, t] = "-->"
    #                         self.window_causal_graph[j, i, t] = "<--"
    #                         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", i, j,
    #                               t)
    #     # return causal_order

    def noise_based(self):
        cycle_groups, list_cycles, instantaneous_nodes = self.find_cycle_groups()
        print(instantaneous_nodes)
        print(cycle_groups)


        list_columns = list(self.data.columns)
        if len(instantaneous_nodes)>1:
            for idx in cycle_groups.keys():
                instantaneous_nodes = cycle_groups[idx]

                parents_nodes = list(set(list_columns) - set(instantaneous_nodes))
                parents_nodes_temp = parents_nodes.copy()
                for node in instantaneous_nodes:
                    j = list_columns.index(node)
                    for parent_node in parents_nodes_temp:
                        if parent_node in parents_nodes:
                            test_parent = True
                            i = list_columns.index(parent_node)
                            for t in range(1, self.tau_max + 1):
                                if self.window_causal_graph[i, j, t] == "-->":
                                    test_parent = False
                            if test_parent:
                                parents_nodes.remove(parent_node)

                sub_data = self.data[instantaneous_nodes + parents_nodes]
                causal_order = run_varlingam(sub_data, self.tau_max)
                print(causal_order)
                # causal_order = run_timino2(instantaneous_nodes, parents_nodes, sub_data, self.tau_max)

                for col_i in instantaneous_nodes:
                    for col_j in instantaneous_nodes:
                        if (causal_order[col_j].loc[col_i] == 2) and (causal_order[col_i].loc[col_j] == 1):
                            i = list_columns.index(col_i)
                            j = list_columns.index(col_j)
                            t = 0
                            if (self.window_causal_graph[i, j, t] == "o-o") or (
                                    self.window_causal_graph[i, j, t] == "x-x") or \
                                    (self.window_causal_graph[i, j, t] == "-->") or (self.window_causal_graph[i, j, t] == "<--"):
                                self.window_causal_graph[i, j, t] = "-->"
                                self.window_causal_graph[j, i, t] = "<--"
                                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", i, j,t)

    def construct_summary_causal_graph(self):
        summary_matrix = pd.DataFrame(np.zeros([self.data.shape[1], self.data.shape[1]]), columns=self.data.columns,
                                      index=self.data.columns)
        for i in range(len(self.data.columns)):
            for j in range(len(self.data.columns)):
                for t in range(0, self.tau_max + 1):
                    if self.window_causal_graph[i, j, t] == '-->':
                        if (self.data.columns[i], -t) not in self.window_causal_graph_dict[self.data.columns[j]]:
                            self.window_causal_graph_dict[self.data.columns[j]].append((self.data.columns[i], -t))
                            summary_matrix[self.data.columns[j]].loc[self.data.columns[i]] = 1
                    elif self.window_causal_graph[i, j, t] == '<--':
                        if (self.data.columns[j], -t) not in self.window_causal_graph_dict[self.data.columns[i]]:
                            self.window_causal_graph_dict[self.data.columns[i]].append((self.data.columns[j], -t))
                            summary_matrix[self.data.columns[i]].loc[self.data.columns[j]] = 1
                    elif (self.window_causal_graph[i, j, t] == "o-o") or (self.window_causal_graph[i, j, t] == "x-x"):
                        # sub_data = self.data[[self.data.columns[i], self.data.columns[j]]]
                        # sub_data = self.data
                        # causal_order = self.noise_based(sub_data)
                        # if causal_order[self.data.columns[j]].loc[self.data.columns[i]] == 2 and \
                        #         (causal_order[self.data.columns[i]].loc[self.data.columns[j]] == 1):
                        #     self.window_causal_graph_dict[self.data.columns[j]].append((self.data.columns[i], -t))
                        #     summary_matrix[self.data.columns[j]].loc[self.data.columns[i]] = 1
                        print("##################################################################!!!!!!!!!!!!!", i, j,
                              t)

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
        print("######## Running Constraint-based ########")
        self.constraint_based()
        print(self.window_causal_graph, 111)
        print("######## Running Noise-based ########")
        self.noise_based()
        print(self.window_causal_graph, 222)
        print("######## Construct summary causal graph ########")
        self.construct_summary_causal_graph()


def uniform_with_gap(min_value=-1, max_value=1, min_gap=-0.5, max_gap=0.5):
    while True:
        r = random.uniform(min_value, max_value)
        if min_gap>r or max_gap<r:
            break
    return r


if __name__ == '__main__':
    # from test_simulated_data import v_structure_generator
    # param_data = v_structure_generator()
    # # print(param_data)
    #
    # # res = run_timino(param_data, 4)
    # # print(res)
    #
    # nbcb = CBNBc(param_data, 4, 0.05)
    # nbcb.run()
    # print(nbcb.causal_order)
    # print(nbcb.window_causal_graph_dict)

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(0, 3)
    g.add_edge(0, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(0, 5)
    g.add_edge(0, 6)
    g.add_edge(6, 7)
    g.add_edge(5, 7)

    list_cycles = nx.cycle_basis(g)
    print(list_cycles)





