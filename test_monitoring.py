import random

from cbnb_e import CBNBe
from nbcb_e import NBCBe
from nbcb_w import NBCBw
from cbnb_w import CBNBw
import baselines

from evalusation_measure import f1_score
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

import numpy as np
import pandas as pd


if __name__ == '__main__':
    param_method = "CBNB_e"  # CBNB_w NBCB_w CBNB_e NBCB_e GCMVL CCM PCMCI PCGCE VarLiNGAM
    param_tau_max = 5
    param_sig_level = 0.05

    f1_adjacency_list = []
    f1_orientation_list = []
    percentage_of_detection_skeleton = 0
    for j in range(2):
        structures = ["storm_fork", "storm_chain"]
        files_name = ["contiForkData", "contiChainData"]
        structure = structures[j]
        file_name = files_name[j]
        for i in range(1, 11):
            print("######################################### iter " + str(i) + " #########################################")
            param_data = pd.read_csv("./data/monitoring/returns/"+ structure + "/" + file_name + str(i) + ".csv", delimiter=',', index_col=False, header=0)
            param_data = param_data[param_data.columns[:3]]

            three_col_format = np.loadtxt("./data/monitoring/ground_truth/" + structure + ".csv",
                                          delimiter=',')

            summary_matrix = pd.DataFrame(np.zeros([param_data.shape[1], param_data.shape[1]]), columns=param_data.columns,
                                          index=param_data.columns)
            for i in range(three_col_format.shape[0]):
                c = param_data.columns[int(three_col_format[i, 0])]
                e = param_data.columns[int(three_col_format[i, 1])]
                summary_matrix[e].loc[c] = 1

            list_nodes = []
            for col_i in param_data.columns:
                list_nodes.append(GraphNode(col_i))
            causal_graph_true = GeneralGraph(list_nodes)
            for col_i in summary_matrix.columns:
                for col_j in summary_matrix.columns:
                    if (summary_matrix[col_j].loc[col_i] != 0) and (summary_matrix[col_i].loc[col_j] != 0):
                        causal_graph_true.add_edge(Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
                    elif summary_matrix[col_j].loc[col_i] != 0:
                        causal_graph_true.add_edge(Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))
                    elif summary_matrix[col_i].loc[col_j] != 0:
                        causal_graph_true.add_edge(Edge(GraphNode(col_j), GraphNode(col_i), Endpoint.TAIL, Endpoint.ARROW))

            print(causal_graph_true)
            if param_method == "NBCB_w":
                nbcb = NBCBw(param_data, param_tau_max, param_sig_level, model="linear",  indtest="linear", cond_indtest="linear")
                nbcb.run()
                causal_graph_hat = nbcb.causal_graph
            elif param_method == "CBNB_w":
                cbnb = CBNBw(param_data, param_tau_max, param_sig_level, model="linear", indtest="linear",
                             cond_indtest="linear")
                cbnb.run()
                causal_graph_hat = cbnb.causal_graph
            elif param_method == "NBCB_e":
                nbcb = NBCBe(param_data, param_tau_max, param_sig_level, model="linear", indtest="linear",
                             cond_indtest="linear")
                nbcb.run()
                causal_graph_hat = nbcb.causal_graph
            elif param_method == "CBNB_e":
                cbnb = CBNBe(param_data, param_tau_max, param_sig_level, model="linear", indtest="linear",
                             cond_indtest="linear")
                cbnb.run()
                causal_graph_hat = cbnb.causal_graph
            elif param_method == "GCMVL":
                causal_graph_hat = baselines.granger_lasso(param_data, tau_max=param_tau_max, sig_level=param_sig_level)
            elif param_method == "CCM":
                causal_graph_hat = baselines.ccm(param_data, tau_max=param_tau_max)
            elif param_method == "PCMCI":
                causal_graph_hat = baselines.pcmciplus(param_data, tau_max=param_tau_max, sig_level=param_sig_level)
            elif param_method == "PCGCE":
                causal_graph_hat = baselines.pcgce(param_data, tau_max=param_tau_max, sig_level=param_sig_level)
            elif param_method == "VarLiNGAM":
                causal_graph_hat = baselines.varlingam(param_data, tau_max=param_tau_max, sig_level=param_sig_level)
            elif param_method == "TiMINO":
                causal_graph_hat = baselines.run_timino_from_r([[param_data, "data"], [param_sig_level, "alpha"], [param_tau_max, "nlags"]])
            else:
                causal_graph_hat = None
            print(causal_graph_hat)

            fa = f1_score(causal_graph_hat, causal_graph_true, ignore_orientation=True)
            fo = f1_score(causal_graph_hat, causal_graph_true, ignore_orientation=False)
            print("F1 adjacency with desired graph= " + str(fa))
            print("F1 orientation with desired graph = " + str(fo))
            f1_adjacency_list.append(fa)
            f1_orientation_list.append(fo)

            if causal_graph_true.get_graph_edges() == causal_graph_hat.get_graph_edges():
                percentage_of_detection_skeleton = percentage_of_detection_skeleton + 1
            print("Percentage so far with true graph= " + str(percentage_of_detection_skeleton) + "/" + str(i + 1))

    print("#############################################")
    print("F1 adjacency with desired graph= " + str(np.mean(f1_adjacency_list)) + " +- " + str(np.var(f1_adjacency_list)))
    print("F1 orientation with desired graph= " + str(np.mean(f1_orientation_list)) + " +- " + str(np.var(f1_orientation_list)))
    print("Percentage of detection with desired graph= " + str(percentage_of_detection_skeleton/100))
