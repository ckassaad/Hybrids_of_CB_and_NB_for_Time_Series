from nbcb_e import NBCBe
from cbnb_e import CBNBe
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
    param_method = "CBNB_w"  # CBNB_w NBCB_w CBNB_e NBCB_e GCMVL CCM PCMCI PCGCE VarLiNGAM
    param_tau_max = 5
    param_sig_level = 0.05
    param_nb_obervation = 500
    param_number_of_species = 10  # 5 or 10

    if param_number_of_species == 5:
        param_number_of_species = "ts5"
    elif param_number_of_species == 10:
        param_number_of_species = "ts10"
    elif param_number_of_species == 20:
        param_number_of_species = "ts20"

    f1_adjacency_list = []
    f1_orientation_list = []
    percentage_of_detection_skeleton = 0

    list_range = list(range(1, 100))
    # list_range.remove(76)
    for i in list_range:
        print("######################################### iter " + str(i) + " #########################################")
        # param_data = pd.read_csv("./data/Lotka_Voltera/" + param_number_of_species + "/data_map_1_graph_1_TS/glv.out.abioticGR_env_0.02_" + str(i) + ".csv")
        param_data = pd.read_csv("./data/Lotka_Voltera/" + param_number_of_species + "/data_map_1_graph_1_TS/ricker.out.abioticKbasal_out0.5_" + str(i) + ".csv")
        param_data = param_data.iloc[:param_nb_obervation]

        param_data.replace(to_replace=0, value=0.0000000001, inplace=True)

        # param_data = param_data.transform(np.log)

        # for col in param_data.columns:
        #     if param_data[col].loc[20:].std() < 0.000001:
        #         param_data[col] + np.random.uniform(-0.000001, 0.000001, size=param_data.shape[0])

        # param_data = param_data + np.random.uniform(-0.000001, 0.000001, size=param_data.shape)
        # param_data = param_data + np.random.normal(0, 0.000001, size=param_data.shape)

        # for col in param_data.columns:
        #     param_data[col] = np.log(param_data[col])
        # import matplotlib.pyplot as plt
        # param_data.plot()
        # plt.show()

        print(param_data)

        param_matrix = pd.read_csv("./data/Lotka_Voltera/" + param_number_of_species + "/int_mat/InteractionMatrix_" + str(i) + ".csv")
        param_matrix.index = param_matrix.columns
        print(param_matrix)
        list_nodes = []
        for col_i in param_matrix.columns:
            list_nodes.append(GraphNode(col_i))
        causal_graph_true = GeneralGraph(list_nodes)
        for col_i in param_matrix.columns:
            for col_j in param_matrix.columns:
                if (param_matrix[col_j].loc[col_i] != 0) and (param_matrix[col_i].loc[col_j] != 0):
                    causal_graph_true.add_edge(Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
                elif param_matrix[col_j].loc[col_i] != 0:
                    causal_graph_true.add_edge(Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))
                elif param_matrix[col_i].loc[col_j] != 0:
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
            param_data = param_data + np.random.uniform(-0.000001, 0.000001, size=param_data.shape)

            nbcb = NBCBe(param_data, param_tau_max, param_sig_level, model="linear", indtest="linear",
                         cond_indtest="linear")
            nbcb.run()
            causal_graph_hat = nbcb.causal_graph
        elif param_method == "CBNB_e":
            param_data = param_data + np.random.uniform(-0.000001, 0.000001, size=param_data.shape)

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
            param_data = param_data + np.random.uniform(-0.000001, 0.000001, size=param_data.shape)
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
