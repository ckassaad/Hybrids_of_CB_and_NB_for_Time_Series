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

import pandas as pd


if __name__ == '__main__':
    param_method = "CBNB_w"  # CBNB_w NBCB_w CBNB_e NBCB_e GCMVL CCM PCMCI PCGCE VarLiNGAM
    param_tau_max = 5
    param_sig_level = 0.05

    f1_adjacency_list = []
    f1_orientation_list = []
    percentage_of_detection_skeleton = 0

    param_data = pd.read_csv("./data/temperature/temperature.csv", index_col=0)
    param_data.columns = param_data.columns.str.replace(' ', '_')

    list_nodes = []
    for col_i in param_data.columns:
        list_nodes.append(GraphNode(col_i))
    causal_graph_true = GeneralGraph(list_nodes)
    causal_graph_true.add_edge(Edge(GraphNode(param_data.columns[1]), GraphNode(param_data.columns[0]), Endpoint.TAIL, Endpoint.ARROW))

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
        nbcb = NBCBe(param_data, param_tau_max, param_sig_level, model="linear",  indtest="linear", cond_indtest="linear")
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
#
#     if causal_graph_true.get_graph_edges() == causal_graph_hat.get_graph_edges():
#         percentage_of_detection_skeleton = percentage_of_detection_skeleton + 1
#     print("Percentage so far with true graph= " + str(percentage_of_detection_skeleton) + "/" + str(i + 1))
