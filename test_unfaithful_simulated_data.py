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

import random


def uniform_with_gap(min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    while True:
        r = random.uniform(min_value, max_value)
        if min_gap > r or max_gap<r:
            break
    return r


def unfaithful_diamond_linear_generator(N=1000, seed=0, verbose=False, gaussian=True):
    N=N+3
    random.seed(seed)
    ax = uniform_with_gap()
    bx = 0.1
    ay = uniform_with_gap()
    by = 0.1
    aw = uniform_with_gap()
    bw = 0.1
    az = uniform_with_gap()
    bz = 0.1

    axy = uniform_with_gap()
    axw = uniform_with_gap()
    ayz = uniform_with_gap()
    awz = -axy*ayz/axw

    if verbose:
        print("Unfaithful Diamond: 3 <- 1 <- 0 -> 2 -> 3")
        print(ax, bx, ay,by, aw, bw, az, bz, axy, axw, ayz, awz)

    if gaussian:
        epsx = np.random.randn(N)
        epsy = np.random.randn(N)
        epsw = np.random.randn(N)
        epsz = np.random.randn(N)
    else:
        epsx = np.random.uniform(-1, 1, N)
        epsy = np.random.uniform(-1, 1, N)
        epsw = np.random.uniform(-1, 1, N)
        epsz = np.random.uniform(-1, 1, N)

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])
    z = np.zeros([N])

    for i in range(0, N):
        x[i] = bx * epsx[i]
        y[i] = axy * x[i] + by * epsy[i]
        w[i] = axw * x[i] + bw * epsw[i]
        z[i] = ayz * y[i] + awz * w[i] + bz * epsz[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])
    z = pd.DataFrame(z, columns=["V4"])

    series = pd.concat([x, y, w, z], axis=1, sort=False)
    series = series.drop(series.index[[0, 1, 2]])
    series = series.reset_index(drop=True)
    return series


if __name__ == '__main__':
    param_method = "PCGCE"  # CBNB_w NBCB_w CBNB_e NBCB_e GCMVL CCM PCMCI PCGCE VarLiNGAM
    param_tau_max = 5
    param_sig_level = 0.05
    param_nb_obervation = 1000
    param_generator_name = "diamond"  # diamond
    param_gaussian = True

    f1_adjacency_list = []
    f1_orientation_list = []
    percentage_of_detection_skeleton = 0
    for i in range(100):
        print("######################################### iter " + str(i) + " #########################################")
        if param_gaussian:
            directory_name = "unfaithful_" + param_generator_name + "_g"
        else:
            directory_name = "unfaithful_" + param_generator_name + "_ng"
        param_data = pd.read_csv("./data/simulated_data/" + directory_name + "/dataset_" + str(i) + ".csv", index_col=0)
        # if param_generator_name == "diamond":
        #     param_data = unfaithful_diamond_linear_generator(param_nb_obervation, gaussian=param_gaussian)
        # else:
        #     param_data = None
        # param_data.to_csv("./data/simulated_data/" + directory_name + "/dataset_" + str(i) + ".csv")

        # param_data = param_data.iloc[:param_nb_obervation]

        list_nodes = []
        for col_i in param_data.columns:
            list_nodes.append(GraphNode(col_i))
        causal_graph_true = GeneralGraph(list_nodes)
        if param_generator_name == "diamond":
            causal_graph_true.add_edge(Edge(GraphNode("V1"), GraphNode("V2"), Endpoint.TAIL, Endpoint.ARROW))
            causal_graph_true.add_edge(Edge(GraphNode("V1"), GraphNode("V3"), Endpoint.TAIL, Endpoint.ARROW))
            causal_graph_true.add_edge(Edge(GraphNode("V2"), GraphNode("V4"), Endpoint.TAIL, Endpoint.ARROW))
            causal_graph_true.add_edge(Edge(GraphNode("V3"), GraphNode("V4"), Endpoint.TAIL, Endpoint.ARROW))
        else:
            causal_graph_true = None

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

        if causal_graph_true.get_graph_edges() == causal_graph_hat.get_graph_edges():
            percentage_of_detection_skeleton = percentage_of_detection_skeleton + 1
        print("Percentage so far with true graph= " + str(percentage_of_detection_skeleton) + "/" + str(i + 1))

    print("#############################################")
    print("F1 adjacency with desired graph= " + str(np.mean(f1_adjacency_list)) + " +- " + str(np.var(f1_adjacency_list)))
    print("F1 orientation with desired graph= " + str(np.mean(f1_orientation_list)) + " +- " + str(np.var(f1_orientation_list)))
    print("Percentage of detection with desired graph= " + str(percentage_of_detection_skeleton/100))
