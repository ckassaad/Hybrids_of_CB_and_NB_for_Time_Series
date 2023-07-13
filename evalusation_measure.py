from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

def precision(graph_hat, graph_true, ignore_orientation=False):
    if not ignore_orientation:
        list_edge_g1 = []
        for edge in graph_hat.get_graph_edges():
            # print(edge.get_node1(), edge.get_endpoint1(), edge.get_endpoint2(), edge.get_node2())
            # if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
            if edge.get_endpoint2() == Endpoint.ARROW:
                # list_edge_g1.append(str(edge.get_node1()) + str(edge.get_endpoint1()) + str(edge.get_endpoint2()) + str(edge.get_node2()))
                list_edge_g1.append(str(edge.get_node1()) + str(edge.get_endpoint2()) + str(edge.get_node2()))
            if edge.get_endpoint1() == Endpoint.ARROW:
                list_edge_g1.append(str(edge.get_node2()) + str(edge.get_endpoint1()) + str(edge.get_node1()))
        list_edge_g2 = []
        for edge in graph_true.get_graph_edges():
            # print(edge.get_node1(), edge.get_endpoint1(), edge.get_endpoint2(), edge.get_node2())
            # if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
            if edge.get_endpoint2() == Endpoint.ARROW:
                # list_edge_g2.append(str(edge.get_node1()) + str(edge.get_endpoint1()) + str(edge.get_endpoint2()) + str(edge.get_node2()))
                list_edge_g2.append(str(edge.get_node1()) + str(edge.get_endpoint2()) + str(edge.get_node2()))
            if edge.get_endpoint1() == Endpoint.ARROW:
                list_edge_g2.append(str(edge.get_node2()) + str(edge.get_endpoint1()) + str(edge.get_node1()))
        # print(list_edge_g1)
        # print(list_edge_g2)

        tp = len(list(set(list_edge_g1) & set(list_edge_g2)))
        fp = len(list(set(list_edge_g1) - set(list_edge_g2)))
    else:
        list_edge_g1 = []
        for edge in graph_hat.get_graph_edges():
            if (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.CIRCLE) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.CIRCLE) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.CIRCLE):
                list_edge_g1.append(
                    str(edge.get_node1()) + str(Endpoint.TAIL) + str(Endpoint.TAIL) + str(edge.get_node2()))
                list_edge_g1.append(
                    str(edge.get_node2()) + str(Endpoint.TAIL) + str(Endpoint.TAIL) + str(edge.get_node1()))
        list_edge_g2 = []
        for edge in graph_true.get_graph_edges():
            if (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.CIRCLE) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.CIRCLE) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.CIRCLE):
                list_edge_g2.append(
                    str(edge.get_node1()) + str(Endpoint.TAIL) + str(Endpoint.TAIL) + str(edge.get_node2()))
                list_edge_g2.append(
                    str(edge.get_node2()) + str(Endpoint.TAIL) + str(Endpoint.TAIL) + str(edge.get_node1()))

        tp = int(len(list(set(list_edge_g1) & set(list_edge_g2))) / 2)
        fp = int(len(list(set(list_edge_g1) - set(list_edge_g2))) / 2)
    if (tp == 0) and (fp == 0):
        p = 0
    else:
        p = tp / (tp + fp)
    return p


def recall(graph_hat, graph_true, ignore_orientation=False):
    if not ignore_orientation:
        list_edge_g1 = []
        for edge in graph_hat.get_graph_edges():
            # if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
            if edge.get_endpoint2() == Endpoint.ARROW:
                # list_edge_g1.append(str(edge.get_node1()) + str(edge.get_endpoint1()) + str(edge.get_endpoint2()) + str(edge.get_node2()))
                list_edge_g1.append(str(edge.get_node1()) + str(edge.get_endpoint2()) + str(edge.get_node2()))
            if edge.get_endpoint1() == Endpoint.ARROW:
                list_edge_g1.append(str(edge.get_node2()) + str(edge.get_endpoint1()) + str(edge.get_node1()))
        list_edge_g2 = []
        for edge in graph_true.get_graph_edges():
            # if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
            if edge.get_endpoint2() == Endpoint.ARROW:
                # list_edge_g2.append(str(edge.get_node1()) + str(edge.get_endpoint1()) + str(edge.get_endpoint2()) + str(edge.get_node2()))
                list_edge_g2.append(str(edge.get_node1()) + str(edge.get_endpoint2()) + str(edge.get_node2()))
            if edge.get_endpoint1() == Endpoint.ARROW:
                # list_edge_g2.append(str(edge.get_node1()) + str(edge.get_endpoint1()) + str(edge.get_endpoint2()) + str(edge.get_node2()))
                list_edge_g2.append(str(edge.get_node2()) + str(edge.get_endpoint1()) + str(edge.get_node1()))

        tp = len(list(set(list_edge_g1) & set(list_edge_g2)))
        fn = len(list(set(list_edge_g2) - set(list_edge_g1)))
    else:
        list_edge_g1 = []
        for edge in graph_hat.get_graph_edges():
            if (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.CIRCLE) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.CIRCLE) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.CIRCLE):
                list_edge_g1.append(
                    str(edge.get_node1()) + str(Endpoint.TAIL) + str(Endpoint.TAIL) + str(edge.get_node2()))
                list_edge_g1.append(
                    str(edge.get_node2()) + str(Endpoint.TAIL) + str(Endpoint.TAIL) + str(edge.get_node1()))
        list_edge_g2 = []
        for edge in graph_true.get_graph_edges():
            if (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.ARROW) or \
                    (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.CIRCLE) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.CIRCLE) or \
                    (edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.TAIL) or \
                    (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.CIRCLE):
                list_edge_g2.append(
                    str(edge.get_node1()) + str(Endpoint.TAIL) + str(Endpoint.TAIL) + str(edge.get_node2()))
                list_edge_g2.append(
                    str(edge.get_node2()) + str(Endpoint.TAIL) + str(Endpoint.TAIL) + str(edge.get_node1()))

        tp = int(len(list(set(list_edge_g1) & set(list_edge_g2))) / 2)
        fn = int(len(list(set(list_edge_g2) - set(list_edge_g1))) / 2)
    if (tp == 0) and (fn == 0):
        r = 0
    else:
        r = tp / (tp + fn)
    return r


def f1_score(graph_hat, graph_true, ignore_orientation):
    p = precision(graph_hat, graph_true, ignore_orientation=ignore_orientation)
    r = recall(graph_hat, graph_true, ignore_orientation=ignore_orientation)
    if (p == 0) and (r == 0):
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    return f1
