from itertools import combinations
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_intersection(assemblies, labels=None, title=None):
    # (N, samples)
    if isinstance(assemblies, np.ndarray):
        assemblies = [vec.nonzero()[0] for vec in assemblies.T]
    if len(assemblies) < 2:
        # do nothing
        return
    if labels is None:
        labels = [None for _ in assemblies]
    graph = nx.Graph()
    node_to_class = defaultdict(list)
    for class_id, assembly in enumerate(assemblies, start=1):
        for node in assembly:
            node_to_class[node].append(str(class_id))
        graph.add_nodes_from(assembly)
        graph.add_edges_from(combinations(assembly, 2))
    node_to_class = {node: ','.join(node_to_class[node]) for node in graph.nodes}
    pos = nx.spring_layout(graph)
    cmap = plt.cm.get_cmap("hsv", len(assemblies) + 1)  # +1 is necessary
    plt.figure()
    for class_id, assembly in enumerate(assemblies):
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=assembly,
                               node_color=[cmap(class_id)], alpha=0.2,
                               label=labels[class_id])
    nx.draw_networkx_labels(graph, pos=pos, labels=node_to_class, font_size=6, alpha=0.7)
    plt.title(title)
    plt.legend()
