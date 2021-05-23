from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull


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
    node_to_class = {node: ','.join(node_to_class[node])
                     for node in graph.nodes}
    pos = nx.spring_layout(graph, iterations=100)
    cmap = plt.cm.get_cmap("hsv", len(assemblies) + 1)  # +1 is necessary
    colors = np.array([cmap(i) for i in range(len(assemblies))])
    plt.figure()
    for class_id, assembly in enumerate(assemblies):
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=assembly,
                               node_color=[colors[class_id]], alpha=0.2,
                               label=labels[class_id])
    nx.draw_networkx_labels(graph, pos=pos, labels=node_to_class,
                            font_size=6, alpha=0.7)

    nodes, locations = list(zip(*pos.items()))
    locations = np.array(locations)
    argsort = np.argsort(nodes)
    locations = locations[argsort]
    nodes = np.take(nodes, argsort)
    for class_id, assembly in enumerate(assemblies):
        idx = np.searchsorted(nodes, assembly)
        hull = ConvexHull(locations[idx])
        cent = np.mean(hull.points, axis=0)
        hull_points = np.concatenate(hull.points[hull.simplices], axis=0)
        hull_points = hull_points.tolist()
        hull_points.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                          p[0] - cent[0]))
        hull_points = hull_points[0::2]  # Deleting duplicates
        hull_points = 1.2 * (np.array(hull_points) - cent) + cent
        poly = Polygon(hull_points,
                       facecolor=colors[class_id], alpha=0.1,
                       capstyle='round', joinstyle='round')
        plt.gca().add_patch(poly)

    plt.title(title)
    plt.legend()
