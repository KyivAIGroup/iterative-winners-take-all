from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull


def plot_assemblies(assemblies, n_hidden=2, labels=None, title=None):
    # (N, samples)
    if isinstance(assemblies, np.ndarray):
        assemblies = [vec.nonzero()[0] for vec in assemblies.T]
    hidden_start = np.concatenate(assemblies).max() + 1

    if len(assemblies) < 2:
        # do nothing
        return
    if labels is None:
        labels = [None for _ in assemblies]
    graph = nx.Graph()
    node_labels = defaultdict(list)
    for class_id, assembly in enumerate(assemblies, start=1):
        hidden_idx = range(hidden_start, hidden_start + n_hidden)
        hidden_start += n_hidden
        for node in assembly:
            node_labels[node].append(str(class_id))
        assembly = np.r_[assembly, hidden_idx]
        graph.add_nodes_from(assembly)
        graph.add_edges_from(combinations(assembly, 2))
    node_labels = {node: ','.join(node_labels[node]) for node in graph.nodes}
    pos = nx.spring_layout(graph, iterations=50)
    cmap = plt.cm.get_cmap("hsv", len(assemblies) + 1)  # +1 is necessary
    colors = np.array([cmap(i) for i in range(len(assemblies))])

    fig, ax = plt.subplots()
    for class_id, assembly in enumerate(assemblies):
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=assembly,
                               node_color=[colors[class_id]], alpha=0.2,
                               label=labels[class_id], ax=ax)
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels,
                            font_size=6, alpha=0.7, ax=ax)
    ax.set_title(title)
    ax.legend()

    nodes, locations = list(zip(*pos.items()))
    locations = np.array(locations)
    argsort = np.argsort(nodes)
    locations = locations[argsort]
    nodes = np.take(nodes, argsort)
    for class_id, assembly in enumerate(assemblies):
        if len(assembly) < 3:
            # not enough points to construct a convex hull
            continue
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
        ax.add_patch(poly)

    return ax
