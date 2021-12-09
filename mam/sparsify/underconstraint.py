from typing import Union

import networkx as nx
import numba as nb
import numpy as np
import scipy.spatial.distance as spdist


@nb.njit(cache=True)
def relax_paths(
    path_lengths,  # : np.array of np.double,
    node1: int,
    node2: int,
    shortcut: Union[float, int],
):
    """Given path_lengths which represent the shortest paths in a graph,
    and a single edge which has been added, update all shortest paths
    to take advantage of the new edge.
    This can be done by noting that for all other nodes i, j, if there
    is a new shortest path that takes advantage of the new edge, it
    either goes i -> node1 -> node2 -> j or i -> node2 -> node1 -> j.
    Thus we can update all shortest paths in O(v^2) time. This is
    significantly more efficient than recalculating all shortest
    paths, and it saves significantly on repeatedly running the
    shortest path between two nodes algorithms.

    Note: path_lengths is modified in place.

    Args:
        path_lengths: A matrix where path_lengths[i,j] is the distance from i to j
        node1: one end of the new edge we added
        node2: the other end of the new edge we added
        shortcut: the length of the new edge
    """
    if node1 > node2:
        node1, node2 = node2, node1
    if path_lengths[node1, node2] <= shortcut:
        return
    path_lengths[node1, node2] = shortcut
    path_lengths[node2, node1] = shortcut

    for i in range(len(path_lengths)):
        for j in range(i + 1, len(path_lengths[i])):
            if i == node1 and j == node2:
                continue

            cost = path_lengths[i][j]
            # we make the assumption that path_lengths[node1][node1] == 0, etc
            cost = min(path_lengths[i, node1] + shortcut + path_lengths[node2, j], cost)
            cost = min(path_lengths[i, node2] + shortcut + path_lengths[node1, j], cost)
            if i == node1:
                cost = min(shortcut + path_lengths[node2, j], cost)
            elif i == node2:
                cost = min(shortcut + path_lengths[node1, j], cost)
            elif j == node1:
                cost = min(shortcut + path_lengths[node2, i], cost)
            elif j == node2:
                cost = min(shortcut + path_lengths[node1, i], cost)

            path_lengths[i, j] = cost
            path_lengths[j, i] = cost


def add_underconstraint_perturb_cxns(
    frcs: np.ndarray,
    max_cxn_length: float,
    tolerance: float = 1.8,
    perturb_factor: float = 7.0,
    min_perturb_radius: int = 1,
):
    """Adds connections between every pair of nodes whose perturbed distance is
    greater than their <spatial distance> * tolerance / perturb_factor

    * The graph is modified in place *

    Args:
        frcs: The frcs of a set of faeture activations
        max_cxn_length: Any nodes separate by more then this distance are ignored
        tolerance: Trade off num edges vs how well the constraint is applied
        perturb_factor: How distance and perturb radius is related.
        min_perturb_radius: Never add an edge with less than this much flexibility
    """
    graph = nx.Graph()
    graph.add_nodes_from(
        [(node_idx, dict(frc=frc)) for node_idx, frc in enumerate(frcs)]
    )
    # We will consider all pairs of nodes whose distance is less than
    # max_cxn_length, excluding self-loops
    locns = frcs[:, 1:]
    pairwise_distances = spdist.squareform(spdist.pdist(locns))
    consider = pairwise_distances <= max_cxn_length
    np.fill_diagonal(consider, 0)  # No reason to add self-loop cxns here
    close_pairs = np.nonzero(consider)
    # Smallest to largest
    remaining_pairs = sorted(
        (
            (source, target, pairwise_distances[source, target])
            for source, target in zip(*close_pairs)
            if source < target
        ),
        key=lambda x: x[2],
    )
    num_nodes = graph.number_of_nodes()
    path_lengths = np.full((num_nodes, num_nodes), np.inf)
    for node1, node2, length in remaining_pairs:
        # If the path length was already too low initially, there's
        # no way adding edges would have changed the situation
        dist = path_lengths[node1][node2]
        target_perturb_dist = max(min_perturb_radius, length / perturb_factor)
        if dist >= target_perturb_dist * tolerance:
            edge_dist = int(np.ceil(target_perturb_dist))
            graph.add_edge(
                node1,
                node2,
                perturb_radius=edge_dist,
                distance=length,
            )
            shortcut = edge_dist
            relax_paths(path_lengths, node1, node2, shortcut)

    return graph


def adjust_perturbation_distances(
    graph: nx.Graph,
    perturb_factor: float,
    min_perturb_radius: int = 1,
):
    """Iterates over all edges and adds a perturb_radius' attribute
    to each edge, or overwrites the attributes if they already exist. The distance
    is Euclidean with no rounding unless specified otherwise, and the perturb radius
    is rounded to an integer.

    Args:
        graph: The graph to modify.
        perturb_factor: Used to compute the perturbation radius based on the distance
            between the nodes. radius = distance / perturb_factorj
        min_perturb_radius: Only used with the perturb factor. Defines the minimum perturb
            radius and overrides the perturb factor's radius.
    """
    num_nodes = graph.number_of_nodes()
    frcs = np.array([graph.nodes[ii]['frc'] for ii in range(num_nodes)])
    locns = frcs[:, 1:]
    pairwise_distances = spdist.squareform(spdist.pdist(locns))
    # Balanced rounding
    total_rounding_error = 0
    # We'll round up or down in a balanced way — e.g., if we rounded down
    # the last few times we're likely to round up this time. If we visit
    # edges in a dfs order, this will roughly balance things across the graph.
    for n1, n2 in nx.edge_dfs(graph):
        distance = pairwise_distances[n1, n2]
        desired_radius = max(min_perturb_radius, distance / perturb_factor)
        upper = int(np.ceil(desired_radius))
        lower = int(np.floor(desired_radius))
        round_up_error = total_rounding_error + upper - desired_radius
        round_down_error = total_rounding_error + lower - desired_radius
        if abs(round_up_error) < abs(round_down_error):
            graph.edges[n1, n2]["perturb_radius"] = upper
            total_rounding_error = round_up_error
        else:
            graph.edges[n1, n2]["perturb_radius"] = lower
            total_rounding_error = round_down_error
