import copy
from typing import Dict, FrozenSet, List, Tuple

import jax
import networkx as nx
import numpy as np


def get_interaction_graph_from_feature_activations(
    feature_activations: np.ndarray,
    pools_to_laterals_list: List[
        List[Dict[FrozenSet[Tuple[int, int, int, int]], np.ndarray]]
    ],
    templates_list: List[List[List[int]]],
    subset_laterals: bool = True,
) -> nx.Graph:
    """get_interaction_graph_from_feature_activations.

    Args:
        feature_activations: Array of shape (n_feature_activations, 3)
            frcs of the feature activations
        pools_to_laterals_list: A complete list of pool definitions at each feature
            len(pools_to_laterals_list) == n_features
            len(pools_to_laterals_list[ii]) == n_pools for feature ii
            Each pool is represented as a dictionary
            The keys are symmetric representations of the lateral, using frozenset
            The values are the actual lateral (in terms of connected feature idx, dr and dc)
            For a lateral connecting feature ii and jj with a displacement
            (jj related to ii) dr, dc, the corresponding frozen set would be
            {(ii, jj, dr, dc), (jj, ii, -dr, -dc)}
        templates_list: templates_list
            List of templates for the different features.
            len(templates_list) == n_features
            len(templates_list[ii]) == n_templates for feature ii
            A template is represented as a list of integers, representing indices of
            the involved pools for that template

    Returns:
        interaction_graph: The interaction graph specifying the lateral layer
            Nodes are specified in (f, r, c) tuples.
            For each node, there exist node attributes:
                is_within_boundary (bool): Indicating whether the variable is within boundary
                    True is the node is within boundary
                    False or missing means the node is outside boundary
                templates (List[List[List[np.ndarray]]]): List of configurations (with pooling) for each node
                    First dimension ranges over different templates
                    Second dimension ranges over different pools within a template
                    Third dimension ranges over different laterals within a pool. Each lateral is encoded using
                    the frc of its connected node, in an np array of length 3
            Edges are of the form ((f0, r0, c0), (f1, r1, c1)).
            For each edge, there exist edge attributes:
                idx (int): The flat index of the corresponding edge
                count (int): Number of times an edge appears as part of a template in a node
                    For edges connecting nodes within boundary this should always be 2
                    For edges connecting one node within boundary and one boundary node, this should be 1
                    Used to decide boundary_laterals_indices and boundary_laterals_sides_indices
                sides (dict): A dictionary mapping connected nodes to sides (0 or 1)
    """
    n_features = len(pools_to_laterals_list)
    subset_pools_to_laterals_list = [
        copy.copy(pools_to_laterals) for pools_to_laterals in pools_to_laterals_list
    ]
    if subset_laterals:
        all_possible_laterals_from_feature_activations = set(
            [
                frozenset(
                    [
                        (feature_activations[ii, 0], feature_activations[jj, 0])
                        + tuple(
                            feature_activations[jj, 1:] - feature_activations[ii, 1:]
                        ),
                        (feature_activations[jj, 0], feature_activations[ii, 0])
                        + tuple(
                            feature_activations[ii, 1:] - feature_activations[jj, 1:]
                        ),
                    ]
                )
                for ii in range(feature_activations.shape[0] - 1)
                for jj in range(ii + 1, feature_activations.shape[0])
            ]
        )
        for feature_idx in range(n_features):
            for pool_idx in range(len(subset_pools_to_laterals_list[feature_idx])):
                subset_pools_to_laterals_list[feature_idx][pool_idx] = [
                    subset_pools_to_laterals_list[feature_idx][pool_idx][key]
                    for key in set(
                        list(
                            subset_pools_to_laterals_list[feature_idx][pool_idx].keys()
                        )
                    ).intersection(all_possible_laterals_from_feature_activations)
                ]

    subset_templates_list = [
        [
            [
                subset_pools_to_laterals_list[feature_idx][pool_idx]
                for pool_idx in template
            ]
            for template in templates_list[feature_idx]
        ]
        for feature_idx in range(n_features)
    ]
    interaction_graph = nx.Graph()
    interaction_graph.add_nodes_from(
        [tuple(frc) for frc in feature_activations], is_within_boundary=True
    )
    nodes_list = list(interaction_graph.nodes())
    for node in nodes_list:
        connected_nodes_list = list(
            set(
                jax.tree_util.tree_map(
                    lambda x: (x[0],) + tuple(np.array(node[1:]) + x[1:]),
                    jax.tree_util.tree_leaves(subset_templates_list[node[0]]),
                )
            )
        )
        templates = jax.tree_util.tree_map(
            lambda x: np.array((x[0],) + tuple(np.array(node[1:]) + x[1:])),
            subset_templates_list[node[0]],
        )
        interaction_graph.nodes()[node]['templates'] = templates
        for connected_node in connected_nodes_list:
            interaction_graph.add_edge(
                node,
                connected_node,
                count=interaction_graph.edges()
                .get((node, connected_node), {})
                .get('count', 0)
                + 1,
            )

    for idx, edge in enumerate(interaction_graph.edges()):
        if interaction_graph.nodes()[edge[0]].get(
            'is_within_boundary', False
        ) and interaction_graph.nodes()[edge[1]].get('is_within_boundary', False):
            assert interaction_graph.edges()[edge]['count'] == 2, (
                edge,
                interaction_graph.edges()[edge]['count'],
            )

        interaction_graph.edges()[edge].update(
            {
                'idx': idx,
                'sides': {edge[0]: 0, edge[1]: 1},
            }
        )

    return interaction_graph
