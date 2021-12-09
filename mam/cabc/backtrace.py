import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from mam.utils import (
    INF_REPLACEMENT,
    InputMessages,
    OutputMessages,
    normalize_and_clip,
    pad,
)


class BacktracerWiring(NamedTuple):
    """BacktracerWiring.

    Args:
        elastic_graph_pools_indices: Array of shape (n_elastic_graphs, n_pools_per_elastic_graph)
            for concatenated wirings, and None for individual wiring
            Indices of pools, ranging from 0 to n_total_pools - 1, that correspond
            to a particular elastic_graph
        nodes_frcs: Array of shape (n_nodes, pool_size**2, 3)
            frcs of different nodes in the elastic_graphs
        nodes_factor_masks: Array of shape (n_nodes, n_connected_factors)
            Boolean array specifying connected factors mask
        nodes_indices: Array of shape (n_factor_configurations, 2)
            Indices of the two connected nodes for each configuration in the factors
        nodes_factor_indices: Array of shape (n_factor_configurations, 2)
            Indices of the factor w.r.t. to the connected nodes
        perturbation_configurations: Array of shape (n_factor_configurations, 2)
            A flat list of configurations from the n_factors factors
        perturb_radiuses: Array of shape (n_factors,)
            Perturb radius at each given factor
        n_nodes: Array of shape (n_elastic_graphs,). Number of nodes for each instance
    """

    elastic_graph_pools_indices: Optional[Union[np.ndarray, jnp.ndarray]]
    nodes_frcs: Union[np.ndarray, jnp.ndarray]
    nodes_factor_masks: Union[np.ndarray, jnp.ndarray]
    nodes_indices: Union[np.ndarray, jnp.ndarray]
    nodes_factor_indices: Union[np.ndarray, jnp.ndarray]
    perturbation_configurations: Union[np.ndarray, jnp.ndarray]
    perturb_radiuses: Union[np.ndarray, jnp.ndarray]
    n_nodes: Union[np.ndarray, jnp.ndarray]


def get_perturbation_configurations(perturb_radius: int, pool_size: int):
    locs = np.array(list(itertools.product(np.arange(pool_size), repeat=2)))
    configurations = (
        locs[:, None]
        + np.array(
            list(
                itertools.product(
                    np.arange(-perturb_radius, perturb_radius + 1), repeat=2
                )
            )
        )[None]
    )
    perturbation_configurations_list = []
    for ii in range(locs.shape[0]):
        valid_configurations = configurations[ii][
            np.logical_and(
                np.all(configurations[ii] >= 0, axis=1),
                np.all(configurations[ii] < pool_size, axis=1),
            )
        ]
        perturbation_configurations_list.append(
            np.stack(
                [
                    np.full(
                        valid_configurations.shape[0],
                        fill_value=np.ravel_multi_index(
                            locs[ii], (pool_size, pool_size)
                        ),
                        dtype=int,
                    ),
                    np.ravel_multi_index(
                        valid_configurations.T, (pool_size, pool_size)
                    ),
                ],
                axis=1,
            )
        )

    perturbation_configurations = np.concatenate(
        perturbation_configurations_list, axis=0
    )
    return perturbation_configurations


def get_backtracer_wiring_from_elastic_graph(
    elastic_graph: nx.Graph,
    pool_size: int,
    perturbation_configurations_dict: Optional[Dict[int, np.ndarray]] = None,
) -> BacktracerWiring:
    """get_backtracer_wiring_from_elastic_graph.

    Args:
        elastic_graph: Nodes range from 0 to n_nodes
            Nodes have attribute frc, which contains the frcs of the nodes.
            Edges have attribute perturb_radius, which contains the perturb radiuses
            of different lateral interactions.
        pool_size: pool_size

    Returns:
        BacktracerWiring
    """
    if perturbation_configurations_dict is None:
        perturb_radiuses_list = list(
            set(nx.get_edge_attributes(elastic_graph, 'perturb_radius').values())
        )
        perturbation_configurations_dict = {
            perturb_radius: get_perturbation_configurations(perturb_radius, pool_size)
            for perturb_radius in perturb_radiuses_list
        }

    n_connected_factors = np.max(list(dict(elastic_graph.degree).values()))
    nodes_frcs = (
        np.array(
            [
                elastic_graph.nodes[node]['frc']
                for node in range(elastic_graph.number_of_nodes())
            ]
        )[:, None]
        + np.concatenate(
            [
                np.zeros((pool_size ** 2, 1), dtype=int),
                np.array(list(itertools.product(np.arange(pool_size), repeat=2))),
            ],
            axis=1,
        )[None]
    )
    nodes_factor_masks = np.zeros(
        (elastic_graph.number_of_nodes(), n_connected_factors), dtype=bool
    )
    for node in elastic_graph.nodes:
        nodes_factor_masks[node, : elastic_graph.degree[node]] = True
        for neighbor_idx, neighbor in enumerate(elastic_graph.neighbors(node)):
            elastic_graph.edges[(node, neighbor)][f'idx_for_{node}'] = neighbor_idx

    nodes_indices = np.array(list(elastic_graph.edges()))
    nodes_factor_indices = np.array(
        [
            [
                elastic_graph.edges[tuple(edge)][f'idx_for_{edge[0]}'],
                elastic_graph.edges[tuple(edge)][f'idx_for_{edge[1]}'],
            ]
            for edge in nodes_indices
        ]
    )
    perturb_radiuses = np.array(
        [elastic_graph.edges[tuple(edge)]['perturb_radius'] for edge in nodes_indices]
    )
    n_perturbation_configurations = np.array(
        [
            perturbation_configurations_dict[perturb_radius].shape[0]
            for perturb_radius in perturb_radiuses
        ]
    )
    nodes_indices = np.repeat(nodes_indices, n_perturbation_configurations, axis=0)
    nodes_factor_indices = np.repeat(
        nodes_factor_indices, n_perturbation_configurations, axis=0
    )
    perturbation_configurations = np.concatenate(
        [
            perturbation_configurations_dict[perturb_radius]
            for perturb_radius in perturb_radiuses
        ],
        axis=0,
    )
    n_nodes = np.array([elastic_graph.number_of_nodes()])
    wiring = BacktracerWiring(
        elastic_graph_pools_indices=None,
        nodes_frcs=nodes_frcs,
        nodes_factor_masks=nodes_factor_masks,
        nodes_indices=nodes_indices,
        nodes_factor_indices=nodes_factor_indices,
        perturbation_configurations=perturbation_configurations,
        perturb_radiuses=perturb_radiuses,
        n_nodes=n_nodes,
    )
    return wiring


def concatenate_backtracer_wiring(
    wiring_loc_list: List[Tuple[BacktracerWiring, np.ndarray]],
) -> BacktracerWiring:
    """concatenate_backtracer_wiring.

    Args:
        wiring_loc_list: List of wiring and loc tuples. loc is an array of
            shape (3,), with first element being 0 and last 2 elements being
            dr and dc

    Returns:
        concatenated_wiring: BacktracerWiring
    """
    n_nodes_cumsum = np.cumsum(
        [0] + [wiring_loc[0].n_nodes for wiring_loc in wiring_loc_list]
    )
    min_nodes_frcs_list = [
        np.concatenate(
            [
                np.array([0]),
                np.min(wiring_loc[0].nodes_frcs[..., 1:].reshape((-1, 2)), axis=0),
            ]
        )
        for wiring_loc in wiring_loc_list
    ]

    def concatenate_fun(x):
        if x[0] is None:
            return None

        return np.concatenate(x, axis=0)

    n_connected_factors = np.max(
        [wiring_loc[0].nodes_factor_masks.shape[1] for wiring_loc in wiring_loc_list]
    )
    updated_wiring_list = [
        wiring_loc[0]._replace(
            nodes_indices=wiring_loc[0].nodes_indices + n_nodes_cumsum[ii],
            nodes_frcs=wiring_loc[0].nodes_frcs
            - min_nodes_frcs_list[ii]
            + wiring_loc[1],
            nodes_factor_masks=np.pad(
                wiring_loc[0].nodes_factor_masks,
                (
                    (0, 0),
                    (
                        0,
                        n_connected_factors - wiring_loc[0].nodes_factor_masks.shape[1],
                    ),
                ),
                mode='constant',
                constant_values=False,
            ),
        )
        for ii, wiring_loc in enumerate(wiring_loc_list)
    ]
    concatenated_wiring = BacktracerWiring(
        **{
            field: concatenate_fun(
                [getattr(wiring, field) for wiring in updated_wiring_list]
            )
            for field in updated_wiring_list[0]._fields
        }
    )._replace(
        elastic_graph_pools_indices=pad(
            [
                np.arange(n_nodes_cumsum[idx], n_nodes_cumsum[idx + 1])
                for idx in range(len(wiring_loc_list))
            ],
            -1,
        )
    )
    concatenated_wiring = concatenated_wiring._replace(
        nodes_frcs=concatenated_wiring.nodes_frcs.astype(np.int16, copy=False),
        nodes_factor_masks=concatenated_wiring.nodes_factor_masks,
        nodes_indices=concatenated_wiring.nodes_indices.astype(np.int16, copy=False),
        nodes_factor_indices=concatenated_wiring.nodes_factor_indices.astype(
            np.int8, copy=False
        ),
        perturbation_configurations=concatenated_wiring.perturbation_configurations.astype(
            np.int16, copy=False
        ),
    )
    return concatenated_wiring


def update_messages_outgoing(
    messages_incoming: jnp.ndarray, messages_unary: jnp.ndarray
) -> jnp.ndarray:
    """update_messages_outgoing.

    Args:
        messages_incoming: Array of shape (n_nodes, n_connected_factors, pool_size**2)
            Incoming messages from n_connected factors to pool_size**2 states
        messages_unary: Array of shape (n_nodes, pool_size**2)
            Unary messages

    Returns:
        messages_outgoing: Array of shape (n_nodes, n_connected_factors, pool_size**2)
            Outgoing messages from pool_size**2 states to n_connected factors
    """
    messages_outgoing = (
        jnp.sum(messages_incoming, axis=-2, keepdims=True)
        + messages_unary[..., None, :]
        - messages_incoming
    )
    return messages_outgoing


def update_messages_incoming(
    messages_outgoing: jnp.ndarray,
    nodes_factor_masks: jnp.ndarray,
    nodes_indices: jnp.ndarray,
    nodes_factor_indices: jnp.ndarray,
    perturbation_configurations: jnp.ndarray,
):
    """update_messages_incoming.

    Args:
        messages_outgoing: Array of shape (n_nodes, n_connected_factors, pool_size**2)
        nodes_factor_masks: Array of shape (n_nodes, n_connected_factors)
            Boolean array specifying connected factors mask
        nodes_indices: Array of shape (n_factor_configurations, 2)
        nodes_factor_indices: Array of shape (n_factor_configurations, 2)
        perturbation_configurations: Array of shape (n_factor_configurations, 2)

    Returns:
        messages_incoming: Array of shape (n_nodes, n_connected_factors, pool_size**2)
    """
    messages_incoming = (
        jnp.full(shape=messages_outgoing.shape, fill_value=-INF_REPLACEMENT)
        .at[
            nodes_indices[..., jnp.array([1, 0], dtype=jnp.int32)],
            nodes_factor_indices[..., jnp.array([1, 0], dtype=jnp.int32)],
            perturbation_configurations[..., jnp.array([1, 0], dtype=jnp.int32)],
        ]
        .max(
            messages_outgoing[
                nodes_indices,
                nodes_factor_indices,
                perturbation_configurations,
            ]
        )
    )
    messages_incoming = normalize_and_clip(
        jnp.where(nodes_factor_masks[..., None], messages_incoming, 0.0)
    )
    return messages_incoming


class BacktracerInternalMessages(NamedTuple):
    """BacktracerInternalMessages.
    Args:
        incoming: Array of shape (n_nodes, n_connected_factors, pool_size**2)
    """

    incoming: Union[np.ndarray, jnp.ndarray]


class BacktracerMessages(NamedTuple):
    input: InputMessages
    output: OutputMessages
    internal: BacktracerInternalMessages


@dataclass
class Backtracer:
    damping: float

    def __post_init__(self):
        pass

    def initialize_messages(
        self,
        shape: Tuple[int, int, int],
        wiring: BacktracerWiring,
        post_init: Optional[Callable] = None,
        context: Optional[Any] = None,
    ):
        if post_init is None:

            def identity_func(messages, context):
                return messages

            post_init = identity_func

        messages = jax.device_put(
            post_init(
                BacktracerMessages(
                    input=InputMessages(
                        from_top=0.0,
                        from_bottom=np.zeros(shape),
                    ),
                    output=OutputMessages(
                        to_top=0.0,
                        to_bottom=np.zeros(wiring.nodes_frcs.shape[:-1]),
                    ),
                    internal=BacktracerInternalMessages(
                        incoming=np.zeros(
                            (
                                wiring.nodes_frcs.shape[0],
                                wiring.nodes_factor_masks.shape[1],
                                wiring.nodes_frcs.shape[1],
                            )
                        ),
                    ),
                ),
                context,
            )
        )
        return messages

    def make_messages_update_func(self):
        def update_messages(
            messages: BacktracerMessages,
            wiring: BacktracerWiring,
        ) -> BacktracerMessages:
            messages_unary = normalize_and_clip(
                messages.input.from_bottom[
                    wiring.nodes_frcs[..., 0],
                    wiring.nodes_frcs[..., 1],
                    wiring.nodes_frcs[..., 2],
                ]
            )
            messages_outgoing = update_messages_outgoing(
                messages.internal.incoming, messages_unary
            )
            messages_incoming = update_messages_incoming(
                messages_outgoing,
                wiring.nodes_factor_masks,
                wiring.nodes_indices,
                wiring.nodes_factor_indices,
                wiring.perturbation_configurations,
            )
            messages = jax.tree_util.tree_multimap(
                lambda a, b: self.damping * a + (1 - self.damping) * b,
                messages,
                messages._replace(
                    output=messages.output._replace(
                        to_bottom=jnp.sum(messages_incoming, axis=-2),
                    ),
                    internal=BacktracerInternalMessages(incoming=messages_incoming),
                ),
            )
            return messages

        return update_messages

    def make_infer_func(self):
        update_messages = self.make_messages_update_func()

        def infer(
            messages: BacktracerMessages,
            wiring: BacktracerWiring,
            n_bp_iter: int,
        ) -> BacktracerMessages:
            @jax.remat
            def bp_update(messages, x):
                return (update_messages(messages, wiring), None)

            messages, _ = jax.lax.scan(
                bp_update,
                messages,
                None,
                n_bp_iter,
            )
            return messages

        return infer


@jax.partial(
    jnp.vectorize,
    excluded=(1, 2, 3),
    signature='(n)->()',
)
def do_recounting_for_instances_single_feature(
    inst_indices: jnp.ndarray,
    all_backtraced_locs: jnp.ndarray,
    evidences: jnp.ndarray,
    overlap_penalty: float = 0.0,
):
    """do_recounting_for_instances_single_feature.

    Args:
        inst_indices (jnp.ndarray): Array of shape (n_instances,)
        all_backtraced_locs: Array of shape (n_elastic_graphs, n_pools_per_elastic_graph, n_children, 3)
        evidences (jnp.ndarray): Array of shape (1, M, N, 2)
        overlap_penalty: penalty for having overlap between multiple instances

    Returns:
        recount_score: Recount score for the combination of instances
    """
    backtraced_locs = all_backtraced_locs[inst_indices]
    recount_score = jnp.sum(
        jnp.zeros((evidences.shape[0], evidences.shape[1] + 1, evidences.shape[2] + 1))
        .at[backtraced_locs[..., 0], backtraced_locs[..., 1], backtraced_locs[..., 2]]
        .set(
            evidences[
                backtraced_locs[..., 0],
                backtraced_locs[..., 1],
                backtraced_locs[..., 2],
                1,
            ]
        )[:, :-1, :-1]
    )
    n_occurences = (
        jnp.zeros((evidences.shape[0], evidences.shape[1] + 1, evidences.shape[2] + 1))
        .at[backtraced_locs[..., 0], backtraced_locs[..., 1], backtraced_locs[..., 2]]
        .add(1.0)[:, :-1, :-1]
    )
    recount_score = recount_score + jnp.sum(
        jnp.where(n_occurences > 1, (n_occurences - 1) * overlap_penalty, 0)
    )
    return recount_score


def do_recounting_single_feature(
    inst_indices: jnp.ndarray,
    evidences: jnp.ndarray,
    messages: BacktracerMessages,
    wiring: BacktracerWiring,
    features_description: jnp.ndarray,
    overlap_penalty: float = 0.0,
):
    """do_recounting_single_feature.

    Args:
        inst_indices (jnp.ndarray): Array of shape (..., n_instances)
            Calculate recount scores for a list of combinations of multiple instances
        evidences (jnp.ndarray): Array of shape (1, M, N, 2)
        messages: BacktracerMessages
        wiring: BacktracerWiring
        features_description: features_description in forward_pass.ORLayerWiring

    Returns:
        recount_scores: Array of shape inst_indices.shape[:-1]
            Recount scores for each combination of instances in the list
    """
    features_beliefs = (
        messages.output.to_bottom
        + messages.input.from_bottom[
            wiring.nodes_frcs[..., 0],
            wiring.nodes_frcs[..., 1],
            wiring.nodes_frcs[..., 2],
        ]
    )
    frcs_all_instances = wiring.nodes_frcs[
        jnp.arange(features_beliefs.shape[0]), jnp.argmax(features_beliefs, axis=1)
    ]
    all_backtraced_locs = jnp.where(
        wiring.elastic_graph_pools_indices[..., None, None] == -1,
        -1,
        frcs_all_instances[wiring.elastic_graph_pools_indices][..., None, :]
        + features_description[0, :, :-1],
    )
    recount_scores = do_recounting_for_instances_single_feature(
        inst_indices, all_backtraced_locs, evidences, overlap_penalty
    )
    return recount_scores, all_backtraced_locs
