from typing import NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from mam.utils import pad

MAX_MSG = 1000.0
INF_REPLACEMENT = 1e6


class MessagesPool2H(NamedTuple):
    """MessagesPool2H.

    Args:
        largest: Array of shape (n_feature_locs, n_templates, n_pools)
        second_largest: Array of shape (n_feature_locs, n_templates, n_pools)
        largest_indices: Array of shape (n_feature_locs, n_templates, n_pools)
    """

    largest: jnp.ndarray
    second_largest: jnp.ndarray
    largest_indices: jnp.ndarray


class MessagesH2Pool(NamedTuple):
    """MessagesH2Pool.

    Args:
        regular: Array of shape (n_feature_locs, n_templates, n_pools)
        special: Array of shape (n_feature_locs, n_templates, n_pools)
        special_indices: Array of shape (n_feature_locs, n_templates, n_pools)
            Same as largest_indices in MessagesPool2H
    """

    regular: jnp.ndarray
    special: jnp.ndarray
    special_indices: jnp.ndarray


def get_messages_pool2h(
    messages_l2h: jnp.ndarray,
    laterals_indices: jnp.ndarray,
    laterals_sides_indices: jnp.ndarray,
) -> MessagesPool2H:
    """get_messages_pool2h.

    Args:
        messsages_l2h: Array of shape (n_laterals, 2)
            For a particulalr lateral i corresponding to the edge (j, k),
            messages_l2h[i, 0] contains messages from lateral i to the HOF
            at feature j, and messages_l2h[i, 1] contains messages from
            lateral i to the HOF at feature k.
        laterals_indices : Array of shape (n_feature_locs, n_templates, n_pools, n_laterals_per_pool)
            Contains the indices of the relevant laterals for the different pools in the different templates.
            -1 is used for padding. For the HOF at a particular feature location, the allowed configurations
            are all off and having exactly one lateral in each pool to be on in each template.
        laterals_sides_indices : Binary array of shape (n_feature_locs, n_templates, n_pools, n_laterals_per_pool)
            0/ means the feature location is on the 0/1 side of the corresponding lateral.
            -1 is used for padding

    Returns:
        messages_pool2h: MessagesPool2H
    """
    top2_values, top2_indices = jax.lax.top_k(
        jnp.concatenate(
            [messages_l2h, jnp.full((1, 2), fill_value=-INF_REPLACEMENT)], axis=0
        )[laterals_indices, laterals_sides_indices],
        2,
    )
    is_padding = jnp.all(laterals_indices < 0, axis=-1)
    n_feature_locs, n_templates, n_pools = laterals_indices.shape[:3]
    messages_pool2h = MessagesPool2H(
        largest=jnp.where(is_padding, 0, top2_values[..., 0]),
        second_largest=top2_values[..., 1],
        largest_indices=jnp.where(
            is_padding,
            -1,
            laterals_indices[
                jnp.arange(n_feature_locs)[:, None, None],
                jnp.arange(n_templates)[None, :, None],
                jnp.arange(n_pools)[None, None, :],
                top2_indices[..., 0],
            ],
        ),
    )
    return messages_pool2h


def update_messages_h2v_for_feature_loc(
    messages_pool2h: MessagesPool2H, messages_f2h: float, logw: jnp.ndarray
) -> Tuple[MessagesH2Pool, float]:
    """update_messages_h2pool.

    Args:
        messages_pool2h: MessagesPool2H
            largest: Array of shape (n_templates, n_pools)
                For padding templates and pools, the corresponding value should be 0
            second_largest: Array of shape (n_templates, n_pools)
                For padding templates and pools, and pools with only 1 lateral,
                the corresponding value should be -INF_REPLACEMENT
            largest_indices: Array of shape (n_templates, n_pools)
                For padding templates and pools, the corresponding index should be -1
        messages_f2h: float
        logw: Array of shape (n_templates,)
            For padding templates, the corresponding value should be -INF_REPLACEMENT

    Returns:
        messages_h2pool: MessagesH2Pool
            regular: Array of shape (n_templates, n_pools)
            special: Array of shape (n_templates, n_pools)
            special_indices: Array of shape (n_templates, n_pools)
                Same as largest_indices in MessagesPool2H
        messages_h2f: float
    """
    unnormalized_factor_log_beliefs_with_largest = (
        jnp.sum(messages_pool2h.largest, axis=1) + messages_f2h + logw
    )
    to_1s = (
        unnormalized_factor_log_beliefs_with_largest[..., None]
        - messages_pool2h.largest
    )
    regular_to_0s = jnp.maximum(
        0.0, jnp.max(unnormalized_factor_log_beliefs_with_largest)
    )
    special_to_0s = jnp.maximum(
        0.0,
        jnp.maximum(
            to_1s + messages_pool2h.second_largest,
            jnp.max(
                jnp.where(
                    jnp.any(
                        messages_pool2h.largest_indices[..., None, None]
                        == messages_pool2h.largest_indices[None, None],
                        axis=-1,
                    ),
                    -INF_REPLACEMENT,
                    unnormalized_factor_log_beliefs_with_largest[None, None],
                ),
                axis=-1,
            ),
        ),
    )
    messages_h2pool = MessagesH2Pool(
        regular=(to_1s - regular_to_0s).clip(-MAX_MSG, MAX_MSG),
        special=(to_1s - special_to_0s).clip(-MAX_MSG, MAX_MSG),
        special_indices=messages_pool2h.largest_indices,
    )
    messages_h2f = jnp.max(
        unnormalized_factor_log_beliefs_with_largest - messages_f2h
    ).clip(-MAX_MSG, MAX_MSG)
    return messages_h2pool, messages_h2f


def update_messages_l2h(
    messages_h2pool: MessagesH2Pool,
    features_pools_indices: jnp.array,
    boundary_laterals_indices: jnp.ndarray,
    boundary_laterals_sides_indices: jnp.ndarray,
    boundary_conditions: float,
) -> jnp.ndarray:
    """update_messages_l2h.

    Args:
        messages_h2pool: MessagesH2Pool
            regular: Array of shape (n_feature_locs, n_templates, n_pools)
            special: Array of shape (n_feature_locs, n_templates, n_pools)
            special_indices: Array of shape (n_feature_locs, n_templates, n_pools)
                Same as largest_indices in MessagesPool2H
        features_pools_indices: Array of shape (n_laterals, 2, n_template_pools, 3)
            The 3 elements in the last dimension are feature_loc_idx, template_idx and pool_idx
            The array records templates and pools of which the corresponding lateral is a part of
            We use -1 for padding
        boundary_laterals_indices: Array of shape (n_boundary_laterals,)
            Indices of the laterals connecting to variables outside the boundary
        boundary_laterals_sides_indices: Binary array of shape (n_boundary_laterals,)
            Specifies which side of the lateral connects to the variable outside the boundary
        boundary_conditions: Array of shape (n_boundary_laterals,)
            Specifies the boundary conditions for the different laterals connecting to variables
            outside the boundary
        n_laterals : Total number of laterals. Used to construct messages_l2h

    Returns:
        messsages_l2h: Array of shape (n_laterals, 2)
            For a particulalr lateral i corresponding to the edge (j, k),
            messages_l2h[i, 0] contains messages from lateral i to the HOF
            at feature j, and messages_l2h[i, 1] contains messages from
            lateral i to the HOF at feature k.
    """
    n_laterals = features_pools_indices.shape[0]
    is_padding = jnp.all(features_pools_indices == -1, axis=-1)
    messages_l2h = jnp.max(
        jnp.where(
            jnp.where(
                is_padding,
                -1,
                messages_h2pool.special_indices[
                    features_pools_indices[..., 0],
                    features_pools_indices[..., 1],
                    features_pools_indices[..., 2],
                ],
            )
            == jnp.arange(n_laterals)[:, None, None],
            jnp.where(
                is_padding,
                -INF_REPLACEMENT,
                messages_h2pool.special[
                    features_pools_indices[..., 0],
                    features_pools_indices[..., 1],
                    features_pools_indices[..., 2],
                ],
            ),
            jnp.where(
                is_padding,
                -INF_REPLACEMENT,
                messages_h2pool.regular[
                    features_pools_indices[..., 0],
                    features_pools_indices[..., 1],
                    features_pools_indices[..., 2],
                ],
            ),
        ),
        axis=-1,
    )[:, jnp.array([1, 0])]
    messages_l2h = (
        jnp.concatenate([messages_l2h, jnp.zeros((1, 2))], axis=0)
        .at[
            boundary_laterals_indices,
            1 - boundary_laterals_sides_indices,
        ]
        .set(boundary_conditions)[:-1]
    )
    return messages_l2h


class BinaryLateralInternalMessages(NamedTuple):
    l2h: Union[np.ndarray, jnp.ndarray]
    h2pool: MessagesH2Pool


class BinaryLateralMessages(NamedTuple):
    input: float
    internal: BinaryLateralInternalMessages


class BinaryLateralWiring(NamedTuple):
    """BinaryLateralWiring.

    Args:
        nodes_frcs: Array of shape (n_feature_locs, 3)
        edges_frcs: Array of shape (n_laterals, 2, 3)
        laterals_indices : Array of shape (n_feature_locs, n_templates, n_pools, n_laterals_per_pool)
            Contains the indices of the relevant laterals for the different pools in the different templates.
            -1 is used for padding. For the HOF at a particular feature location, the allowed configurations
            are all off and having exactly one lateral in each pool to be on in each template.
        laterals_sides_indices : Binary array of shape (n_feature_locs, n_templates, n_pools, n_laterals_per_pool)
            0/ means the feature location is on the 0/1 side of the corresponding lateral.
            -1 is used for padding
        features_pools_indices: Array of shape (n_laterals, 2, n_template_pools, 3)
            The 3 elements in the last dimension are feature_loc_idx, template_idx and pool_idx
            The array records templates and pools of which the corresponding lateral is a part of
            We use -1 for padding
        boundary_laterals_indices: Array of shape (n_boundary_laterals,)
            Indices of the laterals connecting to variables outside the boundary
        boundary_laterals_sides_indices: Binary array of shape (n_boundary_laterals,)
            Specifies which side of the lateral connects to the variable outside the boundary
    """

    nodes_frcs: Union[np.ndarray, jnp.ndarray]
    edges_frcs: Union[np.ndarray, jnp.ndarray]
    laterals_indices: Union[np.ndarray, jnp.ndarray]
    laterals_sides_indices: Union[np.ndarray, jnp.ndarray]
    features_pools_indices: Union[np.ndarray, jnp.ndarray]
    boundary_laterals_indices: Union[np.ndarray, jnp.ndarray]
    boundary_laterals_sides_indices: Union[np.ndarray, jnp.ndarray]


def get_wiring_from_interaction_graph(
    interaction_graph: nx.Graph,
) -> BinaryLateralWiring:
    """get_wiring_from_interaction_graphs.

    Args:
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

    Returns:
        wiring: BinaryLateralWiring
    """
    nodes_frcs = []
    boundary_laterals_indices_sides_indices = set()
    for node in interaction_graph.nodes():
        if interaction_graph.nodes[node].get('is_within_boundary', False):
            nodes_frcs.append(node)
        else:
            boundary_laterals_indices_sides_indices.update(
                [
                    (
                        interaction_graph.edges[node, connected_node]['idx'],
                        interaction_graph.edges[node, connected_node]['sides'][node],
                    )
                    for connected_node in interaction_graph.neighbors(node)
                ]
            )

    boundary_laterals_indices_sides_indices = np.array(
        list(boundary_laterals_indices_sides_indices)
    )
    if len(boundary_laterals_indices_sides_indices) > 0:
        boundary_laterals_indices = boundary_laterals_indices_sides_indices[:, 0]
        boundary_laterals_sides_indices = boundary_laterals_indices_sides_indices[:, 1]
    else:
        boundary_laterals_indices = np.array([], dtype=int)
        boundary_laterals_sides_indices = np.array([], dtype=int)

    laterals_indices_list = []
    laterals_sides_indices_list = []
    for node in nodes_frcs:
        laterals_indices_list.append(
            jax.tree_util.tree_map(
                lambda connected_node: interaction_graph.edges[
                    node, tuple(connected_node)
                ]['idx'],
                interaction_graph.nodes[node]['templates'],
            )
        )
        laterals_sides_indices_list.append(
            jax.tree_util.tree_map(
                lambda connected_node: interaction_graph.edges[
                    node, tuple(connected_node)
                ]['sides'][node],
                interaction_graph.nodes[node]['templates'],
            )
        )

    laterals_indices = pad(laterals_indices_list, -1)
    laterals_sides_indices = pad(laterals_sides_indices_list, -1)
    edges_frcs = np.zeros((len(interaction_graph.edges()), 2, 3), dtype=int)
    for edge in interaction_graph.edges():
        edge_dict = interaction_graph.edges[edge]
        edges_frcs[edge_dict['idx'], edge_dict['sides'][edge[0]]] = np.array(edge[0])
        edges_frcs[edge_dict['idx'], edge_dict['sides'][edge[1]]] = np.array(edge[1])

    nodes_indices_dict = {
        tuple(node): node_idx for node_idx, node in enumerate(nodes_frcs)
    }
    features_pools_indices = [
        [
            np.full((2, 0, 3), dtype=int, fill_value=-1)
            for _ in range(edges_frcs.shape[1])
        ]
        for _ in range(edges_frcs.shape[0])
    ]
    for lateral_idx in range(edges_frcs.shape[0]):
        for side_idx in range(2):
            node = tuple(edges_frcs[lateral_idx, side_idx])
            if node not in nodes_indices_dict:
                continue

            feature_loc_idx = nodes_indices_dict[node]
            template_pools_indices = np.argwhere(
                laterals_indices[feature_loc_idx] == lateral_idx
            )[:, :-1]
            features_pools_indices[lateral_idx][side_idx] = np.concatenate(
                [
                    np.full(
                        (template_pools_indices.shape[0], 1),
                        dtype=int,
                        fill_value=feature_loc_idx,
                    ),
                    template_pools_indices,
                ],
                axis=1,
            )

    features_pools_indices = pad(features_pools_indices, -1)

    wiring = BinaryLateralWiring(
        nodes_frcs=np.array(nodes_frcs),
        edges_frcs=edges_frcs,
        laterals_indices=laterals_indices,
        laterals_sides_indices=laterals_sides_indices,
        features_pools_indices=features_pools_indices,
        boundary_laterals_indices=boundary_laterals_indices,
        boundary_laterals_sides_indices=boundary_laterals_sides_indices,
    )
    return wiring


update_messages_h2v = jax.vmap(
    update_messages_h2v_for_feature_loc,
    in_axes=(0, None, 0),
    out_axes=0,
)


def update_messages(
    messages: BinaryLateralMessages,
    wiring: BinaryLateralWiring,
    logw: jnp.ndarray,
    boundary_conditions: float,
    damping: float,
) -> BinaryLateralMessages:
    messages_pool2h = get_messages_pool2h(
        messages.internal.l2h,
        wiring.laterals_indices,
        wiring.laterals_sides_indices,
    )
    messages_f2h = messages.input
    messages_h2pool, messages_h2f = update_messages_h2v(
        messages_pool2h, messages_f2h, logw
    )
    messages_l2h = update_messages_l2h(
        messages_h2pool=messages_h2pool,
        features_pools_indices=wiring.features_pools_indices,
        boundary_laterals_indices=wiring.boundary_laterals_indices,
        boundary_laterals_sides_indices=wiring.boundary_laterals_sides_indices,
        boundary_conditions=boundary_conditions,
    )
    # Apply damping
    messages = messages._replace(
        internal=BinaryLateralInternalMessages(
            h2pool=MessagesH2Pool(
                regular=jnp.where(
                    messages_h2pool.special_indices
                    == messages.internal.h2pool.special_indices,
                    damping * messages.internal.h2pool.regular
                    + (1.0 - damping) * messages_h2pool.regular,
                    messages_h2pool.regular,
                ),
                special=jnp.where(
                    messages_h2pool.special_indices
                    == messages.internal.h2pool.special_indices,
                    damping * messages.internal.h2pool.special
                    + (1.0 - damping) * messages_h2pool.special,
                    messages_h2pool.special,
                ),
                special_indices=messages_h2pool.special_indices,
            ),
            l2h=damping * messages.internal.l2h + (1.0 - damping) * messages_l2h,
        ),
    )
    return messages


def infer(
    messages: BinaryLateralMessages,
    wiring: BinaryLateralWiring,
    logw: jnp.ndarray,
    n_bp_iter: int,
    boundary_conditions: float,
    damping: float,
) -> BinaryLateralMessages:
    def bp_update(messages, x):
        return (
            update_messages(messages, wiring, logw, boundary_conditions, damping),
            None,
        )

    messages, _ = jax.lax.scan(
        bp_update,
        messages,
        None,
        n_bp_iter,
    )
    return messages


def initialize_messages(
    input: float,
    boundary_conditions,
    wiring: BinaryLateralWiring,
):
    messages_l2h = 0.01 * np.random.logistic(size=(wiring.edges_frcs.shape[0] + 1, 2))
    messages_l2h[
        wiring.boundary_laterals_indices, wiring.boundary_laterals_sides_indices
    ] = boundary_conditions
    messages_l2h = messages_l2h[:-1]
    messages_h2pool = MessagesH2Pool(
        regular=np.zeros(wiring.laterals_indices.shape[:3]),
        special=np.zeros(wiring.laterals_indices.shape[:3]),
        special_indices=wiring.laterals_indices[..., 0],
    )
    messages = BinaryLateralMessages(
        input=input,
        internal=BinaryLateralInternalMessages(
            l2h=messages_l2h, h2pool=messages_h2pool
        ),
    )
    return messages


def get_laterals_connectivity(messages, wiring, feature_activations):
    laterals_beliefs = np.sum(messages.internal.l2h, axis=1)
    edges_frcs = wiring.edges_frcs[laterals_beliefs >= 0]
    G = nx.Graph()
    G.add_nodes_from([tuple(node) for node in feature_activations])
    G.add_nodes_from([tuple(node) for node in edges_frcs.reshape((-1, 3))])
    G.add_edges_from([(tuple(edge[0]), tuple(edge[1])) for edge in edges_frcs])
    return G
