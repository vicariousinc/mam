import itertools
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from mam.utils import INF_REPLACEMENT, MAX_MSG, normalize_and_clip, pad
from tqdm import tqdm


class ORLayerWiring(NamedTuple):
    """ORLayerWiring.

    Args:
        features_description : np.array
            Array of shape (n_features, n_children, 4)
            The 4 elements are flattened_channel_idx, dr, dc, parent_idx
        pools_to_children : np.array
            Array of shape (n_features, n_pools + 1, n_children_per_pool), dtype int
            Mapping from pool indices to children indices
            The size of the 2nd dimension is set to n_pools + 1 for padding purposes
        children_to_pools : np.array
            Array of shape (n_features, n_children + 1), dtype int
            Mapping from children indices to the indices of the pool that contain the children
            Pools should be non-overlapping, so a lateral belongs to a single pool
            The size of the 2nd dimension is set to n_children + 1 for padding purposes
        parents_description : np.array
            Array of shape (n_flattened_channels, n_parents, 4)
            The 4 elements are feature_idx, dr, dc, child_idx
    """

    features_description: Union[np.ndarray, jnp.ndarray]
    pools_to_children: Union[np.ndarray, jnp.ndarray]
    children_to_pools: Union[np.ndarray, jnp.ndarray]
    parents_description: Union[np.ndarray, jnp.ndarray]


def get_or_layer_wiring_with_pooling(features_array):
    """get_or_layer_wiring_with_pooling

    Parameters
    ----------
    features_array: Array of shape (n_features, n_channels, M_patch, N_patch, n_states - 1), dtype int
        An integer array giving a dense and intuitive description of the features
        1st dimension corresponds to different features (i.e. different parent/children relationships)
        2nd dimension corresponds to different channels in the children variables
        M_patch, N_patch specifies the size of the receptive field
        n_states specifies the number of states in the children variables. This allows us to support
        categorical children variables
        features_array takes integer values. 0 means the corresponding children variables are not used.
        Nonzero entries correspond to children variables that are used.
        Different nonzero integer entries correspond to different pools in the feature definition.

    Returns
    -------
    wiring : ORLayerWiring
    """
    n_features, n_channels, _, _, n_states_minus_1 = features_array.shape
    n_flattened_channels = n_channels * n_states_minus_1
    n_pools = np.max(features_array)
    features_description = [[] for _ in range(n_features)]
    pools_to_children_list = [[[] for _ in range(n_pools)] for _ in range(n_features)]
    children_to_pools_list = [[] for _ in range(n_features)]
    child_indices = np.zeros((n_features,), dtype=np.int32)
    parents_description = [[] for _ in range(n_flattened_channels)]
    parent_indices = np.zeros((n_flattened_channels,), dtype=np.int32)
    for feature_idx, channel_idx, dr, dc, state_idx_minus_1 in np.array(
        np.nonzero(features_array)
    ).T:
        flattened_channel_idx = channel_idx * n_states_minus_1 + state_idx_minus_1
        features_description[feature_idx].append(
            [flattened_channel_idx, dr, dc, parent_indices[flattened_channel_idx]]
        )
        pool_idx = (
            features_array[feature_idx, channel_idx, dr, dc, state_idx_minus_1] - 1
        )
        pools_to_children_list[feature_idx][pool_idx].append(child_indices[feature_idx])
        children_to_pools_list[feature_idx].append(pool_idx)
        parents_description[flattened_channel_idx].append(
            [feature_idx, -dr, -dc, child_indices[feature_idx]]
        )
        parent_indices[flattened_channel_idx] += 1
        child_indices[feature_idx] += 1

    # Construct features description
    n_children = np.max(list(map(len, features_description)))
    features_description = np.array(
        [
            np.concatenate(
                [
                    np.array(features_description[feature_idx]),
                    np.array([-1, 0, 0, -1])
                    * np.ones((n_children - len(features_description[feature_idx]), 4)),
                ],
                axis=0,
            )
            for feature_idx in range(n_features)
        ],
        dtype=np.int32,
    )
    # Construct pools_to_children
    n_children_per_pool = np.max(
        [
            list(map(len, pools_to_children_for_feature))
            for pools_to_children_for_feature in pools_to_children_list
        ]
    )
    pools_to_children = -np.ones(
        (n_features, n_pools + 1, n_children_per_pool), dtype=np.int32
    )
    for feature_idx in range(n_features):
        pools_to_children[feature_idx, :-1] = np.array(
            [
                np.concatenate(
                    [
                        np.array(pools_to_children_list[feature_idx][pool_idx]),
                        -np.ones(
                            (
                                n_children_per_pool
                                - len(pools_to_children_list[feature_idx][pool_idx])
                            ),
                            dtype=np.int32,
                        ),
                    ]
                )
                for pool_idx in range(n_pools)
            ]
        )
    # Construct children_to_pools
    children_to_pools = np.array(
        [
            np.concatenate(
                [
                    children_to_pools_list[feature_idx],
                    -np.ones(
                        n_children + 1 - len(children_to_pools_list[feature_idx]),
                        dtype=np.int32,
                    ),
                ]
            )
            for feature_idx in range(n_features)
        ]
    )
    # Construct parents_description
    n_parents = np.max(list(map(len, parents_description)))
    parents_description = np.array(
        [
            np.concatenate(
                [
                    np.array(parents_description[channel_idx]),
                    np.array([-1, 0, 0, -1])
                    * np.ones((n_parents - len(parents_description[channel_idx]), 4)),
                ],
                axis=0,
            )
            if len(parents_description[channel_idx]) > 0
            else np.array([-1, 0, 0, -1])
            * np.ones((n_parents - len(parents_description[channel_idx]), 4))
            for channel_idx in range(n_flattened_channels)
        ],
        dtype=np.int32,
    )
    wiring = ORLayerWiring(
        features_description=features_description,
        pools_to_children=pools_to_children,
        children_to_pools=children_to_pools,
        parents_description=parents_description,
    )
    return wiring


def get_features_array_with_pooling_from_binary_features_array(
    features_array: np.ndarray,
):
    features_array_with_pooling = features_array.copy()
    for idx in range(features_array.shape[0]):
        indices = np.nonzero(features_array[idx])
        features_array_with_pooling[idx][indices] = np.arange(1, len(indices[0]) + 1)

    return features_array_with_pooling


def get_wiring_hof_with_pooling(features_description, pools_to_laterals):
    """get_wiring_hof_with_pooling.

    Args:
        features_description (np.array): Array of shape (n_features, n_laterals, 4)
            The 4 elements are flattened_channel_idx, dr, dc, parent_idx
        pools_to_laterals (np.array): Array of shape (n_features, n_pools + 1, n_laterals_per_pool), dtype int32
            Mapping from pool indices to laterals indices
            The size of the 2nd dimension is set to n_pools + 1 for padding purposes

    Returns:
        wiring_hof_with_pooling ({str : np.array}): Wiring in HOF format with pooling
    """
    n_features = features_description.shape[0]
    n_pools = pools_to_laterals.shape[1] - 1
    pools_indices = (
        jnp.tile(jnp.arange(n_pools)[None, None], (n_features, 2, 1)).at[:, 0].set(-1)
    )
    laterals_states = jnp.zeros(pools_indices.shape, dtype=jnp.int32).at[:, 1].set(1)
    features_states = jnp.zeros((n_features, 2), dtype=jnp.int32).at[:, 1].set(1)
    laterals_description = jnp.concatenate(
        [
            features_description,
            jnp.tile(
                jnp.array([-1, 0, 0, -1], dtype=jnp.int32)[None, None],
                (n_features, 1, 1),
            ),
        ],
        axis=1,
    )
    wiring_hof_with_pooling = {
        'pools_indices': pools_indices,
        'laterals_states': laterals_states,
        'features_states': features_states,
        'laterals_description': laterals_description,
    }
    return wiring_hof_with_pooling


def reroute_messages_with_groups_from_dense_messages(
    connections_description, groups_to_connections, source_messages
):
    """reroute_messages

    Parameters
    ----------
    connections_description : np.array
        Array of shape (n_channels, n_connections, 4)
    groups_to_connections : np.array
        Array of shape (n_channels, n_groups, n_connections_per_group)
    source_messages : np.array
        Array of shape (n_channels, M, N, n_parents, n_states)

    Returns
    -------
    rerouted_messages : np.array
        largest : np.array
            Array of shape (n_channels, M, N, n_groups, n_states)
            Largest incoming message per group
        2nd_largest : np.array
            Array of shape (n_channels, M, N, n_groups, n_states)
            2nd largest incoming message per group
        max_indices : np.array
            Array of shape (n_channels, M, N, n_groups, n_states)
            Indices for which connection the largest message comes from for each group and each state
    """
    reroute_messages_for_channel = jax.vmap(
        jax.partial(
            reroute_messages_for_channel_for_group, source_messages=source_messages
        ),
        in_axes=(0, None),
        out_axes=2,
    )
    rerouted_messages = jax.vmap(reroute_messages_for_channel, in_axes=0, out_axes=0)(
        groups_to_connections, connections_description
    )
    return rerouted_messages


def reroute_messages_for_channel_for_group(
    group_description, connections_description, source_messages
):
    """reroute_messages_for_channel_for_group

    Parameters
    ----------
    group_description : np.array
        Array of shape (n_connections_per_group,)
        List of connection indices in the group
    connections_description : np.array
        Array of shape (n_connections, 4)
    source_messages : np.array
        Array of shape (n_channels, M, N, n_parents, n_states)

    Returns
    -------
    rerouted_messages_for_channel_for_group : np.array
        largest : np.array
            Array of shape (M, N, n_states)
            Largest incoming message per group
        2nd_largest : np.array
            Array of shape (M, N, n_states)
            2nd largest incoming message per group
        max_indices : np.array
            Array of shape (M, N, n_states)
            Indices for which connection the largest message comes from for each group and each state
    """
    _, M, N, _, n_states = source_messages.shape
    rerouted_messages_for_channel_for_group = {
        'largest': reroute_messages_for_channel_for_connection_with_groups(
            connections_description[group_description[0]], source_messages
        ),
        '2nd_largest': jnp.full((M, N, n_states), fill_value=-MAX_MSG),
        'max_indices': jnp.full(
            (M, N, n_states), fill_value=group_description[0], dtype=jnp.int32
        ),
    }

    def process_connection(rerouted_messages_for_channel_for_group, connection_idx):
        messages = reroute_messages_for_channel_for_connection_with_groups(
            connections_description[connection_idx], source_messages
        )
        rerouted_messages_for_channel_for_group['max_indices'] = jnp.where(
            messages > rerouted_messages_for_channel_for_group['largest'],
            connection_idx,
            rerouted_messages_for_channel_for_group['max_indices'],
        )
        messages, _ = jax.lax.top_k(
            jnp.stack(
                [
                    messages,
                    rerouted_messages_for_channel_for_group['largest'],
                    rerouted_messages_for_channel_for_group['2nd_largest'],
                ],
                axis=-1,
            ),
            2,
        )
        rerouted_messages_for_channel_for_group['largest'] = messages[..., 0]
        rerouted_messages_for_channel_for_group['2nd_largest'] = messages[..., 1]
        return (rerouted_messages_for_channel_for_group, None)

    rerouted_messages_for_channel_for_group, _ = jax.lax.scan(
        process_connection,
        rerouted_messages_for_channel_for_group,
        group_description[1:],
    )
    return rerouted_messages_for_channel_for_group


def reroute_messages_for_channel_for_connection_with_groups(
    connection_description, source_messages
):
    """reroute_messages_for_channel_for_connection_with_groups
    Parameters
    ----------
    connection_description : np.array
        Array of shape (4,)
        The 4 elements are channel_idx, dr, dc, source_connection_idx
    source_messages : np.array
        Array of shape (n_channels, M, N, n_parents, n_states)

    Returns
    -------
    rerouted_messages_for_channel_for_connection_with_groups : np.array
        Array of shape (M, N, n_states)
    """
    _, M, N, _, n_states = source_messages.shape
    source_channel_idx, dr, dc, source_connection_idx = connection_description
    messages_to_be_rerouted = source_messages[
        source_channel_idx, :, :, source_connection_idx
    ]
    rerouted_messages_for_channel_for_connection_with_groups = jnp.where(
        jnp.logical_and(
            jnp.logical_and(jnp.arange(M) + dr >= 0, jnp.arange(M) + dr < M).reshape(
                (-1, 1, 1)
            ),
            jnp.logical_and(jnp.arange(N) + dc >= 0, jnp.arange(N) + dc < N).reshape(
                (1, -1, 1)
            ),
        ),
        jnp.roll(messages_to_be_rerouted, shift=(-dr, -dc), axis=(0, 1)),
        jnp.full((1, 1, n_states), fill_value=-MAX_MSG).at[:, :, 0].set(0),
    )
    return rerouted_messages_for_channel_for_connection_with_groups


def update_messages_c2o(messages_o2c, messages_unary):
    """update_messages_c2o

    Parameters
    ----------
    messages_o2c : np.array
        Array of shape (n_channels * (n_states - 1), M_lower, N_lower, 2)
        Messages from OR factors to children
    messages_unary : np.array
        Array of shape (n_channels, M_lower, N_lower, n_states)
    mode : str
        'sum' or 'max'

    Returns
    -------
    messages_c2o : np.array
        Array of shape (n_channels * (n_states - 1), M_lower, N_lower, 2)
    """
    n_channels, M_lower, N_lower, n_states = messages_unary.shape
    messages_c2o = jax.vmap(
        jax.partial(update_messages_c2o_for_channel), in_axes=0, out_axes=0
    )(
        messages_o2c.reshape((n_channels, n_states - 1, M_lower, N_lower, 2)),
        messages_unary,
    )
    return messages_c2o.reshape((n_channels * (n_states - 1)), M_lower, N_lower, 2)


def update_messages_c2o_for_channel(messages_o2c, messages_unary):
    """update_messages_c2o_for_channel
    Expand categorical variables into a set of binary variables by having
    an additional factor per location enforcing that at most one binary
    variable can be on at each location

    Parameters
    ----------
    messages_o2c : np.array
        Array of shape (n_states - 1, M, N, 2)
    messages_unary : np.array
        Array of shape (M, N, n_states)

    Returns
    -------
    messages_c2o : np.array
        Array of shape (n_states - 1, M, N, 2)
    """

    @jax.partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def max_over_all_but_one(messages, idx):
        return jnp.max(messages.at[idx].set(-INF_REPLACEMENT), axis=0)

    M, N, n_states = messages_unary.shape
    reshaped_messages_unary = (
        jnp.zeros((n_states - 1, M, N, 2))
        .at[..., 1]
        .set(jnp.moveaxis(messages_unary[..., 1:], -1, 0))
    )
    messages_v2f = messages_o2c + reshaped_messages_unary
    messages_f2v = (
        jnp.zeros_like(messages_o2c)
        .at[..., 0]
        .max(max_over_all_but_one(messages_v2f[..., 1], jnp.arange(n_states - 1)))
    )
    messages_c2o = messages_f2v + reshaped_messages_unary
    messages_c2o = normalize_and_clip(messages_c2o)
    return messages_c2o


def make_get_preproc_func(or_layer_wiring, upper_shape):
    expanded_wiring = get_wiring_hof_with_pooling(
        or_layer_wiring.features_description, or_layer_wiring.pools_to_children
    )
    n_parents = or_layer_wiring.parents_description.shape[1]

    def get_preproc(from_bottom):
        n_channels, M_lower, N_lower, n_states = from_bottom.shape
        n_flattened_channels = n_channels * (n_states - 1)
        messages_c2o = update_messages_c2o(
            jnp.zeros((n_flattened_channels, M_lower, N_lower, 2)),
            from_bottom,
        )
        messages_o2p = jnp.tile(messages_c2o[:, :, :, None], (1, 1, 1, n_parents, 1))
        messages_pool2h = reroute_messages_with_groups_from_dense_messages(
            expanded_wiring['laterals_description'],
            or_layer_wiring.pools_to_children,
            jnp.concatenate(
                [messages_o2p, jnp.zeros(messages_o2p.shape[:3] + (1, 2))],
                axis=-2,
            ),
        )
        to_top = jnp.sum(
            messages_pool2h['largest'][:, : upper_shape[0], : upper_shape[1], :-1],
            axis=-2,
        )
        return to_top

    return get_preproc


class ForwardPassWiring(NamedTuple):
    """ForwardPassWiring.
    Args:
        locs: Array of shape (n_locs, 3)
            locs[:, 0] are all 0
            locs[:, 1:] records the translation we are going to apply to each elastic_graph
        elastic_graph_pools_indices: Array of shape (n_elastic_graphs, n_pools_per_elastic_graph)
            Indices of pools, ranging from 0 to n_total_feature_locs - 1, that correspond
            to a particular elastic_graph
        elastic_graph_frcs_flat: Array of shape (n_total_frcs, n_locs, 3)
            Concatenated frcs of all translated elastic_graphs
            elastic_graph_frcs_flat[elastic_graph_ranges[idx, 0]:elastic_graph_ranges[idx, 1]] gives the
            frcs of the elastic_graph idx
    """

    locs: Union[np.ndarray, jnp.ndarray]
    elastic_graph_pools_indices: Union[np.ndarray, jnp.ndarray]
    elastic_graph_frcs_flat: Union[np.ndarray, jnp.ndarray]


def get_forward_pass_wiring(
    step_size: int,
    frcs_list: List[np.ndarray],
    preproc_shape: Tuple[int, int, int],
):
    locs = np.array(
        list(
            itertools.product(
                np.arange(0, preproc_shape[1], step_size),
                np.arange(0, preproc_shape[2], step_size),
            )
        )
    )
    locs = np.concatenate([np.zeros((locs.shape[0], 1), dtype=int), locs], axis=1)
    n_feature_locs_cumsum = np.cumsum([0] + [frcs.shape[0] for frcs in frcs_list])
    elastic_graph_pools_indices = pad(
        [
            np.arange(n_feature_locs_cumsum[idx], n_feature_locs_cumsum[idx + 1])
            for idx in range(len(frcs_list))
        ],
        -1,
    )
    for frcs in tqdm(frcs_list):
        frcs[:, 1:] -= np.min(frcs[:, 1:], axis=0, keepdims=True)

    elastic_graph_frcs_flat = np.concatenate(frcs_list, axis=0)[:, None] + locs[None]
    return ForwardPassWiring(
        locs=locs,
        elastic_graph_pools_indices=elastic_graph_pools_indices,
        elastic_graph_frcs_flat=elastic_graph_frcs_flat.astype(np.int16, copy=False),
    )


@dataclass
class ForwardPass:
    """ForwardPass.
    Args:
        pool_size: Size of the used pools. Note that pool_size = 2 * pool_radius + 1
        frcs_list: A list of np arrays. Each array is of shape (n_feature_activations, 3),
            and represents the sparsification of a particular elastic graph
    """

    pool_size: int
    frcs_list: List[np.ndarray]
    preproc_shape: Optional[Tuple[int, int, int]] = None
    fp_norm_exponent: float = 0.8
    step_size: Optional[int] = None

    def __post_init__(self):
        if self.step_size is None:
            self.step_size = self.pool_size

        if self.preproc_shape is not None:
            self.wiring = get_forward_pass_wiring(
                self.step_size, self.frcs_list, self.preproc_shape
            )

        all_pool_locs = np.array(
            list(
                itertools.product(
                    np.arange(self.preproc_shape[0]),
                    np.arange(self.preproc_shape[1]),
                    np.arange(self.preproc_shape[2]),
                )
            ),
            dtype=np.int16,
        )
        self.all_pool_locs = jax.device_put(all_pool_locs)
        pooling_shifts = np.array(
            list(itertools.product(np.arange(self.pool_size), repeat=2))
        )
        # Array of shape (pool_size**2, 3)
        self.pooling_shifts = jax.device_put(
            np.concatenate(
                [np.zeros((pooling_shifts.shape[0], 1), dtype=int), pooling_shifts],
                axis=1,
            ).astype(np.int16, copy=False)
        )

    def make_forward_pass_func(self, preproc_shape: Tuple[int, int, int] = None):
        if preproc_shape is not None and preproc_shape != self.preproc_shape:
            wiring = jax.device_put(
                get_forward_pass_wiring(self.step_size, self.frcs_list, preproc_shape)
            )
        else:
            wiring = self.wiring

        def forward_pass(
            preproc: jnp.ndarray, wiring: ForwardPassWiring = wiring
        ) -> jnp.ndarray:
            """forward_pass.

            Args:
                preproc: Array of shape (n_channels, M, N)

            Returns:
                fp_scores: Array of shape (n_elastic_graphs, n_locs)
                    Scores of elastic_graphs at different locations
            """
            all_pool_scores = jnp.pad(
                jnp.max(
                    preproc[
                        self.all_pool_locs[..., None, 0] + self.pooling_shifts[:, 0],
                        self.all_pool_locs[..., None, 1] + self.pooling_shifts[:, 1],
                        self.all_pool_locs[..., None, 2] + self.pooling_shifts[:, 2],
                    ],
                    axis=-1,
                ).reshape(
                    (
                        self.preproc_shape[0],
                        self.preproc_shape[1],
                        self.preproc_shape[2],
                    )
                ),
                pad_width=((0, 0), (0, 1), (0, 1)),
                mode='constant',
                constant_values=-INF_REPLACEMENT,
            )
            pooling_scores = all_pool_scores[
                wiring.elastic_graph_frcs_flat[..., 0],
                wiring.elastic_graph_frcs_flat[..., 1],
                wiring.elastic_graph_frcs_flat[..., 2],
            ]
            fp_scores = jnp.sum(
                jnp.where(
                    wiring.elastic_graph_pools_indices[..., None] == -1,
                    0,
                    pooling_scores[wiring.elastic_graph_pools_indices],
                ),
                axis=1,
            ) / (
                jnp.sum(wiring.elastic_graph_pools_indices >= 0, axis=1)[:, None]
                ** self.fp_norm_exponent
            )
            return fp_scores

        return forward_pass
