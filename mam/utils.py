from collections import defaultdict
from typing import NamedTuple, Union

import jax.numpy as jnp
import numpy as np

MAX_MSG = 1000.0
INF_REPLACEMENT = 1e6


class InputMessages(NamedTuple):
    """InputMessages.

    Args:
        from_top: Array of shape (n_features, M_upper, N_upper, 2)
        from_bottom: Array of shape (n_channels, M_lower, N_lower, n_states)
    """

    from_top: Union[np.ndarray, jnp.ndarray]
    from_bottom: Union[np.ndarray, jnp.ndarray]


class OutputMessages(NamedTuple):
    """OutputMessages.

    Args:
        to_top: Array of shape (n_features, M_upper, N_upper, 2)
        to_bottom: Array of shape (n_channels, M_lower, N_lower, n_states)
    """

    to_top: Union[np.ndarray, jnp.ndarray]
    to_bottom: Union[np.ndarray, jnp.ndarray]


def normalize_and_clip(messages, max_msg=MAX_MSG):
    messages = messages - messages[..., [0]]
    messages = messages.clip(-max_msg, max_msg)
    return messages


# Generic padding function, taken from https://codereview.stackexchange.com/questions/222623/pad-a-ragged-multidimensional-array-to-rectangular-shape
def get_dimensions(array, level=0):
    yield level, len(array)
    try:
        for row in array:
            yield from get_dimensions(row, level + 1)
    except TypeError:  # not an iterable
        pass


def get_max_shape(array):
    dimensions = defaultdict(int)
    for level, length in get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


def iterate_nested_array(array, index=()):
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError:  # final level
        yield (*index, slice(len(array))), array


def pad(array, fill_value):
    dimensions = get_max_shape(array)
    result = np.full(dimensions, fill_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result
