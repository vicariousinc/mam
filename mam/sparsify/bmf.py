# Sparsification with the sparsifier using given object parts

import numba as nb
import numpy as np


def get_structure(bottom_up, W):
    assert len(bottom_up.shape) == 3 and len(W.shape) == 4
    assert bottom_up.shape[-1] == W.shape[-1]
    factors_bu, factors, factors_loc, parents_nz = weight_structure(bottom_up, W)
    assert (
        len(factors) > 0
    ), "The given features cannot generate any of the elements in the image"
    factors_bu, factors = np.hstack(factors_bu), np.hstack(factors)
    return factors_bu, factors, factors_loc, parents_nz


def init_messages(td, factors, factors_loc):
    # inhibition messages and beliefs
    n_messages = factors.size
    or_messages = 0.01 * np.random.randn(n_messages)
    beliefs = init_beliefs(td, or_messages, factors, factors_loc)
    return or_messages, beliefs


def update_messages(
    td,
    beliefs,
    or_messages,
    factors_bu,
    factors,
    factors_loc,
    damping=0.5,
    parallel_bp=True,
):
    if parallel_bp:
        return update_or_mp_par(
            td, beliefs, or_messages, factors_bu, factors, factors_loc, damping
        )
    else:
        return update_or_mp(
            td, beliefs, or_messages, factors_bu, factors, factors_loc, damping
        )


def densify(x, parents_nz):
    nz = (parents_nz > -1).nonzero()
    s = parents_nz - np.inf
    s[nz] = x
    return s


def infer(
    td,
    factors_bu,
    factors,
    factors_loc,
    max_iter,
    damping=0.5,
    parallel_bp=True,
    retries=1,
    threshold=1e-10,
):
    for retry in range(retries):
        converged = False
        or_messages, beliefs = init_messages(td, factors, factors_loc)
        for i in range(max_iter):
            delta_max = update_messages(
                td,
                beliefs,
                or_messages,
                factors_bu,
                factors,
                factors_loc,
                damping,
                parallel_bp,
            )
            if delta_max < threshold:
                converged = True
                break
        if converged:
            if retry > 1:
                print("Retried {} times".format(retry))
            break
    else:
        print(
            'Warning, inference reached the maximum number of iterations ({}), retried {} times.'.format(
                max_iter, retry + 1
            )
        )

    return beliefs


def sparsify(
    x,
    W,
    max_iter,
    min_score,
    score_scale=1.0,
    td=-1.0,
    damping=0.1,
    parallel_bp=True,
    retries=1,
    threshold=1e-10,
):
    assert len(x.shape) == 3
    h, w, n_ch = x.shape
    assert 0 <= x.min() and x.max() <= 1
    bottom_up = score_scale * x + min_score
    factors_bu, factors, factors_loc, parents_nz = get_structure(bottom_up, W)
    beliefs = infer(
        td,
        factors_bu,
        factors,
        factors_loc,
        max_iter,
        damping,
        parallel_bp,
        retries,
        threshold,
    )
    beliefs_dense = densify(beliefs, parents_nz)
    S = np.vstack(sharpen(beliefs_dense).nonzero()).T
    return S, beliefs_dense


@nb.njit(cache=True)
def or_fwd(in_mess, out_mess):
    """Message updates for logical OR factor.
    This function computes the outgoing messages at the input of an OR gate,
    given the incoming messages at such input, and the incoming message at the gate's output.

    Parameters
    ----------
    in_mess : np.array, in_mess.shape = n_parents,
        Incoming messages at the input of the OR gate
    out_mess : float
        Incoming message at the output of the OR gate
    Returns
    -------
    res_mess : np.array, res_mess.shape = n_parents,
        Outgoing messages at the input of the OR gate
    """
    assert len(in_mess) > 0, "logical gate with zero inputs"
    if len(in_mess) == 1:
        return np.array([out_mess])

    bu = 0.0
    maxval = -np.inf
    maxval2 = -np.inf
    arg_max_val = 0

    for i in range(len(in_mess)):
        if np.isfinite(in_mess[i]):
            bu += max(0.0, in_mess[i])
        if in_mess[i] > maxval:
            if maxval > maxval2:
                maxval2 = maxval
            maxval = in_mess[i]
            arg_max_val = i
        elif in_mess[i] > maxval2:
            maxval2 = in_mess[i]

    k = -min(0.0, maxval)
    k2 = -min(0.0, maxval2)
    res_mess = np.zeros(len(in_mess))
    if not np.isfinite(maxval) and maxval > 0:
        for i in range(len(in_mess)):
            if i != arg_max_val or np.isinf(maxval2):
                res_mess[i] = 0.0
            else:
                res_mess[i] = min(out_mess + bu, k2)
    else:
        for i in range(len(in_mess)):
            if i != arg_max_val:
                res_mess[i] = min(out_mess + bu - max(0.0, in_mess[i]), k)
            else:
                res_mess[i] = min(out_mess + bu - max(0.0, in_mess[i]), k2)
    return res_mess


@nb.njit(cache=True)
def weight_structure(bottom_up, W):
    height, width, n_ch = bottom_up.shape
    n_features, f_h, f_w, n_ch2 = W.shape
    assert n_ch == n_ch2
    assert f_h <= height and f_w <= width
    parents_nz = -1 + np.zeros(
        (height - f_h + 1, width - f_w + 1, n_features), np.int64
    )
    next_id = 0
    for r in range(parents_nz.shape[0]):
        for c in range(parents_nz.shape[1]):
            for feat in range(parents_nz.shape[2]):
                if np.logical_and(
                    bottom_up[r : r + f_h, c : c + f_w] > 0, W[feat]
                ).any():  # iterate over sparsification elements
                    # if True:
                    # if that parent could have generated something in the image, include it
                    parents_nz[r, c, feat] = next_id
                    next_id += 1
    factors = []
    factors_loc_list = []
    factors_bu = []
    s_valid = np.empty(f_h * f_w * n_features, np.int64)
    for r in range(height):
        for c in range(width):
            for ch in range(n_ch):  # iterate over image elements
                patch = parents_nz[
                    max(0, r - f_h + 1) : min(height - f_h, r) + 1,
                    max(0, c - f_w + 1) : min(width - f_w, c) + 1,
                ]
                Wpatch = W[
                    :,
                    max(0, r - height + f_h) : min(r + 1, f_h),
                    max(0, c - width + f_w) : min(c + 1, f_w),
                    ch,
                ].transpose(1, 2, 0)
                # s = patch[Wpatch[::-1, ::-1]]
                # block below is simply: s = patch[Wpatch[::-1, ::-1]]
                # ------
                els = np.vstack(Wpatch[::-1, ::-1].nonzero()).T
                s = np.zeros(len(els), np.int64)
                for i in range(len(els)):
                    u, v, w = els[i]
                    s[i] = patch[u, v, w]
                # s_valid = s[s >= 0]
                # block below is simply: s_valid = s[s>=0]
                # ------
                s_valid_len = 0
                for el in s.flat:
                    if el >= 0:
                        s_valid[s_valid_len] = el
                        s_valid_len += 1
                # ------
                if s_valid_len > 0:
                    factors_bu.append(bottom_up[r, c, ch])
                    factors.append(s_valid[:s_valid_len].copy())
                    factors_loc_list.append(s_valid_len)
    factors_loc = np.hstack(
        (np.zeros(1, np.int64), np.cumsum(np.array(factors_loc_list)))
    )
    return factors_bu, factors, factors_loc, parents_nz


@nb.njit(cache=True)
def init_beliefs(td, or_messages, factors, factors_loc):
    n_messages = factors.size
    n_parents = factors.max() + 1
    assert (
        n_messages == factors_loc[-1]
    ), "Incompatible factors/factors_loc specification"
    beliefs = td + np.zeros(n_parents, dtype=or_messages.dtype)
    n_factors = factors_loc.size - 1
    for f in range(n_factors):
        # add beliefs corresponding to factor f
        idxs = factors[factors_loc[f] : factors_loc[f + 1]]
        out_messages = or_messages[factors_loc[f] : factors_loc[f + 1]]
        beliefs[idxs] += out_messages
    return beliefs


@nb.njit(cache=True)
def update_or_mp(
    td, beliefs, or_messages, factors_bu, factors, factors_loc, damping=1.0
):
    n_messages = factors.size
    n_factors = factors_loc.size - 1
    assert (
        n_messages == factors_loc[-1]
    ), "Incompatible factors/factors_loc specification"
    assert or_messages.size == n_messages
    delta_max = 0
    for f in np.random.permutation(n_factors):
        idxs = factors[factors_loc[f] : factors_loc[f + 1]]
        inc_messages = beliefs[idxs] - or_messages[factors_loc[f] : factors_loc[f + 1]]
        out_messages = or_fwd(inc_messages, factors_bu[f])
        delta = out_messages - or_messages[factors_loc[f] : factors_loc[f + 1]]
        delta_max = max(delta_max, np.abs(delta).max())
        or_messages[factors_loc[f] : factors_loc[f + 1]] += damping * delta
        beliefs[idxs] += damping * delta
    beliefs[:] = init_beliefs(
        td, or_messages, factors, factors_loc
    )  # refresh, might not be necessary
    return delta_max


@nb.njit(cache=True)
def update_or_mp_par(
    td, beliefs, or_messages, factors_bu, factors, factors_loc, damping=1.0
):
    n_messages = factors.size
    n_factors = factors_loc.size - 1
    assert (
        n_messages == factors_loc[-1]
    ), "Incompatible factors/factors_loc specification"
    assert or_messages.size == n_messages
    delta_max = 0
    new_or_messages = np.zeros_like(or_messages)
    for f in range(n_factors):
        idxs = factors[factors_loc[f] : factors_loc[f + 1]]
        inc_messages = beliefs[idxs] - or_messages[factors_loc[f] : factors_loc[f + 1]]
        out_messages = or_fwd(inc_messages, factors_bu[f])
        new_or_messages[factors_loc[f] : factors_loc[f + 1]] = out_messages
    delta = new_or_messages - or_messages
    delta_max = np.abs(delta).max()
    or_messages += damping * delta
    beliefs[:] = init_beliefs(td, or_messages, factors, factors_loc)  # refresh beliefs
    return delta_max


@nb.njit(cache=True)
def sharpen(a):
    """Converts a matrix in which the last dimension is a belief into a binary matrix of the same shape.
    For beliefs that are all negative, the belief becomes all zeros. Otherwise, it will have a 1 at the
    maximum along that dimension (stochastically if there are many identical maxima).

    Example:

    Input:
    [[-0.52 -0.33  1.54  1.54]
    [ 0.6  -1.03  1.19  0.33]
    [-1.34 -0.67 -2.25 -0.08]
    [-0.34  0.85  0.37  0.04]]

    Output:
    [[0. 0. 0. 1.]
     [0. 0. 1. 0.]
     [0. 0. 0. 0.]
     [0. 1. 0. 0.]]
    """
    assert len(a.shape) > 1
    a = a.copy()
    x = a.reshape(-1, a.shape[-1])
    for r in range(x.shape[0]):
        m = x[r].max()
        if m <= 0:
            x[r] = 0
        else:
            acc = 0.0
            for c in range(x.shape[1]):
                if x[r, c] == m:
                    x[r, c] = 1
                    acc += 1
                else:
                    x[r, c] = 0
            if acc > 1:
                x[r] = np.random.multinomial(1, x[r] / acc)
    return x.reshape(a.shape)
