# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""mHC -- manifold / hyper-connections, V4's replacement for the residual.

Reference: ``inference/model.py`` ``Block.hc_pre`` / ``hc_post`` / ``hc_head``
and ``inference/kernel.py::hc_split_sinkhorn_kernel``.

The residual stream is ``hc_mult`` parallel copies, ``[b, s, hc, d]``, produced
by broadcasting the embedding and collapsed again just before the LM head.
Every sublayer is wrapped by a pair:

* ``hc_pre`` contracts the ``hc`` copies to one ``[b, s, d]`` input, by a
  per-token weighted sum.
* ``hc_post`` puts the sublayer's output back, scaled per copy, on top of a
  *mixed* residual -- each copy out is a learned combination of all copies in,
  not its own lane.

All three weight sets (``pre``, ``post``, ``comb``) are produced per token from
the state itself: flatten the copies, RMS-scale, and project through ``hc_*_fn``
to ``mix_hc = (2 + hc) * hc`` numbers, then split. ``hc_*_scale`` is three
scalars, one per part, and ``hc_*_base`` a bias per output.

The combination block is the interesting one. It is softmaxed along rows, then
run through ``hc_sinkhorn_iters`` rounds of alternating row/column
normalization, which drives it toward doubly stochastic -- no copy can be
amplified or dropped across the round trip, which is what keeps the ``hc``
lanes from collapsing into each other over 43 layers.

Two places where this diverges from what the shapes suggest:

* The whole path runs in float32, on a bf16 model. The parameters are float32 in
  the checkpoint, the reference casts the state up before the mixer, and the
  Sinkhorn division chain is not safe in bf16.
* ``hc_head``, the final contraction, is *not* the same function: it emits only
  ``hc`` numbers, applies a plain sigmoid, and has no Sinkhorn step at all --
  there is nothing downstream to keep orthogonal.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import TensorValue, ops


def hc_mix_width(hc_mult: int) -> int:
    """``mix_hc`` in the reference: ``hc`` pre + ``hc`` post + ``hc * hc`` comb."""
    return (2 + hc_mult) * hc_mult


def _mixes(flat: TensorValue, fn: TensorValue, norm_eps: float) -> TensorValue:
    """``F.linear(x, fn) * rsqrt(mean(x^2) + eps)`` over the flattened copies.

    Note the RMS scale multiplies the *projection*, not the input. Algebraically
    the same thing for a linear map, but it is one scalar per token instead of
    ``hc * d`` values, and the reference writes it this way.
    """
    rsqrt = ops.rsqrt(ops.mean(flat * flat, axis=-1) + norm_eps)
    return ops.matmul(flat, ops.transpose(fn, 0, 1)) * rsqrt


def _flatten_copies(x: TensorValue, hc_mult: int) -> TensorValue:
    """``[b, s, hc, d]`` -> float32 ``[b, s, hc * d]``."""
    return ops.cast(
        ops.reshape(x, [x.shape[0], x.shape[1], hc_mult * int(x.shape[3])]),
        DType.float32,
    )


def hc_split_sinkhorn(
    mixes: TensorValue,
    scale: TensorValue,
    base: TensorValue,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Split ``[b, s, mix_hc]`` into pre, post and the Sinkhorn'd combination.

    ``inference/kernel.py::hc_split_sinkhorn_kernel``. The slicing is
    positional: ``[0:hc]`` is pre, ``[hc:2*hc]`` post, and the rest is the
    ``hc x hc`` combination in row-major order.

    Each part gets its own scalar from ``scale``, and ``post`` carries a factor
    of two the others do not -- a copy may be amplified up to 2x by the
    sublayer's contribution while ``pre`` stays inside ``(0, 1) + eps``.
    """
    hc = hc_mult
    pre = ops.sigmoid(mixes[..., :hc] * scale[0:1] + base[:hc]) + eps
    post = 2.0 * ops.sigmoid(
        mixes[..., hc : 2 * hc] * scale[1:2] + base[hc : 2 * hc]
    )

    comb = mixes[..., 2 * hc :] * scale[2:3] + base[2 * hc :]
    comb = ops.reshape(comb, [comb.shape[0], comb.shape[1], hc, hc])

    # Row softmax, then alternating column/row normalization. The first column
    # pass is outside the loop because the softmax already normalized the rows,
    # so the loop runs one fewer time than ``sinkhorn_iters`` suggests.
    comb = ops.softmax(comb, axis=-1) + eps
    comb = comb / (ops.sum(comb, axis=-2) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (ops.sum(comb, axis=-1) + eps)
        comb = comb / (ops.sum(comb, axis=-2) + eps)
    return pre, post, comb


def hc_pre(
    x: TensorValue,
    fn: TensorValue,
    scale: TensorValue,
    base: TensorValue,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Contract ``[b, s, hc, d]`` to ``[b, s, d]``, and hand back post and comb.

    ``post`` and ``comb`` are computed here and consumed by ``hc_post`` after
    the sublayer runs: they are read off the state *before* the sublayer, not
    after it.
    """
    flat = _flatten_copies(x, hc_mult)
    pre, post, comb = hc_split_sinkhorn(
        _mixes(flat, fn, norm_eps),
        scale,
        base,
        hc_mult,
        sinkhorn_iters,
        hc_eps,
    )
    copies = ops.reshape(flat, x.shape)
    y = ops.squeeze(ops.sum(ops.unsqueeze(pre, -1) * copies, axis=2), axis=2)
    return ops.cast(y, x.dtype), post, comb


def hc_post(
    x: TensorValue,
    residual: TensorValue,
    post: TensorValue,
    comb: TensorValue,
) -> TensorValue:
    """``[b, s, d]`` sublayer output back onto the ``[b, s, hc, d]`` stream.

    ``out[..., k, :] = post[..., k] * x + sum_j comb[..., j, k] * residual[..., j, :]``

    The mixing term is a rank-5 elementwise product reduced over ``j`` rather
    than the equivalent ``comb^T @ residual``. It is the reference's form, and
    on the accelerator it is also the accurate one: a batched matmul here would
    run the contraction in TF32 over the whole residual stream.
    """
    direct = ops.unsqueeze(post, -1) * ops.unsqueeze(x, -2)
    mixed = ops.squeeze(
        ops.sum(
            ops.unsqueeze(comb, -1)
            * ops.unsqueeze(ops.cast(residual, comb.dtype), -2),
            axis=2,
        ),
        axis=2,
    )
    return ops.cast(ops.cast(direct, comb.dtype) + mixed, x.dtype)


def hc_head(
    x: TensorValue,
    fn: TensorValue,
    scale: TensorValue,
    base: TensorValue,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
) -> TensorValue:
    """Final contraction to ``[b, s, d]`` before the norm and LM head.

    Same shape of computation as ``hc_pre`` but ``fn`` emits only ``hc``
    numbers, ``scale`` is a single scalar, and there is no Sinkhorn -- the
    output does not feed another copy-carrying layer, so nothing needs to stay
    doubly stochastic.
    """
    flat = _flatten_copies(x, hc_mult)
    pre = ops.sigmoid(_mixes(flat, fn, norm_eps) * scale + base) + hc_eps
    copies = ops.reshape(flat, x.shape)
    y = ops.squeeze(ops.sum(ops.unsqueeze(pre, -1) * copies, axis=2), axis=2)
    return ops.cast(y, x.dtype)


def expand_copies(x: TensorValue, hc_mult: int) -> TensorValue:
    """``[b, s, d]`` -> ``[b, s, hc, d]``, the stream the first block reads.

    ``h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`` in the reference. Every copy
    starts identical; they diverge through the per-copy ``post`` scaling.
    """
    return ops.broadcast_to(
        ops.unsqueeze(x, 2),
        [x.shape[0], x.shape[1], hc_mult, x.shape[2]],
    )
