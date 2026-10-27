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
"""Single-pass manifold-constrained Hyper-Connections (mHC) for V4.1.

V4.1's mHC runs ``hc_mult`` parallel residual streams through every decoder
sublayer. Unlike V4 there is no ``hc_head``: the sublayer input is collapsed
with the *previous* sublayer's ``pre`` (the "shifted post+pre" schedule --
each site's coefficients are computed from the stream *after* the previous
site's post/comb write-back), and the final trunk output is collapsed with
the *last FFN's* ``pre``.

These are pure functions: the layer holds the checkpoint parameters
(``hc_{attn,ffn}_{fn,base,scale}``) and passes them in, so the caller owns
every tensor. The three ``pre_mix`` states are distinct (see
:class:`_PreMixOmitted` and :func:`mhc_shifted_post_pre`).

The gate coefficients run through the graph ``hyper_connection_gates`` op
(``mo.hyper_connection.gates``); there is no fused mega/all-reduce step
here -- a tensor-parallel caller reduces its sublayer partials with the
ordinary all-reduce and then feeds the whole ``y`` to
:func:`mhc_shifted_post_pre`.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.graph import Dim
from max.nn.common_layers.functional_kernels import (
    hyper_connection_gates,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping


class _PreMixOmitted:
    """Sentinel distinguishing an omitted ``pre_mix`` from ``None``.

    ``pre_mix=None`` selects the V4.1 layer-0 identity collapse (copy stream
    0, unweighted); omitting the argument selects the classic behavior
    (collapse with this site's own freshly computed ``pre``). Conflating the
    two is a silent correctness bug: at the trunk's first sublayer every
    stream copy holds the same broadcast embedding, so a fresh-``pre``
    collapse there computes ``sum(pre) * stream[:, 0]`` and ``sum(pre)`` is
    generically not 1.
    """


_PRE_MIX_OMITTED = _PreMixOmitted()


def claim_replicated(t: Tensor, like: Tensor | None = None) -> Tensor:
    """Claims a rule-less op result on its mesh.

    Ops without a sharding rule (the mHC gate kernel, the Engram gather)
    return ``Unknown``-placed results under TP, which a rule-based ``F``
    op cannot consume. Every rank computes them from its whole,
    replicated operands, so the placement is genuinely ``Replicated``.
    Under data parallelism the operands are per-rank rows, so the claim
    follows a distributed ``like``'s mapping instead.

    Args:
        t: The rule-less op result, single-device or Unknown-distributed.
        like: The operand the result's rows follow; a distributed ``like``
            (data parallelism) puts the claim on its mapping.

    Returns:
        ``t``, claimed on its mesh (single-device tensors pass through
        unchanged).
    """
    if not t.is_distributed:
        return t
    if like is not None and like.is_distributed:
        return t.rebind_mapping(like.mapping)
    return t.rebind_mapping(DeviceMapping.replicated(t.mesh))


def _hc_gates(
    stream: Tensor,
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the ``pre`` / ``post`` / ``comb`` coefficients for one site.

    ``stream`` is ``[rows, hc_mult, hidden]``; the RMS statistic runs over the
    whole flattened ``hc_mult * hidden`` vector, one statistic per row,
    matching the reference's ``hc_mixes``. The gate math runs in
    float32 and the graph op also expects a float32 projection, so the
    stream is cast up front.
    """
    rows = stream.shape[0]
    flat = stream.cast(DType.float32).reshape([rows, hc_mult * stream.shape[2]])
    rms = F.rsqrt(F.mean(flat * flat, axis=-1) + norm_eps)
    pre, post, comb = hyper_connection_gates(
        (flat @ hc_fn.T) * rms,
        hc_base,
        hc_scale,
        hc_mult=hc_mult,
        hc_eps=hc_eps,
        hc_sinkhorn_iters=sinkhorn_iters,
    )
    # The gate op has no sharding rule, so its results are Unknown-placed
    # under a mesh and the reshape below would carry that Unknown into
    # placements the collapse's rule math cannot match. Every rank
    # computes them from its whole (replicated) operands -- or, under
    # data parallelism, from its own row shard, which matches the
    # stream's mapping -- so claim them there before anything else.
    pre = claim_replicated(pre, stream)
    post = claim_replicated(post, stream)
    comb = claim_replicated(comb, stream)
    return pre, post, comb.reshape([rows, hc_mult, hc_mult])


def _collapse(stream: Tensor, pre_mix: Tensor) -> Tensor:
    """Weighted-sums the streams down to one copy, accumulating float32.

    ``sum`` keeps the reduced axis at size 1 (max reduction semantics),
    so the ``[rows, 1, hidden]`` result is squeezed back to a 2-D copy.
    """
    return (
        (pre_mix.unsqueeze(-1) * stream.cast(DType.float32))
        .sum(axis=1)
        .squeeze(1)
    )


def _identity_pre_mix(rows: int | Dim, hc_mult: int, ref: Tensor) -> Tensor:
    """The one-hot mix: ``[rows, 1, 0, 0]``, float32 on ``ref``'s mapping.

    Built from the mapping, never ``ref.device``: a multi-device mesh has
    no single device, and the mix's rows must follow the stream's
    placement (per-rank rows under data parallelism).
    """
    return F.concat(
        [
            F.ones([rows, 1], dtype=DType.float32, device=ref.mapping),
            F.zeros(
                [rows, hc_mult - 1], dtype=DType.float32, device=ref.mapping
            ),
        ],
        axis=1,
    )


def _identity_collapse(stream: Tensor, hc_mult: int) -> Tensor:
    """Collapses by copying stream 0 unweighted (the one-hot ``pre_mix``)."""
    identity = _identity_pre_mix(stream.shape[0], hc_mult, stream)
    return _collapse(stream, identity).cast(stream.dtype)


def mhc_pre(
    stream: Tensor,
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    pre_mix: Tensor | _PreMixOmitted | None = _PRE_MIX_OMITTED,
    norm_eps: float = 1e-20,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Starts one site on an already post-updated stream: gates and collapse.

    This is the pre-only half of :func:`mhc_shifted_post_pre`. Use it when
    the caller must touch the stream between the previous site's write-back
    and this site's gates, which is the Engram case: the reference computes
    ``Engram(post(y, residual))`` and only then the gates and the carried
    collapse.
    Engram's per-stream gates do not commute with ``comb``, so the order is
    load-bearing::

        stream = mhc_post(y, stream, post, comb, hc_mult)
        stream = engram(stream)  # all hc_mult streams, per-stream gates
        post, comb, pre, collapsed = mhc_pre(
            stream, ..., hc_mult, pre_mix=carried_pre
        )

    Args:
        stream: The post-updated (and post-Engram, when applicable)
            streams, ``[rows, hc_mult, hidden]``.
        hc_fn: This site's projection, float32 ``[hc_mult * (hc_mult + 2),
            hc_mult * hidden]``.
        hc_scale: Per-band scale, float32 ``[3]``, ``pre``/``post``/``comb``.
        hc_base: Per-output bias, float32 ``[hc_mult * (hc_mult + 2)]``.
        hc_mult: The number of parallel streams.
        pre_mix: The collapse weights; three distinct states.

            - *omitted* (the default): this site's own freshly computed
              ``pre`` collapses the stream (the classic ModuleV3
              :class:`~max.nn.HyperConnection` reading).
            - ``None``: the identity carry. The trunk's first attention site
              copies stream 0 unweighted, because
              every stream holds the same broadcast embedding there.
            - a carried tensor: the previous site's ``pre``, float32
              ``[rows, hc_mult]``.
        norm_eps: The unweighted RMSNorm epsilon. V4.1 uses ``rms_norm_eps``
            (``1e-20``); the V3 module defaults to ``1e-6``, so pass it
            explicitly.
        hc_eps: The ``pre`` offset and Sinkhorn epsilon (``hc_eps``, ``1e-6``).
        sinkhorn_iters: Sinkhorn-Knott iterations (20).

    Returns:
        ``(post, comb, pre, collapsed)``: this site's write-back coefficients
        (for the next boundary), its collapse weights ``pre`` (the next
        site's ``pre_mix``; the final collapse's input at the last FFN), and
        the sublayer input ``collapsed`` ``[rows, hidden]`` in ``stream``'s
        dtype, for the caller to normalize.
    """
    pre, post, comb = _hc_gates(
        stream,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult,
        norm_eps,
        hc_eps,
        sinkhorn_iters,
    )
    if pre_mix is None:
        collapsed = _identity_collapse(stream, hc_mult)
    else:
        mix = pre_mix if isinstance(pre_mix, Tensor) else pre
        collapsed = _collapse(stream, mix).cast(stream.dtype)
    return post, comb, pre, collapsed


def mhc_shifted_post_pre(
    stream: Tensor,
    sublayer_out: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    pre_mix: Tensor | _PreMixOmitted | None = _PRE_MIX_OMITTED,
    norm_eps: float = 1e-20,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """One mHC boundary: finish the previous site, start this one.

    The previous site's output ``y`` (``sublayer_out``) is written back onto
    the stream with the previous site's ``post`` and ``comb`` *first*; this
    site's ``pre``/``post``/``comb`` and the collapse then both read the
    post-updated stream -- the same order as the reference's
    ``Block.forward``. This is :func:`mhc_post` followed by
    :func:`mhc_pre`.

    Do not use it when Engram runs at this layer: Engram must see the
    post-updated stream and the gates must see the Engram-injected one, so
    the caller composes :func:`mhc_post`, the Engram injection, and
    :func:`mhc_pre` instead. Injecting into ``stream`` before this call
    would mix the injection through the previous ``comb``.

    Args:
        stream: The parallel streams *before* the previous site's
            write-back, ``[rows, hc_mult, hidden]``.
        sublayer_out: The previous site's output, ``[rows, hidden]``.
        post_mix: The previous site's ``post``, float32 ``[rows, hc_mult]``.
        comb_mix: The previous site's ``comb``, float32 ``[rows, hc_mult,
            hc_mult]``.
        hc_fn: See :func:`mhc_pre`.
        hc_scale: See :func:`mhc_pre`.
        hc_base: See :func:`mhc_pre`.
        hc_mult: The number of parallel streams.
        pre_mix: The collapse weights; see :func:`mhc_pre` for the three
            states.
        norm_eps: See :func:`mhc_pre`.
        hc_eps: See :func:`mhc_pre`.
        sinkhorn_iters: See :func:`mhc_pre`.

    Returns:
        A tuple ``(stream_next, post, comb, pre, collapsed)``:

        - ``stream_next``: the post-updated stream, ``[rows, hc_mult,
          hidden]``, in ``stream``'s dtype; the one whose coefficients are
          ``post``/``comb``/``pre``.
        - ``post`` / ``comb``: this site's write-back coefficients, for the
          caller to feed the next boundary.
        - ``pre``: this site's collapse weights, float32 ``[rows, hc_mult]``
          -- the *next* site's ``pre_mix``, and what the final collapse
          consumes at the trunk's last layer.
        - ``collapsed``: this site's sublayer input, ``[rows, hidden]``, in
          ``stream``'s dtype, for the caller to normalize.
    """
    stream_next = mhc_post(sublayer_out, stream, post_mix, comb_mix, hc_mult)
    post, comb, pre, collapsed = mhc_pre(
        stream_next,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult,
        pre_mix,
        norm_eps,
        hc_eps,
        sinkhorn_iters,
    )
    return stream_next, post, comb, pre, collapsed


def mhc_post(
    sublayer_out: Tensor,
    stream: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
    hc_mult: int,
) -> Tensor:
    """Applies the previous site's ``post`` and ``comb`` to the stream.

    ``post_out[j] = post[j] * y + sum_i comb[i, j] * stream[i]`` -- the
    ``comb`` axis order: row ``i`` is the *input* stream index, column
    ``j`` the *output* stream index (the reference's
    ``einsum("...ij,...ih->...jh", comb, residual)``).

    Args:
        sublayer_out: The previous site's output ``y``, ``[rows, hidden]``.
        stream: The streams to mix, ``[rows, hc_mult, hidden]``.
        post_mix: ``post``, float32 ``[rows, hc_mult]``.
        comb_mix: ``comb``, float32 ``[rows, hc_mult, hc_mult]``.
        hc_mult: The number of parallel streams.

    Returns:
        The mixed stream, ``[rows, hc_mult, hidden]``, in ``stream``'s dtype.
    """
    y = sublayer_out.cast(DType.float32)
    # mixed[r, j, d] = sum_i comb[i, j] * stream[i, d]: contract comb's
    # first (input-stream) axis against the stream axis of `stream`.
    mixed = comb_mix.permute([0, 2, 1]) @ stream.cast(DType.float32)
    out = post_mix.unsqueeze(-1) * y.unsqueeze(-2) + mixed
    return out.cast(stream.dtype)


def mhc_initial(
    stream: Tensor,
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    norm_eps: float = 1e-20,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The trunk's first site: gates from the initial stream, identity carry.

    The embedding is broadcast to ``hc_mult`` copies, so the first attention
    site collapses with the identity mix (copy stream 0 unweighted, HF
    the reference): :func:`mhc_pre` with ``pre_mix=None`` on the
    broadcast embedding, with no previous post to apply.

    Returns:
        ``(post, comb, pre, collapsed)`` for the first attention site.
        ``collapsed`` is the layer-0 attention input -- stream 0, copied
        unweighted -- in ``stream``'s dtype, for the caller to normalize;
        the identity collapse is applied here because this function holds
        no norm weights.
    """
    return mhc_pre(
        stream,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult,
        None,
        norm_eps,
        hc_eps,
        sinkhorn_iters,
    )


def mhc_final_collapse(stream: Tensor, last_ffn_pre: Tensor) -> Tensor:
    """Collapses the final stream with the last FFN's ``pre`` -- no ``hc_head``.

    V4.1 has no ``hc_head`` weights: the trunk's last FFN ``hc_post``
    write-back produces the full un-collapsed stream, and the model output is
    the plain weighted sum over the stream axis using that same layer's own
    freshly computed FFN ``pre``, followed by the final ``norm`` (HF
    the reference). Collapsing the bare FFN output instead would
    silently skip that layer's write-back.

    Args:
        stream: The full stream after the last layer's FFN ``post``,
            ``[rows, hc_mult, hidden]``.
        last_ffn_pre: The last layer's FFN ``pre`` (the fourth value of the
            last :func:`mhc_shifted_post_pre` call), float32 ``[rows,
            hc_mult]``.

    Returns:
        ``[rows, hidden]``, in ``stream``'s dtype. The caller applies the
        final RMSNorm.
    """
    return _collapse(stream, last_ffn_pre).cast(stream.dtype)


def mhc_identity_writeback(stream: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """A write-back that leaves ``stream`` unchanged: ``(y, post, comb)``.

    ``y = 0``, ``post = 0`` and ``comb = I`` make :func:`mhc_post` return
    ``stream`` exactly (the sum is ``0 * 0 + 1 * stream`` in float32, then
    rounded to the bf16 it already was). It starts the trunk and follows
    Engram, where the stream is already post-updated, so one layer body
    serves every position.

    Returns:
        ``y`` bf16 ``[rows, hidden]``, ``post`` float32 ``[rows, hc_mult]``
        and the row-major ``comb`` float32 ``[rows, hc_mult * hc_mult]``.
    """
    rows, hidden = stream.shape[0], stream.shape[2]
    y = F.zeros([rows, hidden], dtype=DType.bfloat16, device=stream.mapping)
    post, comb = _identity_gates(stream)
    return y, post, comb


def _identity_gates(
    stream: Tensor,
) -> tuple[Tensor, Tensor]:
    """The identity write-back's ``post`` and row-major ``comb``."""
    rows, hc_mult = stream.shape[0], int(stream.shape[1])
    post = F.zeros([rows, hc_mult], dtype=DType.float32, device=stream.mapping)
    one = F.ones([rows, 1], dtype=DType.float32, device=stream.mapping)
    # [1, 0..0, 1, 0..0, ...]: a one every hc_mult + 1 entries, row-major.
    pieces = []
    for _ in range(hc_mult - 1):
        pieces += [one, post]
    pieces.append(one)
    return post, F.concat(pieces, axis=1)
