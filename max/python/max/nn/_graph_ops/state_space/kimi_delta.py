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
"""Python wrapper for the KDA (Kimi Delta Attention) recurrence op.

Also the home of the prefill-routing predicates and the gfx950 workspace
bound every KDA architecture shares, so no architecture imports them from
another.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from max.driver import accelerator_api, accelerator_architecture_name
from max.dtype import DType
from max.graph import (
    BufferValue,
    Dim,
    Graph,
    StaticDim,
    TensorType,
    TensorValue,
    ops,
)
from max.nn._graph_ops import _check_dtype, _check_rank, _check_same_dtype

__all__ = [
    "KDA_GATE_LOWER_BOUND",
    "STATE_LAYOUT",
    "KdaBetaMode",
    "KdaGateMode",
    "KdaInputMode",
    "KdaStateLayout",
    "amd_kda_chunk_eligible",
    "amd_kda_chunk_workspace_bytes",
    "amd_kda_long_row_scratch_bytes",
    "kda_chunk",
    "kda_chunk_conv",
    "kda_chunk_max_segments",
    "kda_chunk_predicate",
    "kda_chunk_rows",
    "kda_chunk_rows_servable",
    "kda_chunk_scratch_rows",
    "kda_chunk_servable",
    "kda_chunk_workspace_bytes",
    "kda_decode",
    "kda_gate_mode",
    "kda_ring_record_elements",
    "kda_state_fold",
    "kda_verify_ring",
    "kda_verify_ring_conv",
    "kda_verify_ring_conv_packed",
    "kda_verify_ring_kgroup_conv",
    "kda_verify_ring_kgroup_conv_servable",
]

KdaGateMode = Literal["original", "safe"]
"""``"original"`` is the stable-softplus gate; ``"safe"`` the bounded gate."""

KdaBetaMode = Literal["logits", "probability"]
"""Whether the kernel applies the sigmoid to ``beta_logits`` itself."""

KdaInputMode = Literal["activated", "conv_raw"]
"""Whether ``q``/``k``/``v`` arrive activated or as the conv's raw output."""

KdaStateLayout = Literal["K_FIRST", "V_FIRST"]
"""Pool axis order: ``[N, HV, K, V]`` or ``[N, HV, V, K]``."""

STATE_LAYOUT: KdaStateLayout = "K_FIRST"
"""Pool axis order, ``[num_rows, heads, head_dim, head_dim]``.

Pinned here so the recurrence ops and the caches' declared leaves cannot
disagree; it also matches the ``[num_rows, num_v_heads, key_dim,
val_dim]`` layout Qwen3.5's recurrent state cache declares.
"""

_SUPPORTED_HEAD_DIMS = ((32, 32), (128, 128))
"""``(key_head_dim, value_head_dim)`` pairs the op is compiled for."""

_AMD_CHUNK_HEAD_DIM = 128
"""Key and value head width of the AMD ``kda_chunk`` path."""

_AMD_CHUNK_ARCH = "gfx950"
"""The only AMD architecture the ``kda_chunk`` path is built for."""


def _is_amd_chunk_arch() -> bool:
    """Reports whether the accelerator is the AMD ``kda_chunk`` target."""
    return accelerator_architecture_name().startswith(_AMD_CHUNK_ARCH)


def kda_chunk_is_registered() -> bool:
    """Reports whether the current graph's kernel library has ``kda_chunk``.

    The kernel ships only with the closed-source kernel packages, so a
    source-only build lacks it. Callers that can fall back to the sequential
    recurrence should ask this, alongside
    :func:`kda_chunk_supports_head_dims`, before routing to :func:`kda_chunk`.

    Returns:
        ``True`` when ``kda_chunk`` can be added to the current graph.
    """
    return "kda_chunk" in Graph.current._kernel_library


def kda_chunk_supports_head_dims(
    key_head_dim: int, value_head_dim: int
) -> bool:
    """Reports whether ``kda_chunk`` is compiled for this head-dim pair.

    Callers that can fall back to the sequential recurrence should ask this
    before routing to :func:`kda_chunk`, which raises on an unsupported pair.
    On an AMD accelerator only gfx950 runs ``kda_chunk``, and only at
    ``(128, 128)`` with as many key heads as value heads (``Hk == Hv``),
    which this helper cannot see and the caller must ensure.

    Args:
        key_head_dim: Per-head key width.
        value_head_dim: Per-head value width.

    Returns:
        ``True`` when the pair is one the op is compiled for.
    """
    if accelerator_api() == "hip":
        return (
            _is_amd_chunk_arch()
            and key_head_dim == value_head_dim == _AMD_CHUNK_HEAD_DIM
        )
    return (key_head_dim, value_head_dim) in _SUPPORTED_HEAD_DIMS


def kda_chunk_rows_servable(key_head_dim: int, value_head_dim: int) -> bool:
    """Reports whether ``kda_chunk_rows`` can run this head geometry.

    Row select needs the sm100 fused or bt16 path, whose kernels only
    instantiate at a head width the fused arm accepts; a geometry outside
    that keeps the non-skip serial ring.

    Args:
        key_head_dim: Per-head key width.
        value_head_dim: Per-head value width.

    Returns:
        ``True`` when the fused arm accepts these head widths; the caller
        gates the accelerator and the registration alongside.
    """
    return (
        key_head_dim == value_head_dim
        and key_head_dim % 64 == 0
        and value_head_dim == 128
    )


def _check_amd_chunk_contract(
    q: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    state_pool: BufferValue,
    *,
    output_dtype: DType,
    gate_mode: KdaGateMode,
    use_computebound: bool,
) -> None:
    """Rejects what the AMD ``kda_chunk`` path does not implement.

    The op would reject these too, but at graph compile time or execution,
    with a diagnostic that names a Mojo parameter rather than the argument.
    """
    if not _is_amd_chunk_arch():
        raise ValueError(
            f"kda_chunk on AMD requires {_AMD_CHUNK_ARCH}, got "
            f"{accelerator_architecture_name()}"
        )
    if gate_mode != "safe":
        raise ValueError(
            f"kda_chunk on AMD supports gate_mode='safe' only, got {gate_mode!r}"
        )
    if use_computebound:
        raise ValueError("kda_chunk on AMD has no compute-bound variant")
    if (
        q.dtype != DType.bfloat16
        or output_dtype not in (DType.bfloat16, DType.float32)
        or raw_gate.dtype != DType.float32
        or state_pool.dtype != DType.float32
    ):
        raise ValueError(
            "kda_chunk on AMD takes bf16 q/k/v, a bf16 or fp32 output and fp32"
            " gates/state, "
            f"got q {q.dtype}, output {output_dtype}, raw_gate "
            f"{raw_gate.dtype}, state_pool {state_pool.dtype}"
        )
    _check_rank(3, q=q, v=v)
    dims = (q.shape[1], q.shape[2], v.shape[1], v.shape[2])
    if all(isinstance(d, StaticDim) for d in dims):
        got = tuple(int(d) for d in dims)
        num_key_heads, key_head_dim, num_value_heads, value_head_dim = got
        if (
            key_head_dim != _AMD_CHUNK_HEAD_DIM
            or value_head_dim != _AMD_CHUNK_HEAD_DIM
            or num_key_heads != num_value_heads
        ):
            raise ValueError(
                "kda_chunk on AMD requires key_head_dim == value_head_dim == "
                f"{_AMD_CHUNK_HEAD_DIM} and num_key_heads == num_value_heads, "
                "got (num_key_heads, key_head_dim, num_value_heads, "
                f"value_head_dim) = {got}"
            )


KDA_GATE_LOWER_BOUND = -5.0
"""Hard-coded lower bound for ``gate_mode="safe"``."""

_CHUNK_MIN_COMPUTE_CAPABILITY = 100
"""Lowest NVIDIA compute capability (``sm_100`` -> 100) ``kda_chunk`` builds for."""

_AMD_CHUNK_LEN = 32
"""Tokens per chunk of the AMD ``kda_chunk`` recurrence."""

_AMD_CU_CEILING = 512
"""Conservative CU count standing in for the device's in the single-sequence
segment rule of :func:`kda_chunk_max_segments`."""


def kda_gate_mode(lower_bound: float | None) -> KdaGateMode:
    """Selects the kernel's forget-gate form from the config's lower bound.

    Raises:
        ValueError: If the bound is neither absent nor the one value the
            ``"safe"`` kernel path hard-codes. The constant lives in the
            kernel, so a checkpoint with a different bound would run and be
            quietly wrong.
    """
    if lower_bound is None:
        return "original"
    if lower_bound != KDA_GATE_LOWER_BOUND:
        raise ValueError(
            "KDA's bounded forget gate is compiled with a lower bound of "
            f"{KDA_GATE_LOWER_BOUND}, but this checkpoint asks for "
            f"{lower_bound}. Thread a new compile-time axis through the kernel "
            "rather than accepting the mismatch."
        )
    return "safe"


def kda_chunk_servable(
    head_dim: int, gate_mode: KdaGateMode, pool_dtype: DType
) -> bool:
    """Returns whether prefill can take ``kda_chunk``, not ``kda_decode``.

    The op's NVIDIA kernels only instantiate on SM100+, and its AMD kernels
    only on gfx950, where they also take only the ``"safe"`` gate and a
    float32 pool. The AMD kernels take bf16 q/k/v and a bf16 or fp32 output,
    so callers cast at the op boundary there. Builds without the closed-source kernels lack
    the op. Anything outside that keeps the sequential recurrence for prefill.

    Call this while a graph is being built: it reads that graph's kernel
    library.

    Args:
        head_dim: Key and value head width.
        gate_mode: The forget-gate form the layer passes the op.
        pool_dtype: Dtype of the recurrent state pool.

    Returns:
        ``True`` when ``kda_chunk`` accepts this layer's arguments.
    """
    api = accelerator_api()
    if api == "hip":
        return (
            amd_kda_chunk_eligible(head_dim, gate_mode, pool_dtype)
            and kda_chunk_is_registered()
        )
    # TODO(KERN-3635): widen once kda_chunk compiles below SM100.
    name = accelerator_architecture_name()
    digits = name.removeprefix("sm_").rstrip("abcdefghijklmnopqrstuvwxyz")
    return (
        api == "cuda"
        and digits.isdigit()
        and int(digits) >= _CHUNK_MIN_COMPUTE_CAPABILITY
        and kda_chunk_supports_head_dims(head_dim, head_dim)
        and kda_chunk_is_registered()
    )


def amd_kda_chunk_eligible(
    head_dim: int, gate_mode: KdaGateMode, pool_dtype: DType
) -> bool:
    """Returns whether gfx950 prefill takes ``kda_chunk``, registration aside.

    The part of :func:`kda_chunk_servable` that needs no graph, so the memory
    planner can ask it too.

    Args:
        head_dim: Key and value head width.
        gate_mode: The forget-gate form the layer passes the op.
        pool_dtype: Dtype of the recurrent state pool.

    Returns:
        ``True`` on a gfx950 accelerator whose AMD ``kda_chunk`` contract
        these arguments meet.
    """
    return (
        accelerator_api() == "hip"
        and _is_amd_chunk_arch()
        and gate_mode == "safe"
        and pool_dtype == DType.float32
        and kda_chunk_supports_head_dims(head_dim, head_dim)
    )


def kda_chunk_bf16_launch() -> tuple[DType, bool]:
    """Returns how :func:`kda_chunk` takes bf16 q/k/v on this accelerator.

    On gfx950 the output is bf16 and there is no compute-bound variant. On
    NVIDIA a float32 output with ``use_computebound`` selects the fused
    tensor-core kernel; float32 q/k/v would fall back to the L1/L2/L3
    pipeline, whose scan runs one CTA per (sequence, head) and is ~8x slower
    at 8K tokens.

    Returns:
        ``(output_dtype, use_computebound)`` to pass :func:`kda_chunk`.
    """
    if accelerator_api() == "hip":
        return DType.bfloat16, False
    return DType.float32, True


def kda_chunk_predicate(
    total_tokens: TensorValue, batch_size: TensorValue
) -> TensorValue:
    """Returns whether a batch takes the ``kda_chunk`` side of the cond.

    A batch with more tokens than sequences is a prefill. The gfx950 launch
    indexes in 64 bits, so no size bound applies on any accelerator. Mirrors
    ``kda_amd_decode_runs_chunk_pipeline`` in
    ``Kernels/lib/kda/amd_prefill.mojo``; keep the two in step.

    Args:
        total_tokens: Scalar token count of the batch.
        batch_size: Scalar sequence count of the batch.

    Returns:
        A boolean scalar.
    """
    return total_tokens > batch_size


def kda_chunk_max_segments(tokens: int, seqs: int, heads: int) -> int:
    """Largest segment count the AMD ``kda_chunk`` launch picks for a shape.

    Mirrors ``kda_max_batch_segments`` in
    ``Kernels/lib/kda/amd_prefill.mojo``; the launcher uses the
    device's CU count, so the single-sequence residency clamp substitutes a
    512-CU ceiling, and the launcher's further one-wave clamp, which only
    lowers the count, is left out. The result is never below what the
    launcher picks, which the workspace bound relies on. Keep the two
    expressions in sync.

    Args:
        tokens: Total tokens across the batch.
        seqs: Number of sequences (rows of ``state_indices``).
        heads: Number of value heads.

    Returns:
        The largest per-sequence segment count the launch can use.
    """
    chunks = -(-tokens // _AMD_CHUNK_LEN)
    if seqs > 1:
        avg_chunks = -(-chunks // seqs)
        if avg_chunks < 40:
            return 1
        return min(12, max(8, (avg_chunks + 15) // 16))
    if tokens <= 2048:
        segments = max(1, (tokens + 255) // 256)
    elif tokens <= 8192:
        segments = (tokens + 511) // 512
    else:
        segments = (tokens + 1023) // 1024
    segments = min(segments, max(1, _AMD_CU_CEILING // heads))
    if 2048 < tokens <= 8192:
        # The launcher may raise a mid-length sequence's count to fill one
        # replay wave, never past 12 chunks per segment.
        fill = min(max(1, _AMD_CU_CEILING // (2 * heads)), -(-chunks // 12))
        segments = max(segments, fill)
    return segments


def kda_chunk_scratch_rows(tokens: int, seqs: int, heads: int) -> int:
    """Workspace of one AMD ``kda_chunk`` launch, in output-row units.

    One row holds ``heads * 128`` bf16 elements: 138 rows per 32-token
    chunk -- four bf16 chunk matrices, the bf16 within-chunk score block
    and the fp32 gamma strip -- plus, when the launch splits sequences into
    segments, 768 rows per (sequence, segment) of fp32 segment maps and
    segment states.

    Args:
        tokens: Total tokens across the batch.
        seqs: Number of sequences (rows of ``state_indices``).
        heads: Number of value heads.

    Returns:
        Workspace rows the launch allocates (an upper bound).
    """
    max_chunks = -(-tokens // _AMD_CHUNK_LEN) + seqs
    max_segments = kda_chunk_max_segments(tokens, seqs, heads)
    segment_slots = seqs * max_segments if max_segments > 1 else 0
    rows_per_chunk = 138
    rows_per_segment = 768
    return max_chunks * rows_per_chunk + segment_slots * rows_per_segment


def kda_chunk_workspace_bytes(
    num_heads: int, head_dim: int, batch_tokens: int, batch_size: int
) -> int:
    """Returns an upper bound on one gfx950 ``kda_chunk`` call's workspace.

    Uses :func:`kda_chunk_scratch_rows`, which mirrors the launcher's
    workspace layout in ``Kernels/lib/kda/amd_prefill.mojo``: per
    32-token chunk and per (sequence, segment), in rows of ``heads *
    head_dim`` bf16 elements. The segment count only grows as a device holds
    fewer heads, so rows at one head bound every device of any head split,
    and the bytes for a set of devices is at most those rows times their
    heads. The bound covers every batch the chunk path runs (see
    :func:`kda_chunk_predicate`: more tokens than sequences) of at most
    ``batch_tokens`` tokens and ``batch_size`` sequences, not only the full
    one. A batch with no more tokens than sequences takes the token-serial
    kernel and needs no workspace, so the sweep stops at ``tokens``
    sequences.

    Args:
        num_heads: KDA heads the reservation sizes.
        head_dim: Key and value head width.
        batch_tokens: Most tokens one execute prefills.
        batch_size: Most sequences one execute prefills.

    Returns:
        Workspace bytes for ``num_heads``.
    """
    assert head_dim == _AMD_CHUNK_HEAD_DIM, (
        "the gfx950 kda_chunk kernel needs 128-wide heads"
    )
    # The workspace is not monotone in either count: fewer, longer sequences
    # cross the threshold where the launch starts splitting them into
    # segments, and the single-sequence segment rule steps down just past
    # 2048 and 8192 tokens. Past one sequence per chunk nothing splits and
    # the rows only grow with the sequence count.
    rows = 0
    for tokens in {batch_tokens} | {
        t for t in (2048, 8192) if t < batch_tokens
    }:
        chunks = -(-tokens // _AMD_CHUNK_LEN)
        largest = max(1, min(batch_size, tokens))
        for seqs in [*range(1, min(largest, chunks) + 1), largest]:
            rows = max(rows, kda_chunk_scratch_rows(tokens, seqs, 1))
    return rows * num_heads * head_dim * DType.bfloat16.size_in_bytes


def amd_kda_chunk_workspace_bytes(
    *,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
    state_dtype: DType,
    decode_only: bool,
    max_batch_input_tokens: int,
    enable_chunked_prefill: bool,
    max_batch_size: int | None,
    max_length: int | None,
) -> int:
    """Returns what to reserve for gfx950 ``kda_chunk`` prefill, or ``0``.

    Nonzero only where AMD prefill is eligible for ``kda_chunk``; see
    :func:`amd_kda_chunk_eligible`. The runtime numbers arrive as plain
    arguments because this module cannot import the runtime config that
    holds them.

    Args:
        num_heads: KDA heads the reservation sizes. Every term is per head,
            so pass one device's heads and multiply by the device count, or
            every device's heads for the whole cluster.
        head_dim: Key and value head width.
        lower_bound: The forget gate's lower bound, ``None`` for the
            unbounded gate.
        state_dtype: Dtype of the recurrent state pool.
        decode_only: Whether the pipeline never prefills.
        max_batch_input_tokens: Most tokens one execute prefills.
        enable_chunked_prefill: Whether the scheduler splits long prompts.
        max_batch_size: Most sequences in one batch, or ``None`` when not
            yet known.
        max_length: Longest sequence the model serves, or ``None`` when
            unset.

    Returns:
        Workspace bytes for ``num_heads``.
    """
    if decode_only or not amd_kda_chunk_eligible(
        head_dim, kda_gate_mode(lower_bound), state_dtype
    ):
        return 0
    tokens = max_batch_input_tokens
    # Without chunked prefill the token budget is soft: the scheduler admits
    # an oversized prompt whole on top of a full batch.
    if not enable_chunked_prefill:
        tokens += max_length or 0
    if max_batch_size and max_length:
        tokens = min(tokens, max_batch_size * max_length)
    sequences = min(max_batch_size or tokens, tokens)
    return kda_chunk_workspace_bytes(num_heads, head_dim, tokens, sequences)


def amd_kda_long_row_scratch_bytes(
    *,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
    state_dtype: DType,
    max_step_tokens: int,
) -> int:
    """Returns what a mixed verify's gfx950 long-row path allocates, or ``0``.

    ``kda_verify_ring`` with ``chunk_long_rows`` gathers a mixed verify's
    prompt rows into dense scratch sized by the step's tokens, outside the
    graph's memory plan: bf16 q, k and v, fp32 gate and output, and an fp32
    beta per head. One layer's call holds it at a time, so it is reserved
    once.

    Args:
        num_heads: KDA heads the reservation sizes, as for
            :func:`amd_kda_chunk_workspace_bytes`.
        head_dim: Key and value head width.
        lower_bound: The forget gate's lower bound, ``None`` for the
            unbounded gate.
        state_dtype: Dtype of the recurrent state pool.
        max_step_tokens: Most tokens one verify step carries, drafts
            included.

    Returns:
        Scratch bytes for ``num_heads``.
    """
    if not amd_kda_chunk_eligible(
        head_dim, kda_gate_mode(lower_bound), state_dtype
    ):
        return 0
    per_token_head = 3 * head_dim * 2 + 2 * head_dim * 4 + 4
    return max_step_tokens * num_heads * per_token_head


def _check_same_dim(axis: str, **dims: Dim) -> None:
    """Raises ``ValueError`` unless every named dim is the same.

    Two dims that are both symbolic but differently named are rejected: the
    op requires them equal and nothing here can prove that they are.

    Note: The kwarg names are used in the error message, so naming matters.
    """
    first_name, first = next(iter(dims.items()))
    for name, dim in list(dims.items())[1:]:
        if dim != first:
            raise ValueError(
                f"expected {first_name} and {name} to agree on {axis}, but "
                f"got {first} and {dim}, respectively"
            )


def _check_dtypes(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_indices: TensorValue,
) -> None:
    """Rejects the dtype groupings the op cannot express."""
    _check_same_dtype(q=q, k=k, v=v)
    _check_same_dtype(
        raw_gate=raw_gate,
        beta_logits=beta_logits,
        a_log=a_log,
        dt_bias=dt_bias,
    )
    _check_dtype(
        DType.int32, cu_seqlens=cu_seqlens, state_indices=state_indices
    )


def _check_shapes(
    op_name: str,
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    state_layout: KdaStateLayout,
    input_mode: KdaInputMode = "activated",
) -> None:
    """Rejects the shape combinations the kernel would read out-of-bounds on."""
    _check_rank(3, q=q, k=k, v=v, raw_gate=raw_gate)
    _check_rank(2, beta_logits=beta_logits, dt_bias=dt_bias)
    _check_rank(
        1, a_log=a_log, cu_seqlens=cu_seqlens, state_indices=state_indices
    )
    _check_rank(4, state_pool=state_pool)

    total_tokens = q.shape[0]
    num_key_heads = q.shape[1]
    key_head_dim = q.shape[2]
    num_value_heads = v.shape[1]
    value_head_dim = v.shape[2]

    _check_same_dim(
        "the packed token axis",
        q=total_tokens,
        k=k.shape[0],
        v=v.shape[0],
        raw_gate=raw_gate.shape[0],
        beta_logits=beta_logits.shape[0],
    )
    # `k` shares `q`'s head extents exactly -- this is the pair whose mismatch
    # the kernel reads out of bounds on.
    _check_same_dim("the key-head axis", q=num_key_heads, k=k.shape[1])
    _check_same_dim("the key-head width", q=key_head_dim, k=k.shape[2])
    if input_mode == "conv_raw":
        # `a_log` is the per-channel `[HV, K]` parameter, packed flat for the
        # op's rank-1 slot by `kda_decode`, so it is checked against the flat
        # channel count here and stays out of the head-axis agreement below.
        expected_a_log = num_value_heads * key_head_dim
        if a_log.shape[0] != expected_a_log:
            raise ValueError(
                "expected conv_raw a_log to hold num_value_heads *"
                f" key_head_dim = {expected_a_log} entries, was"
                f" {a_log.shape[0]}"
            )
        _check_same_dim(
            "the value-head axis",
            v=num_value_heads,
            raw_gate=raw_gate.shape[1],
            beta_logits=beta_logits.shape[1],
            dt_bias=dt_bias.shape[0],
            state_pool=state_pool.shape[1],
        )
    else:
        _check_same_dim(
            "the value-head axis",
            v=num_value_heads,
            raw_gate=raw_gate.shape[1],
            beta_logits=beta_logits.shape[1],
            a_log=a_log.shape[0],
            dt_bias=dt_bias.shape[0],
            state_pool=state_pool.shape[1],
        )
    _check_same_dim(
        "the key-head width",
        q=key_head_dim,
        raw_gate=raw_gate.shape[2],
        dt_bias=dt_bias.shape[1],
    )

    # The pool's trailing axes are ordered by `state_layout`; getting this
    # wrong transposes the state rather than failing a shape check.
    pool_axes = (state_pool.shape[2], state_pool.shape[3])
    expected = (
        (
            key_head_dim,
            value_head_dim,
        )
        if state_layout == "K_FIRST"
        else (value_head_dim, key_head_dim)
    )
    if pool_axes != expected:
        raise ValueError(
            f"expected state_pool to have trailing axes {expected} for "
            f"state_layout={state_layout!r}, was {pool_axes}"
        )

    if isinstance(cu_seqlens.shape[0], StaticDim) and isinstance(
        state_indices.shape[0], StaticDim
    ):
        if int(cu_seqlens.shape[0]) != int(state_indices.shape[0]) + 1:
            raise ValueError(
                "expected cu_seqlens to hold one more entry than "
                f"state_indices, was {cu_seqlens.shape[0]} and "
                f"{state_indices.shape[0]}"
            )

    if isinstance(num_key_heads, StaticDim) and isinstance(
        num_value_heads, StaticDim
    ):
        if int(num_value_heads) % int(num_key_heads):
            raise ValueError(
                "expected v's head count to be divisible by q's, so that a "
                f"value head maps to a key head, but got {num_value_heads} "
                f"and {num_key_heads}"
            )

    # The kernels index the pool at 64 bits but take its strides as 32-bit
    # values, so one slot (every head of one request) must stay below 2**32
    # elements. The slot count is a runtime dim; the slab binding in
    # max.nn.legacy.kv_cache.cache_params bounds it.
    if all(
        isinstance(d, StaticDim)
        for d in (num_value_heads, key_head_dim, value_head_dim)
    ):
        slot_elements = (
            int(num_value_heads) * int(key_head_dim) * int(value_head_dim)
        )
        if slot_elements >= 2**32:
            raise ValueError(
                f"{op_name}: a state_pool slot holds {slot_elements} "
                "elements, but the kernel addresses a slot with 32-bit "
                "strides, so it must hold fewer than 2**32"
            )

    if isinstance(key_head_dim, StaticDim) and isinstance(
        value_head_dim, StaticDim
    ):
        pair = (int(key_head_dim), int(value_head_dim))
        if pair not in _SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"{op_name} is compiled for (key_head_dim, value_head_dim) "
                f"in {list(_SUPPORTED_HEAD_DIMS)}, got {pair}"
            )


def _kda_recurrence(
    op_name: Literal["kda_decode", "kda_chunk"],
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    *,
    output_dtype: DType,
    gate_mode: KdaGateMode,
    beta_mode: KdaBetaMode,
    state_layout: KdaStateLayout,
    use_computebound: bool = False,
    pin_fma: bool = False,
    input_mode: KdaInputMode = "activated",
    activation_dtype: DType | None = None,
) -> TensorValue:
    """Shared implementation for ``kda_decode`` and ``kda_chunk``."""
    if op_name != "kda_decode" and input_mode != "activated":
        raise ValueError(
            f"{op_name}: input_mode is a kda_decode axis, got {input_mode!r}"
        )
    _check_dtypes(
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
    )
    _check_shapes(
        op_name,
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        state_layout,
        input_mode,
    )
    total_tokens, num_value_heads, value_head_dim = v.shape
    parameters: dict[str, bool | int | str | DType] = {
        "gate_mode": gate_mode,
        "beta_mode": beta_mode,
        "state_layout": state_layout,
    }
    if op_name == "kda_chunk":
        parameters["use_computebound"] = use_computebound
    else:
        parameters["pin_fma"] = pin_fma
        if input_mode != "activated":
            parameters["input_mode"] = input_mode
            parameters["activation_dtype"] = (
                activation_dtype
                if activation_dtype is not None
                else DType.float32
            )
    results = ops.inplace_custom(
        op_name,
        v.device,
        [
            ops.unsqueeze(q, 0),
            ops.unsqueeze(k, 0),
            ops.unsqueeze(v, 0),
            ops.unsqueeze(raw_gate, 0),
            ops.unsqueeze(beta_logits, 0),
            a_log,
            dt_bias,
            cu_seqlens,
            state_pool,
            state_indices,
        ],
        [
            TensorType(
                output_dtype,
                [1, total_tokens, num_value_heads, value_head_dim],
                v.device,
            )
        ],
        parameters=parameters,
    )
    return ops.squeeze(cast(TensorValue, results[0]), 0)


def kda_decode(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    *,
    output_dtype: DType,
    gate_mode: KdaGateMode = "original",
    beta_mode: KdaBetaMode = "logits",
    state_layout: KdaStateLayout = "K_FIRST",
    pin_fma: bool = False,
    input_mode: KdaInputMode = "activated",
    activation_dtype: DType | None = None,
) -> TensorValue:
    """Runs the KDA recurrence, mutating ``state_pool`` in place.

    ``q`` and ``k`` are L2-normalized and ``q`` scaled by
    ``1 / sqrt(key_head_dim)`` inside the kernel, so pass them raw. The gate
    and beta activations are folded in too, selected by ``gate_mode`` and
    ``beta_mode``.

    Under ``input_mode="conv_raw"`` the kernel also folds the pre-decode
    elementwise passes: ``q``/``k``/``v`` arrive as the conv's raw fp32
    output (the kernel applies SiLU, rounds to ``activation_dtype``, and
    folds the safe gate's fp32 per-channel pre-activation
    ``(raw_gate + dt_bias) * exp(a_log)``), and ``a_log`` is the
    per-channel ``[num_value_heads, key_head_dim]`` parameter, packed flat
    here for the op's rank-1 slot. The op routes on the shapes it holds: a
    decode-shaped batch (one token per sequence) runs that fused prologue,
    while any other batch first runs one in-op activation launch and then
    continues on the chunk pipeline or the token-serial kernel, exactly as
    an ``"activated"`` call would.

    Args:
        q: ``[total_tokens, num_key_heads, key_head_dim]``.
        k: ``[total_tokens, num_key_heads, key_head_dim]``.
        v: ``[total_tokens, num_value_heads, value_head_dim]``.
        raw_gate: ``[total_tokens, num_value_heads, key_head_dim]``
            forget-gate pre-activation, before ``dt_bias`` is added.
        beta_logits: ``[1, total_tokens, num_value_heads]`` (rank 3).
        a_log: ``[num_value_heads]``.
        dt_bias: ``[num_value_heads, key_head_dim]``.
        cu_seqlens: ``[batch_size + 1]`` int32 exclusive prefix offsets.
        state_pool: ``[max_slots, num_value_heads, key_head_dim,
            value_head_dim]`` mutable pool, laid out per ``state_layout``.
        state_indices: ``[batch_size]`` int32 pool slot per sequence.
        output_dtype: Dtype of the returned tensor.
        gate_mode: Forget-gate form; see :data:`KDA_GATE_LOWER_BOUND` before
            selecting ``"safe"``.
        beta_mode: Whether the kernel applies the sigmoid to ``beta_logits``.
        state_layout: Pool axis order; must match how the pool was allocated.
        pin_fma: Whether to spell every multiply-add as an explicit ``fma``,
            so the speculative verify and fold kernels reproduce this one bit
            for bit. Always on for NVIDIA; slower on AMD.
        input_mode: "conv_raw" folds SiLU, the model-dtype rounding and the
            safe gate's pre-activation into the kernel; needs
            ``gate_mode="safe"`` and fp32 q/k/v. Default keeps the activated
            operand convention.
        activation_dtype: The model dtype the fused SiLU result rounds to;
            required by ``input_mode="conv_raw"``.

    Returns:
        ``[total_tokens, num_value_heads, value_head_dim]``.

    Raises:
        ValueError: If the dtype groupings bind no kernel, or the ranks and
            extents disagree. The kernel's own shape guards are
            ``debug_assert``, so they are absent from a production build.
    """
    if input_mode == "conv_raw":
        if gate_mode != "safe":
            raise ValueError(
                "kda_decode: input_mode 'conv_raw' folds the safe gate's"
                " pre-activation, so gate_mode must be 'safe', got"
                f" {gate_mode!r}"
            )
        if activation_dtype is None:
            raise ValueError(
                "kda_decode: input_mode 'conv_raw' needs activation_dtype,"
                " the model dtype the fused SiLU result rounds to"
            )
        if q.dtype != DType.float32:
            raise ValueError(
                "kda_decode: input_mode 'conv_raw' takes the conv's raw fp32"
                f" q/k/v, but q is {q.dtype}"
            )
        _check_rank(2, a_log=a_log)
        if not all(isinstance(d, StaticDim) for d in a_log.shape):
            raise ValueError(
                "kda_decode: input_mode 'conv_raw' packs the per-channel"
                " [num_value_heads, key_head_dim] a_log flat, so its"
                f" extents must be static, was {list(a_log.shape)}"
            )
        a_log = ops.reshape(a_log, [int(a_log.shape[0]) * int(a_log.shape[1])])
    return _kda_recurrence(
        "kda_decode",
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        output_dtype=output_dtype,
        gate_mode=gate_mode,
        beta_mode=beta_mode,
        state_layout=state_layout,
        pin_fma=pin_fma,
        input_mode=input_mode,
        activation_dtype=activation_dtype,
    )


def kda_chunk(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    *,
    output_dtype: DType,
    use_computebound: bool = False,
    gate_mode: KdaGateMode = "original",
    beta_mode: KdaBetaMode = "logits",
    state_layout: KdaStateLayout = "K_FIRST",
) -> TensorValue:
    """Runs the KDA chunk-parallel prefill recurrence, mutating ``state_pool``.

    This is the chunk-parallel prefill path: same recurrence as
    ``kda_decode`` but evaluated in parallel across 16-token chunks.  It is
    intended for multi-token sequences; single-token decode should stay on
    ``kda_decode``.

    ``use_computebound=True`` selects the fused compute-bound kernel instead
    of the L1/L2/L3 pipeline on supported shapes.

    On an AMD accelerator (gfx950 only) the op runs a separate pipeline over
    32-token chunks with a narrower contract: ``key_head_dim ==
    value_head_dim == 128``, ``num_key_heads == num_value_heads``,
    ``gate_mode="safe"``, bf16 ``q``/``k``/``v``, a bf16 or fp32 output, fp32
    gates and state, and ``use_computebound=False``.

    Args:
        q: ``[total_tokens, num_key_heads, key_head_dim]``.
        k: ``[total_tokens, num_key_heads, key_head_dim]``.
        v: ``[total_tokens, num_value_heads, value_head_dim]``.
        raw_gate: ``[total_tokens, num_value_heads, key_head_dim]``
            forget-gate pre-activation, before ``dt_bias`` is added.
        beta_logits: ``[total_tokens, num_value_heads]``.
        a_log: ``[num_value_heads]``.
        dt_bias: ``[num_value_heads, key_head_dim]``.
        cu_seqlens: ``[batch_size + 1]`` int32 exclusive prefix offsets.
        state_pool: ``[max_slots, num_value_heads, key_head_dim,
            value_head_dim]`` mutable pool, laid out per ``state_layout``.
        state_indices: ``[batch_size]`` int32 pool slot per sequence.
        output_dtype: Dtype of the returned tensor.
        use_computebound: Whether to dispatch the fused compute-bound kernel.
        gate_mode: Forget-gate form; see :data:`KDA_GATE_LOWER_BOUND` before
            selecting ``"safe"``.
        beta_mode: Whether the kernel applies the sigmoid to ``beta_logits``.
        state_layout: Pool axis order; must match how the pool was allocated.

    Returns:
        ``[total_tokens, num_value_heads, value_head_dim]``.

    Raises:
        ValueError: If the dtype groupings bind no kernel, the ranks and
            extents disagree, or, on AMD, the arguments fall outside the AMD
            contract. The kernel's own shape guards are ``debug_assert`` on
            NVIDIA, so they are absent from a production build.
    """
    if accelerator_api() == "hip":
        _check_amd_chunk_contract(
            q,
            v,
            raw_gate,
            state_pool,
            output_dtype=output_dtype,
            gate_mode=gate_mode,
            use_computebound=use_computebound,
        )
    return _kda_recurrence(
        "kda_chunk",
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        output_dtype=output_dtype,
        gate_mode=gate_mode,
        beta_mode=beta_mode,
        state_layout=state_layout,
        use_computebound=use_computebound,
    )


def kda_chunk_rows(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    passthrough: TensorValue,
    verify_width: TensorValue,
    *,
    output_dtype: DType,
    gate_mode: KdaGateMode = "original",
    beta_mode: KdaBetaMode = "logits",
    state_layout: KdaStateLayout = "K_FIRST",
) -> TensorValue:
    """Runs the KDA chunk recurrence over the rows a ring leaves to the caller.

    A row longer than one token and its drafts is walked over its first
    ``len - K`` tokens, where ``K`` is ``verify_width``'s length; every
    other row keeps the values ``passthrough`` holds, including its pool
    row, whose updates the ring records for the fold.

    Args:
        q: ``[total_tokens, num_key_heads, key_head_dim]``; as for
            :func:`kda_chunk`.
        k: ``[total_tokens, num_key_heads, key_head_dim]``; as for
            :func:`kda_chunk`.
        v: ``[total_tokens, num_value_heads, value_head_dim]``; as for
            :func:`kda_chunk`.
        raw_gate: Forget-gate pre-activation; as for :func:`kda_chunk`.
        beta_logits: Per-token delta-rule beta; as for :func:`kda_chunk`.
        a_log: Per-head gate scale; as for :func:`kda_chunk`.
        dt_bias: Per-key gate bias; as for :func:`kda_chunk`.
        cu_seqlens: ``[batch_size + 1]`` int32 exclusive prefix offsets; as
            for :func:`kda_chunk`.
        state_pool: Mutable pool, laid out per ``state_layout``; as for
            :func:`kda_chunk`.
        state_indices: ``[batch_size]`` int32 pool slot per sequence; as for
            :func:`kda_chunk`.
        passthrough: ``[total_tokens, num_value_heads, value_head_dim]``
            float32 readout to start from.
        verify_width: The operand
            :func:`~max.nn._graph_ops.state_space.verify_width_operand` returns.
        output_dtype: Dtype of the returned tensor; must be float32.
        gate_mode: Forget-gate form; as for :func:`kda_chunk`.
        beta_mode: Whether the kernel applies the sigmoid to ``beta_logits``;
            as for :func:`kda_chunk`.
        state_layout: Pool axis order; as for :func:`kda_chunk`.

    Returns:
        ``[total_tokens, num_value_heads, value_head_dim]``.
    """
    if output_dtype != DType.float32:
        raise ValueError(
            f"kda_chunk_rows starts from an fp32 passthrough, got {output_dtype}"
        )
    _check_dtype(DType.float32, passthrough=passthrough)
    _check_dtype(DType.int64, verify_width=verify_width)
    _check_rank(3, passthrough=passthrough)
    _check_rank(1, verify_width=verify_width)
    # The launcher's copy is sized by the destination, so a short passthrough
    # would read past its end.
    _check_same_dim(
        "the packed token axis", v=v.shape[0], passthrough=passthrough.shape[0]
    )
    _check_same_dim(
        "the value-head axis", v=v.shape[1], passthrough=passthrough.shape[1]
    )
    _check_same_dim(
        "the value-head width", v=v.shape[2], passthrough=passthrough.shape[2]
    )
    _check_dtypes(
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
    )
    _check_shapes(
        "kda_chunk_rows",
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        state_layout,
    )
    total_tokens, num_value_heads, value_head_dim = v.shape
    results = ops.inplace_custom(
        "kda_chunk_rows",
        v.device,
        [
            ops.unsqueeze(q, 0),
            ops.unsqueeze(k, 0),
            ops.unsqueeze(v, 0),
            ops.unsqueeze(raw_gate, 0),
            ops.unsqueeze(beta_logits, 0),
            a_log,
            dt_bias,
            cu_seqlens,
            state_pool,
            state_indices,
            ops.unsqueeze(passthrough, 0),
            verify_width,
        ],
        [
            TensorType(
                output_dtype,
                [1, total_tokens, num_value_heads, value_head_dim],
                v.device,
            )
        ],
        parameters={
            "gate_mode": gate_mode,
            "beta_mode": beta_mode,
            "state_layout": state_layout,
        },
    )
    return ops.squeeze(cast(TensorValue, results[0]), 0)


def kda_ring_record_elements(key_head_dim: int, value_head_dim: int) -> int:
    """Returns the float32 elements one ring record holds, before padding.

    A record is one token's update to one head's state, laid out as the
    per-channel decay ``exp(g)``, then the raw (not L2-normalized) key, then
    the update factor ``beta * (v - S^T k_raw * inv_norm) * inv_norm``.
    Folding it is ``S = diag(decay) S + key update_factor^T``, the
    recurrence's own update with its operands and rounding, so a fold over the
    first ``j`` records lands bit-for-bit the state :func:`kda_decode` holds
    after ``j`` tokens. The key is stored raw because ``k_hat * delta`` and
    ``k_raw * (delta * inv_norm)`` round differently.

    Args:
        key_head_dim: The key head width.
        value_head_dim: The value head width.
    """
    return 2 * key_head_dim + value_head_dim


def kda_verify_ring(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    ring: BufferValue,
    ring_indices: TensorValue,
    verify_width: TensorValue,
    *,
    output_dtype: DType,
    gate_mode: KdaGateMode = "original",
    beta_mode: KdaBetaMode = "logits",
    state_layout: KdaStateLayout = "K_FIRST",
    chunk_long_rows: bool = False,
    kgroup: bool = False,
) -> TensorValue:
    """Runs the KDA recurrence over a speculative verify window into a ring.

    Produces the same readout as :func:`kda_decode` without writing
    ``state_pool``. Token ``t`` of a request writes its record (see
    :func:`kda_ring_record_elements`) to slot ``t`` of its ``ring`` row,
    which :func:`kda_state_fold` applies once acceptance is known.

    At a verify width of zero this is :func:`kda_decode` on ``state_pool``,
    and no ring is written. A request longer than the ring's ``RING_LEN``
    records nothing and commits ``state_pool`` itself, the same contract as
    :func:`~max.nn._graph_ops.state_space.gated_delta_recurrence_verify_ring_fwd`.

    With ``chunk_long_rows``, a mixed verify's prompt rows -- longer than one
    token and its drafts -- run instead on the gfx950 chunk pipeline over
    their tokens before the drafts, committing ``state_pool`` there and
    recording nothing; their drafts' outputs are zeros. A decode-only verify
    launches exactly what it does without it, and a mixed verify off gfx950
    raises.

    Args:
        q: ``[total_tokens, num_key_heads, key_head_dim]``.
        k: ``[total_tokens, num_key_heads, key_head_dim]``.
        v: ``[total_tokens, num_value_heads, value_head_dim]``.
        raw_gate: ``[total_tokens, num_value_heads, key_head_dim]``
            forget-gate pre-activation, before ``dt_bias`` is added.
        beta_logits: ``[total_tokens, num_value_heads]``.
        a_log: ``[num_value_heads]``.
        dt_bias: ``[num_value_heads, key_head_dim]``.
        cu_seqlens: ``[batch_size + 1]`` int32 exclusive prefix offsets.
        state_pool: ``[max_slots, num_value_heads, key_head_dim,
            value_head_dim]`` live pool, written only at a verify width of
            zero or for a request longer than the ring.
        state_indices: ``[batch_size]`` int32 pool slot per sequence.
        ring: ``[ring_rows, num_value_heads, RING_LEN, record_stride]``
            float32 ring pool. ``record_stride`` is at least
            :func:`kda_ring_record_elements`.
        ring_indices: ``[batch_size]`` int32 ring row per sequence.
        verify_width: The operand
            :func:`~max.nn._graph_ops.state_space.verify_width_operand` returns.
        output_dtype: Dtype of the returned tensor.
        gate_mode: Forget-gate form, as for :func:`kda_decode`.
        beta_mode: Whether the kernel applies the sigmoid to ``beta_logits``.
        state_layout: Pool axis order; must match how the pool was allocated.
        chunk_long_rows: Whether a mixed verify's prompt rows run on the
            chunk pipeline, as above.
        kgroup: Whether a verify at GLM-5.3-Flash's contract runs the
            K-group kernel, which writes the same bits faster. The Mojo op
            ``KdaVerifyRing`` (``Kernels/lib/graph_compiler/kda.mojo``)
            states that contract; any other verify ignores this flag.

    Returns:
        ``[total_tokens, num_value_heads, value_head_dim]`` -- the op's
        ``[1, ...]`` result with its leading axis squeezed, unlike
        :func:`kda_verify_ring_conv`, whose readout keeps the rank its
        ``out_type`` declares.
    """
    _check_dtypes(
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
    )
    _check_shapes(
        "kda_verify_ring",
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        state_layout,
    )
    _check_dtype(DType.float32, ring=ring)
    _check_rank(4, ring=ring)
    total_tokens, num_value_heads, value_head_dim = v.shape
    results = ops.inplace_custom(
        "kda_verify_ring",
        v.device,
        [
            ops.unsqueeze(q, 0),
            ops.unsqueeze(k, 0),
            ops.unsqueeze(v, 0),
            ops.unsqueeze(raw_gate, 0),
            ops.unsqueeze(beta_logits, 0),
            a_log,
            dt_bias,
            cu_seqlens,
            state_pool,
            state_indices,
            ring,
            ring_indices.cast(DType.int32),
            verify_width,
        ],
        [
            TensorType(
                output_dtype,
                [1, total_tokens, num_value_heads, value_head_dim],
                v.device,
            )
        ],
        parameters={
            "gate_mode": gate_mode,
            "beta_mode": beta_mode,
            "state_layout": state_layout,
            "chunk_long_rows": chunk_long_rows,
            "kgroup": kgroup,
        },
    )
    return ops.squeeze(results[0].tensor, 0)


_KGROUP_CONV_OP = "kda_verify_ring_kgroup_conv"
_CHUNK_CONV_OP = "kda_chunk_conv"
_FUSED_HEAD_DIM = 128
"""Key and value head width the fused verify ops are compiled for."""
_FUSED_CONV_TAPS = 4
"""Taps of the short conv the fused verify ops run."""
_CHUNK_CONV_VEC = 2
"""Adjacent channels a lane of the ``kda_chunk_conv`` kernel owns. q, k and
v each span a multiple of it, so no lane's channels straddle two of them."""


def _check_conv_shapes(
    conv_dim: Dim, conv_weight: TensorValue, conv_pool: BufferValue
) -> None:
    """Raises ``ValueError`` unless the conv is the fused ops' 4-tap conv.

    The taps must be ``[conv_dim, 4]`` and the window pool
    ``[rows, conv_dim, 3]``.
    """
    if (
        conv_weight.shape[0] != conv_dim
        or conv_weight.shape[1] != _FUSED_CONV_TAPS
    ):
        raise ValueError(
            f"expected conv_weight to be [{conv_dim}, {_FUSED_CONV_TAPS}],"
            f" was {conv_weight.shape}"
        )
    if (
        conv_pool.shape[1] != conv_dim
        or conv_pool.shape[2] != _FUSED_CONV_TAPS - 1
    ):
        raise ValueError(
            f"expected conv_pool to be [conv_rows, {conv_dim},"
            f" {_FUSED_CONV_TAPS - 1}], was {conv_pool.shape}"
        )


def _check_fused_head_dims(
    op_name: str, key_head_dim: Dim, value_head_dim: Dim
) -> None:
    """Raises ``ValueError`` unless static head widths are both 128.

    The fused verify ops are compiled for 128-wide key and value heads.
    """
    if isinstance(key_head_dim, StaticDim) and isinstance(
        value_head_dim, StaticDim
    ):
        pair = (int(key_head_dim), int(value_head_dim))
        if pair != (_FUSED_HEAD_DIM, _FUSED_HEAD_DIM):
            raise ValueError(
                f"{op_name} is compiled for (key_head_dim, value_head_dim)"
                f" = ({_FUSED_HEAD_DIM}, {_FUSED_HEAD_DIM}), got {pair}"
            )


def _check_conv_operands(
    op_name: str,
    qkv: TensorValue,
    conv_weight: TensorValue,
    conv_pool: BufferValue,
    conv_rows: TensorValue,
) -> None:
    """Raises ``ValueError`` unless the shared conv operands fit the ops.

    The K-group verify and the chunk conv take the same raw input, taps,
    window pool and pool rows. Both widen a bf16 raw input to float32 as
    they load it, which is exact, so a bf16 ``qkv`` gives the float32
    conv's sums.
    """
    if qkv.dtype not in (DType.bfloat16, DType.float32):
        raise ValueError(
            f"{op_name} takes a bf16 or float32 qkv, got {qkv.dtype}"
        )
    _check_dtype(DType.float32, conv_weight=conv_weight)
    _check_dtype(DType.uint32, conv_rows=conv_rows)
    _check_rank(2, qkv=qkv, conv_weight=conv_weight)
    _check_rank(3, conv_pool=conv_pool)
    _check_rank(1, conv_rows=conv_rows)


def _window_rows(rows: Dim, verify_width: TensorValue) -> Dim:
    """Returns the rows of a verify's conv input copy, ``K + 1`` a sequence."""
    return Dim(rows) * (Dim(verify_width.shape[0]) + 1)


def kda_chunk_conv(
    qkv: TensorValue,
    conv_weight: TensorValue,
    conv_pool: BufferValue,
    conv_rows: TensorValue,
    input_row_offsets: TensorValue,
    verify_width: TensorValue,
    *,
    num_key_heads: int,
    num_value_heads: int,
    head_dim: int,
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
    """Runs a draft-free verify's short conv into :func:`kda_chunk`'s q, k, v.

    Computes
    :func:`~max.nn._graph_ops.state_space.gated_delta_conv1d_verify_fwd`
    with ``silu`` at a verify width of zero, which commits every conv
    window, then the bf16 cast of its output and the split of that cast into
    q, k and v, and returns what those return, bit for bit, from one launch.
    The conv takes ``qkv`` widened to float32, which is exact, so a bf16
    ``qkv`` is read as it is. A verify with drafts must leave the conv windows
    to the rollback, and this op commits them, so it raises at run time for
    one: call it behind a draft-free predicate.

    The contract is GLM-5.3-Flash's prefill on NVIDIA, which ships the op
    beside :func:`kda_verify_ring_kgroup_conv`: a 4-tap conv, with ``qkv``
    holding q and k of the key heads and then v of the value heads.

    Args:
        qkv: ``[total_tokens, conv_dim]`` bf16 or float32 raw conv input.
        conv_weight: ``[conv_dim, 4]`` float32 depthwise conv taps.
        conv_pool: ``[conv_rows, conv_dim, 3]`` conv window pool, committed
            in place.
        conv_rows: ``[batch_size]`` uint32 conv pool row per sequence.
        input_row_offsets: ``[batch_size + 1]`` uint32 ragged offsets.
        verify_width: The operand
            :func:`~max.nn._graph_ops.state_space.verify_width_operand` returns.
        num_key_heads: Heads of q and of k.
        num_value_heads: Heads of v.
        head_dim: Channels a head.

    Returns:
        q and k, ``[total_tokens, num_key_heads, head_dim]``, and v,
        ``[total_tokens, num_value_heads, head_dim]``, all bf16, then the
        conv input copy, ``[batch_size * (K + 1), conv_dim]`` float32, which
        a verify of no drafts leaves unwritten.

    Raises:
        ValueError: If the dtypes, ranks or extents fall outside the contract.
    """
    _check_conv_operands(_CHUNK_CONV_OP, qkv, conv_weight, conv_pool, conv_rows)
    _check_dtype(DType.uint32, input_row_offsets=input_row_offsets)
    _check_dtype(DType.int64, verify_width=verify_width)
    _check_rank(
        1, input_row_offsets=input_row_offsets, verify_width=verify_width
    )
    total_tokens, conv_dim = qkv.shape
    qk_channels = num_key_heads * head_dim
    v_channels = num_value_heads * head_dim
    if qk_channels % _CHUNK_CONV_VEC or v_channels % _CHUNK_CONV_VEC:
        raise ValueError(
            f"{_CHUNK_CONV_OP} splits q, k and v at multiples of"
            f" {_CHUNK_CONV_VEC} channels, got {qk_channels} and {v_channels}"
        )
    _check_same_dim(
        "the channel axis",
        qkv=conv_dim,
        heads=Dim(2 * qk_channels + v_channels),
    )
    _check_conv_shapes(conv_dim, conv_weight, conv_pool)
    device = qkv.device
    qk_type = TensorType(
        DType.bfloat16, [total_tokens, num_key_heads, head_dim], device
    )
    results = ops.inplace_custom(
        _CHUNK_CONV_OP,
        device,
        [
            qkv,
            conv_weight,
            conv_pool,
            conv_rows,
            input_row_offsets,
            verify_width,
        ],
        [
            qk_type,
            qk_type,
            TensorType(
                DType.bfloat16,
                [total_tokens, num_value_heads, head_dim],
                device,
            ),
            TensorType(
                DType.float32,
                [_window_rows(conv_rows.shape[0], verify_width), conv_dim],
                device,
            ),
        ],
    )
    q, k, v, window = (result.tensor for result in results)
    return q, k, v, window


def kda_verify_ring_kgroup_conv_servable(
    head_dim: int, gate_mode: KdaGateMode, pool_dtype: DType
) -> bool:
    """Returns whether a verify can take :func:`kda_verify_ring_kgroup_conv`.

    The op runs the K-group verify kernel, so it serves that kernel's
    contract only: NVIDIA, ``K = V = 128``, the ``"safe"`` gate and a
    float32 pool, in a build that ships the closed-source kernels, which
    also ship :func:`kda_chunk_conv` for the draft-free branch beside it.
    Call this while a graph is being built: it reads that graph's kernel
    library.

    Args:
        head_dim: Key and value head width.
        gate_mode: The forget-gate form the layer passes the op.
        pool_dtype: Dtype of the recurrent state pool.

    Returns:
        ``True`` when the op accepts this layer's verify.
    """
    return (
        accelerator_api() == "cuda"
        and head_dim == _FUSED_HEAD_DIM
        and gate_mode == "safe"
        and pool_dtype == DType.float32
        and _KGROUP_CONV_OP in Graph.current._kernel_library
        and _CHUNK_CONV_OP in Graph.current._kernel_library
    )


def kda_verify_ring_kgroup_conv(
    qkv: TensorValue,
    conv_weight: TensorValue,
    conv_pool: BufferValue,
    conv_rows: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    ring: BufferValue,
    ring_indices: TensorValue,
    verify_width: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Runs a verify's short conv and its :func:`kda_verify_ring` as one op.

    Computes
    :func:`~max.nn._graph_ops.state_space.gated_delta_conv1d_verify_fwd`
    with ``silu`` over the raw conv input, then :func:`kda_verify_ring`
    with ``kgroup`` over that conv's q, k and v, and returns what the two
    ops return, bit for bit: the readout, the ring records, the pools and the
    conv input copy. With drafts it is one launch that keeps the conv output
    out of memory. Unlike :func:`kda_verify_ring_conv`, which fuses the conv
    into the ring kernel that ``kgroup`` replaces, it keeps the K-group
    kernel. At a verify width of zero, and for a batch holding a prompt row,
    it runs the two ops' own kernels.

    The contract is :func:`kda_verify_ring_kgroup_conv_servable`'s: float32
    operands and pools but for ``qkv``, ``K = V = 128``, the ``"safe"`` gate
    with logit beta, a ``K_FIRST`` pool and a 4-tap conv, with ``qkv``
    holding q and k of the key heads and then v of the value heads. The conv
    takes ``qkv`` widened to float32, which is exact, so the op widens a
    bf16 ``qkv`` itself.

    Args:
        qkv: ``[total_tokens, conv_dim]`` bf16 or float32 raw conv input.
        conv_weight: ``[conv_dim, 4]`` float32 depthwise conv taps.
        conv_pool: ``[conv_rows, conv_dim, 3]`` conv window pool, committed
            in place only at a verify width of zero.
        conv_rows: ``[batch_size]`` uint32 conv pool row per sequence.
        raw_gate: ``[total_tokens, num_value_heads, key_head_dim]`` float32
            forget-gate pre-activation, before ``dt_bias`` is added.
        beta_logits: ``[total_tokens, num_value_heads]`` float32.
        a_log: ``[num_value_heads]`` float32.
        dt_bias: ``[num_value_heads, key_head_dim]`` float32.
        cu_seqlens: ``[batch_size + 1]`` int32 exclusive prefix offsets.
        state_pool: ``[max_slots, num_value_heads, key_head_dim,
            value_head_dim]`` float32 live pool.
        state_indices: ``[batch_size]`` int32 pool slot per sequence.
        ring: As for :func:`kda_verify_ring`.
        ring_indices: ``[batch_size]`` ring row per sequence.
        verify_width: The operand
            :func:`~max.nn._graph_ops.state_space.verify_width_operand` returns.

    Returns:
        The readout, ``[total_tokens, num_value_heads, value_head_dim]``
        float32, and the conv input copy the rollback places the conv windows
        from, ``[batch_size * (K + 1), conv_dim]`` float32.

    Raises:
        ValueError: If the dtypes, ranks or extents fall outside the contract.
    """
    _check_conv_operands(
        _KGROUP_CONV_OP, qkv, conv_weight, conv_pool, conv_rows
    )
    _check_dtype(
        DType.float32,
        raw_gate=raw_gate,
        beta_logits=beta_logits,
        a_log=a_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        ring=ring,
    )
    _check_dtype(
        DType.int32, cu_seqlens=cu_seqlens, state_indices=state_indices
    )
    _check_rank(2, beta_logits=beta_logits, dt_bias=dt_bias)
    _check_rank(3, raw_gate=raw_gate)
    _check_rank(4, state_pool=state_pool, ring=ring)
    _check_rank(
        1,
        a_log=a_log,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        ring_indices=ring_indices,
    )
    total_tokens, num_value_heads, key_head_dim = raw_gate.shape
    conv_dim = qkv.shape[1]
    _check_same_dim(
        "the token axis",
        qkv=qkv.shape[0],
        raw_gate=total_tokens,
        beta_logits=beta_logits.shape[0],
    )
    _check_same_dim(
        "the value-head axis",
        raw_gate=num_value_heads,
        beta_logits=beta_logits.shape[1],
        a_log=a_log.shape[0],
        dt_bias=dt_bias.shape[0],
        state_pool=state_pool.shape[1],
        ring=ring.shape[1],
    )
    # The pool is K_FIRST, so its third axis is the key-head width.
    _check_same_dim(
        "the key-head width",
        raw_gate=key_head_dim,
        state_pool=state_pool.shape[2],
    )
    _check_conv_shapes(conv_dim, conv_weight, conv_pool)
    _check_fused_head_dims(_KGROUP_CONV_OP, key_head_dim, state_pool.shape[3])
    device = raw_gate.device
    results = ops.inplace_custom(
        _KGROUP_CONV_OP,
        device,
        [
            qkv,
            conv_weight,
            conv_pool,
            conv_rows,
            ops.unsqueeze(raw_gate, 0),
            ops.unsqueeze(beta_logits, 0),
            a_log,
            dt_bias,
            cu_seqlens,
            state_pool,
            state_indices,
            ring,
            ring_indices.cast(DType.int32),
            verify_width,
        ],
        [
            TensorType(
                DType.float32,
                [1, total_tokens, num_value_heads, _FUSED_HEAD_DIM],
                device,
            ),
            TensorType(
                DType.float32,
                [_window_rows(conv_rows.shape[0], verify_width), conv_dim],
                device,
            ),
        ],
    )
    return (
        ops.squeeze(results[0].tensor, 0),
        cast(TensorValue, results[1]),
    )


def _check_verify_ring_conv(
    qkv: TensorValue,
    conv_weight: TensorValue,
    conv_pool: BufferValue,
    conv_row_id: TensorValue,
    raw_gate: TensorValue | None,
    beta_logits: TensorValue | None,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    ring: BufferValue,
    ring_indices: TensorValue,
    norm_weight: TensorValue,
    norm_gate: TensorValue | None,
    *,
    op_name: str,
    state_layout: KdaStateLayout,
    gate_up_weight: TensorValue | None = None,
) -> Dim:
    """Checks one fused verify call's operands; returns the conv's width.

    Every registration of the op takes the same operands but for how the
    qkv, the o_norm gate, the beta logits and the raw gate arrive, so the
    contract is checked once here. Given ``norm_gate``, ``qkv`` is the raw
    packed qkv whose ``conv_dim``-wide thirds the conv reads, the gate is its
    own operand and ``beta_logits`` is at the gate dtype; without it, ``qkv``
    is the whole packed projection row the op views all three out of, so the
    token axis, the value-head axis, the conv width and beta's dtype come
    from that row. Given ``gate_up_weight`` instead of ``raw_gate``, that row
    also carries the gate's down projection at its tail, and the op runs the
    up projection itself.
    """
    if (raw_gate is None) == (gate_up_weight is None):
        raise ValueError(
            f"{op_name} takes exactly one of raw_gate and gate_up_weight"
        )
    if gate_up_weight is not None and norm_gate is not None:
        raise ValueError(
            f"{op_name}: the gate up projection reads the packed row, which"
            " the norm_gate operand rules out"
        )
    # The qkv family shares one dtype parameter at the op, and it is the
    # model's working dtype: bf16 in production, fp32 in the fp32-exact
    # integration tests. The kernel casts every one of them to fp32 on
    # load, so both round exactly as the unfused chain does at their own
    # dtype.
    qkv_family = [("raw_qkv", qkv), ("norm_weight", norm_weight)]
    if raw_gate is not None:
        qkv_family.append(("raw_gate", raw_gate))
    if norm_gate is not None:
        qkv_family.append(("norm_gate", norm_gate))
    for name, tensor in qkv_family:
        if tensor.dtype not in (DType.bfloat16, DType.float32):
            raise ValueError(
                f"expected {name} to be bfloat16 or float32, was {tensor.dtype}"
            )
    # The op binds `norm_weight` and `norm_gate` to `raw_qkv`'s dtype
    # parameter; a mismatch would fail inside the op's registration rather
    # than here.
    same_dtype = [("norm_weight", norm_weight)]
    if gate_up_weight is not None:
        # The kernel holds the up projection's weight to the packed row's
        # dtype.
        same_dtype.append(("gate_up_weight", gate_up_weight))
    if norm_gate is not None:
        same_dtype.append(("norm_gate", norm_gate))
    for name, tensor in same_dtype:
        if tensor.dtype != qkv.dtype:
            raise ValueError(
                f"expected {name} to share raw_qkv's dtype {qkv.dtype},"
                f" was {tensor.dtype}"
            )
    _check_dtype(DType.float32, conv_weight=conv_weight)
    _check_dtype(DType.float32, a_log=a_log, dt_bias=dt_bias)
    if beta_logits is not None:
        _check_dtype(DType.float32, beta_logits=beta_logits)
    _check_dtype(
        DType.uint32,
        conv_row_id=conv_row_id,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        ring_indices=ring_indices,
    )
    _check_dtype(DType.float32, ring=ring)
    _check_rank(2, raw_qkv=qkv, conv_weight=conv_weight)
    _check_rank(3, conv_pool=conv_pool)
    if norm_gate is not None:
        _check_rank(3, norm_gate=norm_gate)
    if raw_gate is not None:
        _check_rank(4, raw_gate=raw_gate)
    if gate_up_weight is not None:
        _check_rank(2, gate_up_weight=gate_up_weight)
    _check_rank(4, state_pool=state_pool, ring=ring)
    _check_rank(
        1,
        conv_row_id=conv_row_id,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        ring_indices=ring_indices,
        norm_weight=norm_weight,
    )
    if beta_logits is not None:
        _check_rank(3, beta_logits=beta_logits)
    _check_rank(2, a_log=a_log, dt_bias=dt_bias)

    if raw_gate is not None:
        total_tokens = raw_gate.shape[1]
        num_value_heads = raw_gate.shape[2]
        key_head_dim = raw_gate.shape[3]
        gate_name = "raw_gate"
    else:
        assert gate_up_weight is not None
        total_tokens = qkv.shape[0]
        num_value_heads = a_log.shape[0]
        key_head_dim = gate_up_weight.shape[1]
        gate_name = "gate_up_weight"
        if gate_up_weight.shape[0] != num_value_heads * key_head_dim:
            raise ValueError(
                "expected gate_up_weight to be [num_value_heads *"
                f" key_head_dim, key_head_dim], was {gate_up_weight.shape}"
            )
    value_head_dim = (
        norm_weight.shape[0] if norm_gate is None else norm_gate.shape[2]
    )
    batch_size = state_indices.shape[0]
    inner = num_value_heads * value_head_dim
    conv_dim = 3 * inner if norm_gate is None else qkv.shape[1]

    token_axis = {
        gate_name: total_tokens,
        "raw_qkv": qkv.shape[0],
    }
    value_head_axis = {
        gate_name: num_value_heads,
        "a_log": a_log.shape[0],
        "dt_bias": dt_bias.shape[0],
        "state_pool": state_pool.shape[1],
        "ring": ring.shape[1],
    }
    if beta_logits is not None:
        token_axis["beta_logits"] = beta_logits.shape[1]
        value_head_axis["beta_logits"] = beta_logits.shape[2]
    if norm_gate is not None:
        token_axis["norm_gate"] = norm_gate.shape[0]
        value_head_axis["norm_gate"] = norm_gate.shape[1]
    _check_same_dim("the packed token axis", **token_axis)
    _check_same_dim("the value-head axis", **value_head_axis)
    _check_same_dim(
        "the key-head width",
        **{gate_name: key_head_dim},
        a_log=a_log.shape[1],
        dt_bias=dt_bias.shape[1],
    )
    if norm_gate is None:
        # A GEMM that pads its output width leaves the tail unread. The
        # gate's down projection, when the op runs the up projection, sits
        # right after beta.
        needed_dim = 4 * inner + num_value_heads
        if gate_up_weight is not None:
            needed_dim = needed_dim + key_head_dim
        if isinstance(qkv.shape[1], StaticDim) and isinstance(
            needed_dim, StaticDim
        ):
            packed_width = int(qkv.shape[1])
            needed = int(needed_dim)
            if packed_width < needed:
                raise ValueError(
                    f"expected {op_name}'s packed row to hold q | k | v, the"
                    " norm gate, beta"
                    + (
                        " and the gate's down projection"
                        if gate_up_weight is not None
                        else ""
                    )
                    + f" in its leading {needed} columns, was {packed_width}"
                )
        if gate_up_weight is not None and isinstance(
            num_value_heads, StaticDim
        ):
            # The kernels load the down projection in 4-element vectors from
            # column `4 * inner + num_value_heads`.
            if int(num_value_heads) % 4 != 0:
                raise ValueError(
                    f"{op_name} needs a value-head count that is a multiple of"
                    f" 4, was {int(num_value_heads)}"
                )
    else:
        _check_same_dim(
            "the value-head width",
            norm_gate=value_head_dim,
            norm_weight=norm_weight.shape[0],
        )
        if conv_dim != 3 * inner:
            raise ValueError(
                "expected raw_qkv to hold q | k | v thirds of "
                f"{inner} channels ({3 * inner} total), was {conv_dim}"
            )
    _check_conv_shapes(conv_dim, conv_weight, conv_pool)
    _check_fused_head_dims(op_name, key_head_dim, value_head_dim)
    if (
        isinstance(cu_seqlens.shape[0], StaticDim)
        and isinstance(batch_size, StaticDim)
        and int(cu_seqlens.shape[0]) != int(batch_size) + 1
    ):
        raise ValueError(
            "expected cu_seqlens to hold one more entry than "
            f"state_indices, was {cu_seqlens.shape[0]} and {batch_size}"
        )
    if (
        isinstance(conv_row_id.shape[0], StaticDim)
        and isinstance(batch_size, StaticDim)
        and int(conv_row_id.shape[0]) != int(batch_size)
    ):
        raise ValueError(
            "expected conv_row_id to hold one entry per sequence, was "
            f"{conv_row_id.shape[0]} and {batch_size}"
        )
    if (
        isinstance(ring_indices.shape[0], StaticDim)
        and isinstance(batch_size, StaticDim)
        and int(ring_indices.shape[0]) != int(batch_size)
    ):
        raise ValueError(
            "expected ring_indices to hold one entry per sequence, was "
            f"{ring_indices.shape[0]} and {batch_size}"
        )
    # The pool's trailing axes are ordered by `state_layout`; getting this
    # wrong transposes the state rather than failing a shape check.
    pool_axes = (state_pool.shape[2], state_pool.shape[3])
    expected = (
        (key_head_dim, value_head_dim)
        if state_layout == "K_FIRST"
        else (value_head_dim, key_head_dim)
    )
    if pool_axes != expected:
        raise ValueError(
            f"expected state_pool to have trailing axes {expected} for "
            f"state_layout={state_layout!r}, was {pool_axes}"
        )
    return conv_dim


def _launch_verify_ring_conv(
    op_name: str,
    qkv: TensorValue,
    conv_weight: TensorValue,
    conv_pool: BufferValue,
    conv_row_id: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue | None,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    ring: BufferValue,
    ring_indices: TensorValue,
    norm_weight: TensorValue,
    norm_gate: TensorValue | None,
    verify_width: TensorValue,
    out_type: TensorType,
    conv_dim: Dim,
    *,
    norm_eps: float,
    gate_mode: KdaGateMode,
    beta_mode: KdaBetaMode,
    state_layout: KdaStateLayout,
) -> tuple[TensorValue, TensorValue]:
    """Enqueues one fused verify launch on already-checked operands.

    ``raw_gate`` is the operand in the raw gate's slot: the raw gate, or for
    ``kda_verify_ring_conv_packed_gate_up`` the gate's up-projection weight.
    """
    device = raw_gate.device
    window_rows = _window_rows(state_indices.shape[0], verify_width)
    values: list[TensorValue | BufferValue] = [
        qkv,
        conv_weight,
        conv_pool,
        conv_row_id,
        raw_gate,
    ]
    # The packed registration views beta out of `qkv`'s own row, so it takes
    # no beta operand.
    if beta_logits is not None:
        values.append(beta_logits)
    values += [
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        ring,
        ring_indices,
        norm_weight,
    ]
    if norm_gate is not None:
        values.append(norm_gate)
    values.append(verify_width)
    results = ops.inplace_custom(
        op_name,
        device,
        values,
        [
            out_type,
            TensorType(DType.float32, [window_rows, conv_dim], device),
        ],
        parameters={
            "gate_mode": gate_mode,
            "beta_mode": beta_mode,
            "state_layout": state_layout,
            "norm_eps": norm_eps,
        },
    )
    return (
        cast(TensorValue, results[0]),
        cast(TensorValue, results[1]),
    )


def kda_verify_ring_conv(
    raw_qkv: TensorValue,
    conv_weight: TensorValue,
    conv_pool: BufferValue,
    conv_row_id: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    ring: BufferValue,
    ring_indices: TensorValue,
    verify_width: TensorValue,
    norm_weight: TensorValue,
    norm_gate: TensorValue,
    out_type: TensorType,
    *,
    norm_eps: float,
    gate_mode: KdaGateMode = "safe",
    beta_mode: KdaBetaMode = "logits",
    state_layout: KdaStateLayout = "K_FIRST",
) -> tuple[TensorValue, TensorValue]:
    """Runs the KDA verify chain fused into one launch, into a ring.

    The short conv over the raw packed qkv, the glue's silu + bf16 cast, the
    gate pre-activation and the gated o_norm all run inside
    ``kda_verify_ring``'s recurrence kernel, so one launch produces the
    gated-normed readout, the conv window copy the rollback replay reads,
    and the per-token ring records :func:`kda_state_fold` applies once
    acceptance is known. The records and the fold contract are
    :func:`kda_verify_ring`'s unchanged.

    The gate contract differs from :func:`kda_verify_ring`'s and the two
    must not be conflated: this op takes the RAW gate with PER-CHANNEL
    ``a_log [HV, K]`` and ``dt_bias [HV, K]`` and applies the safe mode's
    pre-activation ``(g + dt_bias) * exp(a_log)`` in-kernel, while that op
    takes the pre-activated gate with PER-HEAD ``a_log [HV]``. It is the
    safe gate's op alone; ``gate_mode`` must be ``"safe"``. The raw
    gate stays at the model's working dtype -- the layer passes its bf16
    projection output straight through -- because the kernel casts on
    load and a bf16 load plus an fp32 cast is exact, so the in-kernel
    pre-activation sees the value the graph's fp32 cast of the same
    tensor produced, without the cast launch the fusion would otherwise
    re-add.

    At a verify width of zero (the context step, which runs the same graph)
    the op commits the conv window and the state pool the way the plain
    forward does, writes no ring record and leaves the window copy
    unwritten. With drafts, a request longer than the ring's ``RING_LEN``
    records nothing and commits ``state_pool`` itself, the same contract as
    :func:`kda_verify_ring`.

    ``raw_qkv`` holds the q, k and v projections as contiguous thirds of
    ``num_value_heads * value_head_dim`` channels -- the merged qkv
    projection's output layout -- so the op requires as many key heads as
    value heads; the conv-dim check rejects anything else. The
    qkv family (``raw_qkv``, ``raw_gate``, ``norm_weight``, ``norm_gate``)
    takes the model's working dtype, bf16 or fp32.

    The four row tables take the uint32 the state leaves and the batch
    offsets carry, so a verify layer hands them over without a cast launch
    apiece; the op reads them as int32 where it indexes, which holds every
    row a pool can address.

    Args:
        raw_qkv: ``[total_tokens, 3 * num_value_heads * value_head_dim]``
            raw bf16 (or fp32) qkv, q | k | v contiguous thirds.
        conv_weight: ``[conv_dim, 4]`` fp32 depthwise conv weights.
        conv_pool: ``[conv_rows, conv_dim, 3]`` conv window pool, mutated in
            place at ``conv_row_id`` only at a verify width of zero.
        conv_row_id: ``[batch_size]`` uint32 conv-pool row of each sequence.
        raw_gate: ``[1, total_tokens, num_value_heads, key_head_dim]``
            (the op takes rank 4) RAW
            forget-gate pre-activation at the working dtype, before the
            layer's ``(g + dt_bias) * exp(a_log)`` elementwise pass.
        beta_logits: ``[1, total_tokens, num_value_heads]``.
        a_log: ``[num_value_heads, key_head_dim]`` PER-CHANNEL log of the
            gate's ``A``.
        dt_bias: ``[num_value_heads, key_head_dim]``.
        cu_seqlens: ``[batch_size + 1]`` uint32 exclusive prefix offsets.
        state_pool: ``[max_slots, num_value_heads, key_head_dim,
            value_head_dim]`` live pool, laid out per ``state_layout``.
        state_indices: ``[batch_size]`` uint32 pool slot per sequence.
        ring: ``[ring_rows, num_value_heads, RING_LEN, record_stride]``
            float32 ring pool.
        ring_indices: ``[batch_size]`` uint32 ring row per sequence.
        verify_width: The operand
            :func:`~max.nn._graph_ops.state_space.verify_width_operand`
            returns.
        norm_weight: ``[value_head_dim]`` o_norm weight, the working dtype.
        norm_gate: ``[total_tokens, num_value_heads, value_head_dim]``
            o_norm gate, the working dtype.
        out_type: The op result's declared type, ``[1, total_tokens,
            num_value_heads, value_head_dim]`` on one device; the readout
            comes back at exactly this rank and shape.
        norm_eps: Epsilon of the gated o_norm.
        gate_mode: Forget-gate form; the op is the bounded ``"safe"`` gate's
            alone, whose pre-activation ``(g + dt_bias) * exp(a_log)`` is
            applied in-kernel. Any other mode is a build error: the two
            verify arms implement ``"original"`` differently, so the fused
            chain does not offer it.
        beta_mode: Whether the kernel applies the sigmoid to ``beta_logits``.
        state_layout: Pool axis order; must match how the pool was allocated.

    Returns:
        The gated-normed readout at ``out_type``'s own rank,
        ``[1, total_tokens, num_value_heads, value_head_dim]``, and the conv
        window copy, ``[batch_size * (K + 1), conv_dim]`` fp32. Unlike
        :func:`kda_verify_ring`, the readout keeps its leading axis: the
        layer rebinds a 4-entry TP mapping onto it, as its unfused op
        call's own result, and a squeezed 3-D result would take the
        sharding on the wrong axis.

    Raises:
        ValueError: If the dtype groupings bind no kernel, or the ranks and
            extents disagree. The kernel's own shape guards are
            ``debug_assert``, so they are absent from a production build.
    """
    conv_dim = _check_verify_ring_conv(
        raw_qkv,
        conv_weight,
        conv_pool,
        conv_row_id,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        ring,
        ring_indices,
        norm_weight,
        norm_gate,
        op_name="kda_verify_ring_conv",
        state_layout=state_layout,
    )
    return _launch_verify_ring_conv(
        "kda_verify_ring_conv",
        raw_qkv,
        conv_weight,
        conv_pool,
        conv_row_id,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        ring,
        ring_indices,
        norm_weight,
        norm_gate,
        verify_width,
        out_type,
        conv_dim,
        norm_eps=norm_eps,
        gate_mode=gate_mode,
        beta_mode=beta_mode,
        state_layout=state_layout,
    )


def kda_verify_ring_conv_packed(
    packed_qkvg: TensorValue,
    conv_weight: TensorValue,
    conv_pool: BufferValue,
    conv_row_id: TensorValue,
    raw_gate: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    ring: BufferValue,
    ring_indices: TensorValue,
    verify_width: TensorValue,
    norm_weight: TensorValue,
    out_type: TensorType,
    *,
    norm_eps: float,
    gate_mode: KdaGateMode = "safe",
    beta_mode: KdaBetaMode = "logits",
    state_layout: KdaStateLayout = "K_FIRST",
) -> tuple[TensorValue, TensorValue]:
    """Runs :func:`kda_verify_ring_conv` on the packed input projection row.

    Same op, same launch, same contract -- only the qkv, o_norm-gate and beta
    operands come from one packed tensor: ``packed_qkvg``'s leading
    ``3 * num_value_heads * value_head_dim`` columns are the qkv the conv
    reads, the next ``num_value_heads * value_head_dim`` are the gate, and
    the ``num_value_heads`` after those are the beta logits, which the op
    widens to fp32 exactly. Handing the projection's whole output over skips
    the elementwise copies a ``split`` and its casts would launch per layer
    per step. Every other argument is :func:`kda_verify_ring_conv`'s, minus
    the ``norm_gate`` and ``beta_logits`` operands this op views out of
    ``packed_qkvg``; ``packed_qkvg`` takes its ``raw_qkv``'s place.

    Returns:
        :func:`kda_verify_ring_conv`'s two results.

    Raises:
        ValueError: If the dtype groupings bind no kernel, or the ranks and
            extents disagree.
    """
    conv_dim = _check_verify_ring_conv(
        packed_qkvg,
        conv_weight,
        conv_pool,
        conv_row_id,
        raw_gate,
        None,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        ring,
        ring_indices,
        norm_weight,
        None,
        op_name="kda_verify_ring_conv_packed",
        state_layout=state_layout,
    )
    return _launch_verify_ring_conv(
        "kda_verify_ring_conv_packed",
        packed_qkvg,
        conv_weight,
        conv_pool,
        conv_row_id,
        raw_gate,
        None,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        ring,
        ring_indices,
        norm_weight,
        None,
        verify_width,
        out_type,
        conv_dim,
        norm_eps=norm_eps,
        gate_mode=gate_mode,
        beta_mode=beta_mode,
        state_layout=state_layout,
    )


def kda_verify_ring_conv_packed_gate_up(
    packed_qkvg: TensorValue,
    conv_weight: TensorValue,
    conv_pool: BufferValue,
    conv_row_id: TensorValue,
    gate_up_weight: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    ring: BufferValue,
    ring_indices: TensorValue,
    verify_width: TensorValue,
    norm_weight: TensorValue,
    out_type: TensorType,
    *,
    norm_eps: float,
    gate_mode: KdaGateMode = "safe",
    beta_mode: KdaBetaMode = "logits",
    state_layout: KdaStateLayout = "K_FIRST",
) -> tuple[TensorValue, TensorValue]:
    """Runs :func:`kda_verify_ring_conv_packed` with the gate's up projection.

    The layer's raw gate is two projections, and the fused input GEMM already
    emits the first: ``packed_qkvg`` carries the gate's ``key_head_dim``-wide
    down projection right after beta's columns. This op views it there and
    runs the up projection ``gate_up_weight`` (``[num_value_heads *
    key_head_dim, key_head_dim]``, the up projection's own weight) inside the
    recurrence kernel, so the graph launches neither the up projection's GEMM
    nor the copy that made the down projection contiguous for it. The up
    projection sums in fp32 and rounds once to the model dtype, as the GEMM
    did; its summation order is the kernel's, so a raw gate may differ from
    the GEMM's in that dtype's last bit. Both verify-width arms run the same
    arithmetic.

    Every other argument is :func:`kda_verify_ring_conv_packed`'s, minus
    its ``raw_gate``.

    Returns:
        :func:`kda_verify_ring_conv`'s two results.

    Raises:
        ValueError: If the dtype groupings bind no kernel, or the ranks and
            extents disagree.
    """
    conv_dim = _check_verify_ring_conv(
        packed_qkvg,
        conv_weight,
        conv_pool,
        conv_row_id,
        None,
        None,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        ring,
        ring_indices,
        norm_weight,
        None,
        op_name="kda_verify_ring_conv_packed_gate_up",
        state_layout=state_layout,
        gate_up_weight=gate_up_weight,
    )
    return _launch_verify_ring_conv(
        "kda_verify_ring_conv_packed_gate_up",
        packed_qkvg,
        conv_weight,
        conv_pool,
        conv_row_id,
        gate_up_weight,
        None,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        ring,
        ring_indices,
        norm_weight,
        None,
        verify_width,
        out_type,
        conv_dim,
        norm_eps=norm_eps,
        gate_mode=gate_mode,
        beta_mode=beta_mode,
        state_layout=state_layout,
    )


def kda_state_fold(
    state_pool: BufferValue,
    ring: BufferValue,
    state_row_ids: TensorValue,
    ring_row_ids: TensorValue,
    num_accepted: TensorValue,
    row_offsets: TensorValue,
    verify_width: TensorValue,
    conv_pool: BufferValue | None = None,
    conv_row_ids: TensorValue | None = None,
    qkv_windows: Sequence[TensorValue] | None = None,
    window_row_len: TensorValue | None = None,
    *,
    state_layout: KdaStateLayout = "K_FIRST",
) -> None:
    """Applies a verify's accepted ring records to the state pool.

    Advances ``state_pool`` over the first ``num_accepted[b]`` records
    :func:`kda_verify_ring` wrote, for every layer in one launch. A row with
    ``num_accepted[b] == 0`` is left unchanged, and a row longer than
    ``RING_LEN`` was committed by the verify itself and is skipped. Nothing
    runs at a verify width of zero.

    Given ``conv_pool``, ``conv_row_ids`` and ``qkv_windows``, the same launch
    also places every request's conv window at its accepted length from
    ``qkv_windows``, the conv inputs the verify captured. The windows are
    stacked into one tensor here, one graph op for every layer; at a verify
    width of zero that stack still runs, copying windows the verify left
    unwritten, and nothing reads the result. Given none of the three, the
    launch folds the ring only and leaves the conv pool to the caller, which
    is what a rollback that replays the conv itself does; that launch is a
    different op, ``kda_state_fold_no_conv``.

    ``num_accepted[b]`` must not exceed the row's length in ``row_offsets``:
    past it the ring holds no record of this verify. The kernel clamps a
    larger count to the length, so a zero-length padding row never folds.

    Args:
        state_pool: ``[max_slots, num_value_heads, key_head_dim,
            value_head_dim]`` live pool, folded in place.
        ring: ``[ring_rows, num_value_heads, RING_LEN, record_stride]`` ring
            pool.
        state_row_ids: ``[num_layers, batch_size]`` live pool rows.
        ring_row_ids: ``[num_layers, batch_size]`` ring rows.
        num_accepted: ``[batch_size]`` records to fold.
        row_offsets: ``[batch_size + 1]`` offsets of the verify's rows, the
            ``cu_seqlens`` :func:`kda_verify_ring` ran over.
        verify_width: The operand
            :func:`~max.nn._graph_ops.state_space.verify_width_operand` returns.
        conv_pool: ``[conv_rows, conv_dim, kernel_size - 1]`` conv window
            pool, placed in place. Optional, with the two below: without all
            three the conv pool is left alone.
        conv_row_ids: ``[num_layers, batch_size]`` conv pool rows. Optional;
            give it together with ``conv_pool`` and ``qkv_windows``, or none of
            the three.
        qkv_windows: One ``[window_rows, conv_dim]`` float32 conv input per
            layer, the copies the verify's conv kernel kept; ``window_rows``
            must span ``row_offsets``' rows. Optional, with the two above.
        window_row_len: ``[1]`` int32 scalar: the capture's rows per request.
            0 reads each request's window at its own offset in
            ``row_offsets``; a verify's capture passes its width (``K + 1``),
            the stride the conv kernel wrote each row's tokens at. An operand
            rather than an argument because a served graph declares its draft
            width symbolically, so the width is only known at run time;
            omitted, the placement reads the caller's own layout.
        state_layout: Pool axis order; must match how the pool was allocated.

    Raises:
        ValueError: If only some of ``conv_pool``, ``conv_row_ids`` and
            ``qkv_windows`` are given, if ``qkv_windows`` is empty, or if its
            elements disagree on shape or device.
    """
    _check_dtype(DType.float32, ring=ring)
    _check_rank(4, ring=ring)
    if conv_pool is None and conv_row_ids is None and qkv_windows is None:
        ops.inplace_custom(
            "kda_state_fold_no_conv",
            state_pool.device,
            [
                state_pool,
                ring,
                state_row_ids.cast(DType.uint32),
                ring_row_ids.cast(DType.uint32),
                num_accepted.cast(DType.uint32),
                row_offsets.cast(DType.uint32),
                verify_width,
            ],
            parameters={"state_layout": state_layout},
        )
        return
    if conv_pool is None or conv_row_ids is None or qkv_windows is None:
        raise ValueError(
            "kda_state_fold's conv window placement needs conv_pool,"
            " conv_row_ids and qkv_windows together, or none of them"
        )
    _check_rank(3, conv_pool=conv_pool)
    if not qkv_windows:
        raise ValueError("kda_state_fold needs one conv window per layer")
    windows = list(qkv_windows)
    stride = window_row_len
    if stride is not None:
        # A shape-derived scalar arrives rank 0; the op takes one element.
        if len(stride.shape) == 0:
            stride = ops.reshape(stride, [1])
        if len(stride.shape) != 1 or stride.shape[0] != 1:
            raise ValueError(
                "kda_state_fold's window_row_len takes one int32 element, got"
                f" shape {list(stride.shape)}"
            )
        _check_dtype(DType.int32, window_row_len=stride)
        if stride.device != state_pool.device:
            raise ValueError(
                "kda_state_fold's window_row_len must live on the state"
                f" pool's device ({state_pool.device}), got {stride.device}"
            )
    first = windows[0]
    _check_dtype(DType.float32, qkv_window=first)
    _check_rank(2, qkv_window=first)
    if any(w.shape != first.shape or w.device != first.device for w in windows):
        raise ValueError(
            "kda_state_fold's conv windows must share one shape and device"
        )
    ops.inplace_custom(
        "kda_state_fold",
        state_pool.device,
        [
            state_pool,
            ring,
            state_row_ids.cast(DType.uint32),
            ring_row_ids.cast(DType.uint32),
            num_accepted.cast(DType.uint32),
            row_offsets.cast(DType.uint32),
            conv_pool,
            conv_row_ids.cast(DType.uint32),
            ops.stack(windows, axis=0),
            (
                stride
                if stride is not None
                else ops.reshape(
                    ops.constant(0, DType.int32, device=state_pool.device), [1]
                )
            ),
            verify_width,
        ],
        parameters={"state_layout": state_layout},
    )
