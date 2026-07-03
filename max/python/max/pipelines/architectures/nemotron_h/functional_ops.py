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
"""Python wrappers for the Nemotron-H Mamba-2 state-space kernels.

Two graph-level wrappers over Mojo ops registered in the ``state_space``
package. NOTE: unlike the Qwen3.5 gated-delta ops, ``mamba2_ssd_chunk_scan_varlen_fwd``
and ``causal_conv1d_varlen_fwd`` are NOT yet registered in
``graph_compiler/builtin_kernels/kernels.mojo`` (only ``gated_delta_conv1d_fwd`` /
``gated_delta_recurrence_fwd`` are). They are reachable only via the
``state_space`` kernel package, so the model's ``session.load`` must pass
``custom_extensions=_get_state_space_paths()`` (see ``model.py`` /
the Mamba arch's ``functional_ops._get_state_space_paths``).

  :func:`mamba2_ssd_chunk_scan_varlen_fwd`
    The Mamba-2 SSD chunked-scan, used for BOTH prefill and decode. Decode is
    just a batch of length-1 sequences with ``initial_states`` carried from the
    previous step. Per-head scalar ``A``, grouped ``B``/``C``, per-head ``dt`` +
    ``dt_bias`` softplus. State resets at each ``query_start_loc`` boundary.
    The existing ``varlen_selective_state_update`` decode op only supports
    dstate in {4,8,16}; Nemotron-H uses dstate=128, hence the SSD kernel is
    used for decode too.

  :func:`causal_conv1d_varlen_fwd`
    Slot-indexed depthwise causal conv1d over a ragged batch. Reads/writes a
    per-layer conv-state pool in place at slot ``cache_indices[batch_item]``;
    handles prefill and decode in one op.
"""

from __future__ import annotations

from typing import cast

from max.dtype import DType
from max.graph import BufferValue, TensorType, TensorValue, ops


def mamba2_ssd_chunk_scan_varlen_fwd(
    x: TensorValue,
    dt: TensorValue,
    A: TensorValue,
    B: TensorValue,
    C: TensorValue,
    D: TensorValue,
    dt_bias: TensorValue,
    initial_states: TensorValue,
    query_start_loc: TensorValue,
    has_initial_state: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Mamba-2 SSD chunked-scan forward (prefill and decode).

    Args:
        x: ``[total_len, nheads, head_dim]`` SSM input (model dtype).
        dt: ``[total_len, nheads]`` per-head time deltas (model dtype).
        A: ``[nheads]`` per-head scalar (model dtype; already ``-exp(A_log)``).
        B: ``[total_len, ngroups, dstate]`` grouped input proj (model dtype).
        C: ``[total_len, ngroups, dstate]`` grouped output proj (model dtype).
        D: ``[nheads]`` skip connection (model dtype; empty to disable).
        dt_bias: ``[nheads]`` dt bias (model dtype; empty to disable softplus
            bias).
        initial_states: ``[batch, nheads, head_dim, dstate]`` fp32 initial SSM
            state (empty ``[0, ...]`` for a fresh prefill).
        query_start_loc: ``[batch + 1]`` int32 cumulative sequence lengths.
        has_initial_state: ``[batch]`` bool, whether to load ``initial_states``
            for each sequence (empty to disable).

    Returns:
        ``(y, final_states)`` where ``y`` is ``[total_len, nheads, head_dim]``
        (model dtype) and ``final_states`` is
        ``[batch, nheads, head_dim, dstate]`` fp32.
    """
    device = x.device
    total_len = x.shape[0]
    nheads = x.shape[1]
    head_dim = x.shape[2]
    dstate = B.shape[2]

    y_type = TensorType(x.dtype, [total_len, nheads, head_dim], device)
    final_states_type = TensorType(
        DType.float32,
        [query_start_loc.shape[0] - 1, nheads, head_dim, dstate],
        device,
    )

    results = ops.custom(
        "mamba2_ssd_chunk_scan_varlen_fwd",
        device,
        [
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            initial_states,
            query_start_loc,
            has_initial_state,
        ],
        [y_type, final_states_type],
        # `dt_softplus` is a struct-level parameter of the registered op
        # (`Mamba2SSDChunkScanVarlenFwd[dt_softplus: Bool]`); it must be passed
        # explicitly — it is not inferred from the tensor args.
        parameters={"dt_softplus": True},
    )
    return cast(TensorValue, results[0]), cast(TensorValue, results[1])


def mamba2_ssd_chunk_scan_varlen_fwd_inplace(
    x: TensorValue,
    dt: TensorValue,
    A: TensorValue,
    B: TensorValue,
    C: TensorValue,
    D: TensorValue,
    dt_bias: TensorValue,
    ssm_pool: BufferValue,
    query_start_loc: TensorValue,
    has_initial_state: TensorValue,
    cache_indices: TensorValue,
) -> TensorValue:
    """Mamba-2 SSD chunked-scan forward — in-place SSM-pool write-back.

    Identical to :func:`mamba2_ssd_chunk_scan_varlen_fwd` but writes final
    states directly into ``ssm_pool[cache_indices[b], ...]`` (fp32, in-place)
    instead of returning a separate ``final_states`` output tensor.  This
    eliminates the graph-side ``buffer_load → gather → scatter_nd →
    buffer_store`` whole-pool RMW that otherwise dominates decode GPU time.

    Args:
        x: ``[total_len, nheads, head_dim]`` SSM input (model dtype).
        dt: ``[total_len, nheads]`` per-head time deltas (model dtype).
        A: ``[nheads]`` per-head scalar (model dtype; already ``-exp(A_log)``).
        B: ``[total_len, ngroups, dstate]`` grouped input proj (model dtype).
        C: ``[total_len, ngroups, dstate]`` grouped output proj (model dtype).
        D: ``[nheads]`` skip connection (model dtype; empty to disable).
        dt_bias: ``[nheads]`` dt bias (model dtype; empty to disable softplus).
        ssm_pool: ``[max_slots, nheads, head_dim, dstate]`` fp32 mutable state
            pool.  Read at ``ssm_pool[cache_indices[b]]`` when
            ``has_initial_state[b]`` is true; written in-place with final state.
        query_start_loc: ``[batch + 1]`` int32 cumulative sequence lengths.
        has_initial_state: ``[batch]`` bool, whether to load initial state for
            each sequence (empty to disable).
        cache_indices: ``[batch]`` uint32 slot indices into ``ssm_pool``.

    Returns:
        ``y``: ``[total_len, nheads, head_dim]`` (model dtype).
        ``ssm_pool`` is mutated in place.
    """
    device = x.device
    total_len = x.shape[0]
    nheads = x.shape[1]
    head_dim = x.shape[2]

    y_type = TensorType(x.dtype, [total_len, nheads, head_dim], device)

    results = ops.inplace_custom(
        "mamba2_ssd_chunk_scan_varlen_fwd_inplace",
        device,
        [
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            ssm_pool,
            query_start_loc,
            has_initial_state,
            cache_indices,
        ],
        [y_type],
        parameters={"dt_softplus": True},
    )
    return cast(TensorValue, results[0])


def causal_conv1d_varlen_fwd(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue,
    conv_states: BufferValue,
    query_start_loc: TensorValue,
    cache_indices: TensorValue,
    has_initial_state: TensorValue,
    activation: str = "silu",
) -> TensorValue:
    """Slot-indexed varlen causal depthwise conv1d (prefill and decode).

    Mutates the conv-state pool ``conv_states`` in place at slot
    ``cache_indices[batch_item]`` — the Qwen3.5 GatedDeltaNet conv pattern. The
    builtin registers ``conv_states`` as a ``MutableInputTensor`` at operand
    position 4 (after ``output, x, weight, bias``).

    Args:
        x: ``[dim, total_seqlen]`` input (channels-first, model dtype).
        weight: ``[dim, width]`` depthwise conv weights.
        bias: ``[dim]`` per-channel bias (empty to disable).
        conv_states: ``[max_slots, dim, width - 1]`` mutable conv-state pool.
        query_start_loc: ``[batch + 1]`` int32 cumulative sequence lengths.
        cache_indices: ``[batch]`` int32 slot indices into ``conv_states``.
        has_initial_state: ``[batch]`` bool, whether to use the stored state.
        activation: ``"silu"`` or ``"none"``.

    Returns:
        ``[dim, total_seqlen]`` conv output. ``conv_states`` is mutated in place.
    """
    device = x.device
    dim = x.shape[0]
    total_seqlen = x.shape[1]

    out_type = TensorType(x.dtype, [dim, total_seqlen], device)

    # Operand order matches the builtin registration
    # ``CausalConv1DVarlenFwd.execute``: after the ``output`` operand,
    # ``x, weight, bias, conv_states (MutableInput), query_start_loc,
    # cache_indices, has_initial_state``.
    results = ops.inplace_custom(
        "causal_conv1d_varlen_fwd",
        device,
        [
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
        ],
        [out_type],
        parameters={"activation": activation},
    )
    return cast(TensorValue, results[0])
