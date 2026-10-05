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
"""Python wrapper for the Mamba-2 SSD chunked-scan kernel.

:func:`mamba2_ssd_chunk_scan_varlen_fwd_inplace`
  The Mamba-2 SSD chunked scan, used for both prefill and decode. Decode is
  just a batch of length-1 sequences continuing their pooled state. Per-head
  scalar ``A``, grouped ``B``/``C``, per-head ``dt`` + ``dt_bias`` softplus.
  State resets at each ``query_start_loc`` boundary, and final states are
  written back into a slot-indexed SSM state pool in place.
"""

from __future__ import annotations

from typing import cast

from max.graph import BufferValue, TensorType, TensorValue, ops


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
    """Performs the Mamba-2 SSD chunked-scan forward, writing final states
    back into the SSM pool in place.

    Writes final states directly into ``ssm_pool[cache_indices[b], ...]``
    in place, so the graph never round-trips the whole pool.

    Args:
        x: The ``[total_len, nheads, head_dim]`` SSM input (model
            dtype).
        dt: The ``[total_len, nheads]`` per-head time deltas (model
            dtype).
        A: The ``[nheads]`` per-head scalar (model dtype; already
            ``-exp(A_log)``).
        B: The ``[total_len, ngroups, dstate]`` grouped input proj (model
            dtype).
        C: The ``[total_len, ngroups, dstate]`` grouped output proj
            (model dtype).
        D: The ``[nheads]`` skip connection (model dtype; empty to
            disable).
        dt_bias: The ``[nheads]`` dt bias (model dtype; empty to disable
            softplus).
        ssm_pool: The ``[max_slots, nheads, head_dim, dstate]`` mutable
            state pool (fp32; bf16 on Apple GPUs — storage only, the
            scan accumulates in fp32). Read at
            ``ssm_pool[cache_indices[b]]`` when ``has_initial_state[b]``
            is true; written in-place with final state.
        query_start_loc: The ``[batch + 1]`` int32 cumulative sequence
            lengths.
        has_initial_state: The ``[batch]`` bool, whether to load initial
            state for each sequence (empty to disable).
        cache_indices: The ``[batch]`` uint32 slot indices into
            ``ssm_pool``, or one ``[1, batch]`` row of a per-layer table.

    Returns:
        ``y``, the ``[total_len, nheads, head_dim]`` output (model
        dtype). ``ssm_pool`` is mutated in place.
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
