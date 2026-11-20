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
"""The host MLA extents the K3 pairing's two prefill plans read.

Both the target's verify and the drafter's block run the one-shot
decompression prefill arm on some builds, and the plan needs the step's
total row extent on the host: the kernel that fills the buffer reads the
scalar host-side, to size the allocation and launch the copy. Reading the
plan's own extent back from the device answers that, but the read is a
D2H sync, so a graph holding it cannot be captured as a device graph.

Instead :func:`k3_spec_mla_extent` selects a host value in-graph:

* the batch's page-aligned context length plus an allowance per request,
  whenever the batch schedules more than ``K + 1`` rows for some request,
  which only a prompt or mixed step does. The context length counts each
  request's processed and scheduled tokens, so the allowance covers the
  rest: the drafts a decode row riding the step carries, including the
  previous step's accepted drafts that MAX commits lazily, and the
  drafter's block, whose commit count is decided on the device. Those steps
  are never captured, so the frozen value never outlives its batch;
* otherwise -- every decode step, the only kind capture records -- the
  ``min(batch_size * align_up(max_cache_length, page), mla_buffer_size)``
  bound, which reads
  only host inputs that follow the capture key: ``max_cache_length`` is
  the aligned bucket length at capture and at replay alike, so the frozen
  bound covers every real batch the graph replays. That coverage rests on
  the capture order -- warmup probes the largest length first within each
  key and skips a key already captured, and the key includes the batch
  size, so each graph freezes its bound at the largest cache length that
  maps to it and every replay's smaller bucket stays covered; a capture
  order that recorded a smaller length first would strand it.

The context length changes every step, so the capture host-input guard
reports it on every replay as changed-since-capture even though a captured
(decode) step never reads it -- known-benign, and capture tests should
treat it as such.

An oversized extent is safe -- the cache-to-buffer copy zero-writes
past the plan's own rows -- while an undersized one silently corrupts,
so both arms are upper bounds by construction, and the select errs toward
the bound whenever it cannot prove a step uncaptured.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.support.math import ceildiv

__all__ = ["k3_spec_draft_allowance", "k3_spec_mla_extent"]


def k3_spec_draft_allowance(
    *, num_draft_tokens: int, reach: int, page_size: int
) -> int:
    """Returns the rows the exact arm adds per request past the context length.

    A request reads past its page-aligned context length by at most the
    drafts it carries -- lazily committed and under verification -- plus
    ``reach``, so a page-aligned allowance of that many rows covers it.
    """
    return ceildiv(2 * num_draft_tokens + reach, page_size) * page_size


def k3_spec_mla_extent(
    *,
    context_length: TensorValue,
    reach: int,
    kv: KVCacheInputsPerDevice[TensorValue, BufferValue],
    input_row_offsets: TensorValue,
    num_draft_tokens: int,
    page_size: int,
    mla_buffer_size: int,
) -> TensorValue:
    """Selects this call's MLA prefill-plan extent on the host, in-graph.

    Returns the host ``[1]`` int32 to pass as the plan's ``buffer_length``.
    ``context_length`` is the batch's page-aligned context length, its
    ``batch_context_lengths`` input, and ``reach`` the rows this call reads
    past each request's window: none for the verify, the drafter's block
    width for the drafter. ``kv`` is
    either cache leaf: its ``max_prompt_length`` names the widest merged
    window any request schedules, and its ``max_cache_length`` the bound
    arm keys on; the target's and the drafter's leaves carry the same
    host values. ``input_row_offsets`` is the batch's per-request row
    offsets, whose length names the batch size the bound charges.
    ``mla_buffer_size`` is the one-shot plan's row cap.

    ``kv`` and ``context_length`` are one replica's, so both arms size from
    that replica's lengths alone; Kimi K3 refuses data parallelism
    (``KimiK3Config._attn_kv_params``), which keeps every replica at the
    same length -- a future DP enablement must re-derive this select per
    replica or it undersizes every replica past the first.
    """
    cpu = DeviceRef.CPU()
    # A decode step schedules one anchor row plus at most K drafts per
    # request, so a max_prompt_length past K+1 can only be a prompt or
    # mixed step -- never captured, so its exact value may be read live.
    decode = kv.max_prompt_length <= ops.constant(
        num_draft_tokens + 1, kv.max_prompt_length.dtype, device=cpu
    )
    cache = ops.cast(kv.max_cache_length, DType.int32)
    page = ops.constant(page_size, DType.int32, device=cpu)
    rows = ops.cast(
        ops.shape_to_tensor(input_row_offsets.shape)[0] - 1, DType.int32
    )
    bound = ops.min(
        (cache + (page - 1)) // page * page * rows,
        ops.constant(mla_buffer_size, DType.int32, device=cpu),
    )
    allowance = k3_spec_draft_allowance(
        num_draft_tokens=num_draft_tokens, reach=reach, page_size=page_size
    )
    exact = ops.cast(context_length, DType.int32) + rows * ops.constant(
        allowance, DType.int32, device=cpu
    )
    # The bound covers every step's plan too, so it caps the exact arm.
    return ops.where(decode, bound, ops.min(exact, bound))
