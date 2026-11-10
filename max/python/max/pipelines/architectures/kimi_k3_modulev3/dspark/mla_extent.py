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

Instead the batch carries the extent as a host graph input, and this
module owns both halves: it computes the pair on the host --
:func:`spec_verify_extent`, :func:`spec_block_extent_bound` -- and
selects a capture-safe value for it in-graph, on the host:

* the batched extent -- the exact value for a prompt/mixed step, an upper
  bound for the drafter's block, whose commit count is decided on the
  device -- whenever the batch schedules more than ``K + 1`` rows for some
  request, which only a prompt or mixed step does; those steps are never
  captured, so the frozen value never outlives its batch;
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

The pair itself is a trailing host input whose value changes every step,
so the capture host-input guard reports it on every replay as
changed-since-capture even though a captured (decode) step never reads
it -- known-benign, and capture tests should treat it as such.

An oversized extent is safe -- the cache-to-buffer copy zero-writes
past the plan's own rows -- while an undersized one silently corrupts,
so both arms are upper bounds by construction, and the select errs toward
the bound whenever it cannot prove a step uncaptured.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.paged_kv_cache._common import (
    cache_valid_length_for_context,
)

__all__ = [
    "MLA_SPEC_EXTENTS",
    "k3_spec_mla_extent",
    "spec_block_extent_bound",
    "spec_verify_extent",
]

MLA_SPEC_EXTENTS = "mla_spec_extents"
"""Key in :attr:`~max.pipelines.speculative.block_driver.BlockBatch.extra`
for the host ``int32[2]`` pair a batch's extents ride: the verify plan's
exact extent, then the drafter block's upper bound."""


def _page_aligned(length: int, page_size: int) -> int:
    """Rounds ``length`` up to a multiple of ``page_size``."""
    return (length + page_size - 1) // page_size * page_size


def spec_verify_extent(contexts: Sequence[TextContext], page_size: int) -> int:
    """Returns the plan extent a verify step's merged window needs.

    The one-shot decompression plan's total row count, computed on the host
    from the same terms ``cache_valid_length_for_context(ctx, 0)`` sums --
    cached tokens, the step's scheduled rows and the drafts it verifies --
    each request rounded up to a page, as the plan kernel rounds it. An
    undersized extent silently corrupts (the copy launches only ``length``
    rows), so every consumer must pass this or something larger, never a
    plain decode-step budget.
    """
    return sum(
        _page_aligned(cache_valid_length_for_context(ctx, 0), page_size)
        for ctx in contexts
    )


def spec_block_extent_bound(
    contexts: Sequence[TextContext], page_size: int, num_draft_tokens: int
) -> int:
    """Returns an upper bound on the drafter block's plan extent.

    The block runs after the verify commits: its cache holds the
    pre-verify length plus the committed rows -- at most the whole merged
    window -- and it queries ``num_draft_tokens`` block rows on top, so
    each request needs at most its ``cache_valid_length_for_context`` at
    the pairing's full draft width. Device-side acceptance decides the
    commit, so the exact value is only knowable on the device; this bound
    is the host's stand-in, and the block's real extent never exceeds it.
    """
    return sum(
        _page_aligned(
            cache_valid_length_for_context(ctx, num_draft_tokens), page_size
        )
        for ctx in contexts
    )


def k3_spec_mla_extent(
    *,
    exact: TensorValue,
    kv: KVCacheInputsPerDevice[TensorValue, BufferValue],
    input_row_offsets: TensorValue,
    num_draft_tokens: int,
    page_size: int,
    mla_buffer_size: int,
) -> TensorValue:
    """Selects this call's MLA prefill-plan extent on the host, in-graph.

    Returns the host ``[1]`` int32 to pass as the plan's ``buffer_length``.
    ``exact`` is the batched value this call wants -- the verify's
    :func:`spec_verify_extent`, the drafter block's
    :func:`spec_block_extent_bound` -- and ``kv`` is
    either cache leaf: its ``max_prompt_length`` names the widest merged
    window any request schedules, and its ``max_cache_length`` the bound
    arm keys on; the target's and the drafter's leaves carry the same
    host values. ``input_row_offsets`` is the batch's per-request row
    offsets, whose length names the batch size the bound charges.
    ``mla_buffer_size`` is the one-shot plan's row cap.

    ``kv`` is one replica's leaf, so the bound sizes from that replica's
    ``max_cache_length`` alone; Kimi K3 refuses data parallelism
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
    return ops.where(
        decode,
        bound,
        ops.cast(exact, DType.int32),
    )
