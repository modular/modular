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
"""Draft-aware sizing for the K3 (ModuleV3) MLA layers under speculative decode.

A verify step feeds each request ``1 + num_draft_tokens`` rows (the anchor
plus DSpark's drafted block) through the target's MLA layers, while the
host-side numbers that size the latent path were all derived from plain
decode's single row. The device-computed quantities are already safe: the
prefill plan kernel (``mla_prefill_plan_kernel``) reads the runtime
``cache_lengths`` and ``input_row_offsets`` tensors, so its window is
``align_up(cache_length + active_length, page_size)`` whatever the active
width is. What still carries the decode assumption are the host-side knobs,
each owned by a different file:

* the MLA cache leaf's ``num_draft_tokens`` (``model_config.py``'s
  ``_attn_kv_params`` -- pass :func:`kimi_k3_spec_num_draft_tokens`). Without
  it the block manager sizes a request's pages from
  ``_compute_seq_len(ctx, 0, 0)`` and a draft-carrying step writes past the
  last allocated page: the plan/kernel window is right but the pages it
  indexes are one short, the write lands on the null page and the read-back
  returns its zeros. This is the ModuleV3 form of a defect V2 already fixed
  (``draft_aware_batch_context_length``: one page short when a 9-row verify
  crossed a 128-token page).
* the prefill plan's ``buffer_size`` cap
  (``model_config.py``'s ``apply_runtime`` sizes ``max_batch_context_length``
  with the draft-aware :func:`kimi_k3_mla_buffer_size` for a speculative
  pairing, and keeps the batch token budget otherwise -- the amount the
  context-encoding scheduler already charges). The plan kernel clamps every chunk to
  this cap and, at ``max_chunks=1``, silently drops the rest; which
  regime bounds it depends on the verify kernel arm, so see that
  function for why the decode-fold and prefill arms differ.
* the rope table's row count, which the plan cap must not size: the
  table is indexed at ONE request's absolute positions -- the context
  limit plus its verify rows -- never the batch-wide window the cap
  covers, and on the prefill arm that window spans every batch slot's
  copy. ``model.py`` builds it from :func:`kimi_k3_mla_table_rows`.
"""

from __future__ import annotations

from max.dtype import DType
from max.pipelines.lib import PipelineConfig
from max.support.math import ceildiv

from ..qwen3_5.state_cache import ring_len_for_window

__all__ = [
    "kimi_k3_mla_buffer_size",
    "kimi_k3_mla_folds_on_decode_branch",
    "kimi_k3_mla_table_rows",
    "kimi_k3_mla_verify_rows",
    "kimi_k3_spec_num_draft_tokens",
    "kimi_k3_verify_ring_len",
]

# The MTP verify fold these constants size: `AMD_MLA_DECODE_FOLD_MAX_NUM_HEADS`,
# `AMD_MLA_DECODE_MTP_SEQ_LEN` and the M bound those heads x rows pack into in
# the MLA kernel (`mla.mojo`, `mha_decode_partition_heuristic.mojo`). They model
# the MTP verify fold only (1 + draft_width <= 8 query rows); they do not model
# the kernel's wider short prompt chunk fold, which routes 8 < S <= 128 through
# the same decode branch but never serves a verify step.
_MLA_DECODE_FOLD_MAX_NUM_HEADS = 16
_MLA_DECODE_MAX_SEQ_LEN = 8
_MLA_DECODE_FOLD_M_MAX = 128


def kimi_k3_spec_num_draft_tokens(pipeline_config: PipelineConfig) -> int:
    """Returns the draft width the MLA cache leaf must be sized for.

    Zero without a speculative config, and for any method besides DSpark:
    only DSpark block-drafts through this tower, so every other config
    keeps the plain decode-sized leaf -- the KV cache sizing must not be
    what rejects it. DSpark block-drafts one ``num_speculative_tokens``-wide
    block per step, so that width is the leaf's ``num_draft_tokens``: the
    paged cache manager reads it to reserve pages for the drafts a step
    writes (``_compute_seq_len``) and to keep ``num_draft_tokens_per_step``
    (which selects the block-draft accounting) nonzero.

    Raises:
        ValueError: If DSpark's width was never resolved.
    """
    speculative = pipeline_config.speculative
    if speculative is None or speculative.speculative_method != "dspark":
        return 0
    num_draft_tokens = speculative.num_speculative_tokens
    if num_draft_tokens is None:
        raise ValueError(
            "dspark needs num_speculative_tokens resolved before the KV cache"
            " is sized: pass --num-speculative-tokens, or build the pipeline"
            " with the K3DSparkModel draft architecture, whose"
            " checkpoint_draft_width returns the trained width (the"
            " published draft config carries no block_size to read it"
            " from)."
        )
    return num_draft_tokens


def kimi_k3_mla_verify_rows(num_draft_tokens: int) -> int:
    """Returns the rows one request feeds a verify step: anchor plus drafts."""
    return num_draft_tokens + 1


def kimi_k3_mla_table_rows(
    *, max_seq_len: int, page_size: int, num_draft_tokens: int
) -> int:
    """Returns the rows the identity rope table needs.

    One request at the context limit, plus its verify rows' positions past
    it, padded to a page. The table is indexed at a single request's
    absolute positions, never the batch-wide window the plan cap covers,
    so the cap must not size it -- on the prefill arm the cap spans every
    batch slot's copy of this window (see the module docstring).
    """
    verify_rows = kimi_k3_mla_verify_rows(num_draft_tokens)
    return ceildiv(max_seq_len + verify_rows, page_size) * page_size


def kimi_k3_verify_ring_len(pipeline_config: PipelineConfig) -> int:
    """Returns the verify ring length the state cache must hold, or zero.

    A speculative verify records its whole ``1 + draft_width`` window, so
    the smallest compiled ring that holds it is the scratch leaf's length;
    a config without speculation declares none.
    """
    speculative = pipeline_config.speculative
    if speculative is None:
        return 0
    return ring_len_for_window(1 + speculative.draft_width)


def kimi_k3_mla_folds_on_decode_branch(
    num_q_heads_per_device: int,
    cache_dtype: DType,
    device_api: str,
    num_query_rows: int = 1,
) -> bool:
    """Whether the decode branch's fold can serve a multi-row verify step.

    The MTP verify fold's whole envelope, clause for clause as
    ``mla_decode_max_seq_len`` gates it for a verify step (the kernel also
    folds 8 < S <= 128 prompt chunks on the decode branch, which no verify
    reaches): no more than ``MLA_DECODE_MAX_SEQ_LEN`` (8) query rows per
    request everywhere, and on AMD (``device_api == "hip"``) also FP8 only,
    at most ``AMD_MLA_DECODE_FOLD_MAX_NUM_HEADS`` (16) query heads per
    device, and the product packed within ``AMD_MLA_DECODE_FOLD_M_MAX``
    (128) -- so a BF16 cache or a wider head set on AMD, or more rows than
    the fold holds anywhere, routes to the one-shot decompression prefill
    arm. Off AMD the fold serves any cache dtype and head count. Written
    once here because two sizing decisions key on which arm serves a
    step: this module's plan cap and the DSpark drafter's own plan cap
    (``dspark/model_config.py``) -- they must call this, not restate it, or
    the two can drift.
    """
    return num_query_rows <= _MLA_DECODE_MAX_SEQ_LEN and (
        device_api != "hip"
        or (
            cache_dtype.is_float8()
            and num_q_heads_per_device <= _MLA_DECODE_FOLD_MAX_NUM_HEADS
            and num_q_heads_per_device * num_query_rows
            <= _MLA_DECODE_FOLD_M_MAX
        )
    )


def kimi_k3_mla_buffer_size(
    *,
    max_batch_context_length: int,
    max_seq_len: int,
    page_size: int,
    num_draft_tokens: int,
    max_batch_size: int,
    num_q_heads_per_device: int,
    cache_dtype: DType,
    device_api: str,
) -> int:
    """Returns the cap the MLA plan window must be sized for.

    The rope table takes its own row count (:func:`kimi_k3_mla_table_rows`),
    not this cap.

    The scheduler's total-context budget (``max_batch_context_length``)
    bounds a batch only through context-encoding admission:
    ``TotalContextTokenBudget`` applies to CE and MIXED requests alone
    (``text_batch_constructor.py`` builds it with ``applicable_types=[CE,
    MIXED]``), so a decode-only (TG) batch's committed context is never
    charged against it and is bounded only by what the KV cache holds. What
    the cap must cover therefore depends on which kernel arm serves an
    S > 1 verify (``mla_decode_max_seq_len`` -- on AMD the fold is
    FP8-only, so the arm follows the cache dtype as well as the head
    count):

    * the verify kernel folds the verify rows into a decode step: on AMD
      that needs a float8 cache, at most 16 query heads per device and no
      more than 8 rows (K3 at TP8 in production -- 12 heads and an FP8
      latent); a verify the fold serves never reads
      the plan at all -- the decode branch takes none of the plan's
      outputs (``mla_decode_branch_bf16`` / ``_fp8`` receive no
      ``buffer_row_offsets``/``cache_offsets``/``buffer_length``) -- so the
      cap only has to bound context-encoding windows (the CE budget) and
      the page-aligned window one request plus its verify rows spans
      (:func:`kimi_k3_mla_table_rows`).
    * anything else -- a BF16 latent cache on AMD (12 heads at TP8 under
      a plain bf16 checkpoint's cache dtype), or the one-shot decompression
      prefill arm on AMD at TP2/TP4 -- where the verify step DOES read the
      plan, and its window is the TG worst case: every batch slot at
      ``max_seq_len`` context, each verifying ``1 + num_draft_tokens`` rows.
      The plan kernel clamps every chunk to the cap and, at ``max_chunks=1``,
      silently drops the rest, so an undersized cap truncates trailing
      requests' windows into NaN/garbage rather than failing. Where the
      fold serves any head count (NVIDIA), this branch only over-reserves.

    Args:
        max_batch_context_length: The scheduler's total-context budget
            (``KimiK3Config.max_batch_context_length``).
        max_seq_len: The model's maximum context length.
        page_size: The MLA cache leaf's page size in tokens.
        num_draft_tokens: The speculative draft width, from
            :func:`kimi_k3_spec_num_draft_tokens`; 0 without spec.
        max_batch_size: The scheduler's batch-slot count.
        num_q_heads_per_device: Query heads one device holds, one input to
            the fold gate below.
        cache_dtype: The MLA cache's storage dtype. On AMD the decode fold
            serves an S > 1 verify only out of a float8 cache, so a bf16
            latent cache takes the prefill arm even at the fold's head
            count.
        device_api: The accelerator API the pipeline serves on
            (``max.driver.accelerator_api()``); ``"hip"`` is AMD, whose
            fold is FP8-only.
    """
    verify_rows = kimi_k3_mla_verify_rows(num_draft_tokens)
    # One request at the context limit plus its verify rows, padded to a
    # page: the smallest window any arm's plan must still serve whole.
    table_need = kimi_k3_mla_table_rows(
        max_seq_len=max_seq_len,
        page_size=page_size,
        num_draft_tokens=num_draft_tokens,
    )
    # `mla_decode_max_seq_len`: the widest S the decode branch folds, and
    # on AMD the fold's FP8-and-head-count gate.
    fold_serves_verify = kimi_k3_mla_folds_on_decode_branch(
        num_q_heads_per_device,
        cache_dtype,
        device_api,
        num_query_rows=verify_rows,
    )
    if num_draft_tokens == 0 or fold_serves_verify:
        return max(max_batch_context_length, table_need)
    tg_worst_case = max(1, max_batch_size) * table_need
    return max(max_batch_context_length, table_need, tg_worst_case)
