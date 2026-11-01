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

"""Helpers shared by the legacy and Jenga paged KV cache managers."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

from max.pipelines.context import (
    TextAndVisionContext,
    TextContext,
    TokenHashOverride,
)
from max.pipelines.modeling.types import RequestID

from .block_utils import KVHashAlgo, hash_request_tokens

logger = logging.getLogger("max.pipelines")

KV_CACHE_LENGTHS = "kv_cache_lengths"
"""Device input holding how much cache each request already has."""


def compute_block_hashes(
    ctx: TextContext,
    existing_hashes: Sequence[bytes],
    block_size: int,
    kv_hash_algo: KVHashAlgo,
    kv_hash_seed: bytes | None,
) -> list[bytes]:
    """Computes block hashes for the request beyond ``existing_hashes``.

    Unlike :meth:`compute_hashes_for_request`, this reads and writes no
    per-request state, so it is safe to call for requests that are not
    (and may never be) claimed on this replica — for example, when computing
    prefix-cache overlap for data-parallel routing.

    Args:
        ctx: The request context.
        existing_hashes: Hashes already computed for the request's leading
            blocks; new hashes chain onto the last entry. Pass an empty
            sequence to hash from the start of the prompt.
        block_size: How many tokens one block holds.
        kv_hash_algo: Which hash the keys are built with.
        kv_hash_seed: Seed isolating one deployment's keys from another's, or
            ``None`` for the unseeded default.

    Returns:
        Hashes for the newly hashed full blocks; empty if no additional
        full block is hashable.
    """
    num_hashed_tokens = len(existing_hashes) * block_size
    # We do not compute the hash for the last token because it is ineligible
    # for prefix caching. This is because 100% prefix cache hit is illegal
    # and will result in a 0 input tokens for the request. Hence the minus 1.
    num_hashable_tokens = len(ctx.tokens) - 1
    num_unhashed_tokens = num_hashable_tokens - num_hashed_tokens
    if num_unhashed_tokens < block_size:
        return []

    parent_hash_value: bytes | None = None
    if len(existing_hashes) > 0:
        parent_hash_value = existing_hashes[-1]

    unhashed_tokens = ctx.tokens[num_hashed_tokens:num_hashable_tokens]

    token_hash_overrides: list[TokenHashOverride] = []
    if isinstance(ctx, TextAndVisionContext):
        for img in ctx.images:
            if img.image_hash is None:
                raise ValueError(
                    "hash_request_tokens requires `image_hash` to be present. Found None."
                )
            token_hash_overrides.append(
                TokenHashOverride(
                    token_idx=img.start_idx,
                    token_hash=img.image_hash,
                    source="image",
                )
            )
        token_hash_overrides.extend(ctx.token_hash_overrides)

    return hash_request_tokens(
        token_ids=unhashed_tokens,
        block_size=block_size,
        parent_hash=parent_hash_value,
        prefix_length=num_hashed_tokens,
        token_hash_overrides=token_hash_overrides,
        algo=kv_hash_algo,
        seed=kv_hash_seed,
        salt=ctx.cache_salt,
    )


@dataclass(frozen=True)
class _OnloadSplice:
    """The prefix an onload spliced into its request, as committed indices.

    A failed copy rolls the request back from ``end_idx`` to ``start_idx``,
    which spans the device hit as well as the onload. Its next reuse takes the
    device hit again, without the connector.
    """

    request_id: RequestID
    start_idx: int
    end_idx: int


def _compute_seq_len(
    ctx: TextContext,
    num_draft_tokens: int,
    num_draft_tokens_per_step: int = 0,
    max_num_input_tokens: int | None = None,
) -> int:
    # Each term accounts for one category of tokens that need a KV slot:
    #
    #   ctx.tokens                    : prompt + tokens generated so far
    #   maybe_accepted_draft_tokens   : draft tokens being verified in the
    #                                   *previous* batch (overlap scheduler);
    #                                   conservative: assume all are accepted
    #   2 * num_draft_tokens          : drafts to verify *next* batch
    #                                   + drafts written *during* that batch
    #   1                             : one regular decode step
    #   -1                            : the last generated token has no KV
    #                                   entry yet (overlap's in-flight
    #                                   placeholder, or the just-sampled token)
    #
    # Block-draft correction (DFlash): the draft model's ``forward_block``
    # writes ``num_draft_tokens_per_step + 1`` positions in a single batched
    # call, starting at ``bumped_cache_length = pre_cache_length +
    # commit_lengths``. Compared to the autoregressive-draft accounting above
    # (which assumes one draft KV per step), that's an extra position past the
    # bonus token — exactly the slot the ``- 1`` here was reclaiming under the
    # "last generated token has no KV entry" optimization. For block drafts
    # that bonus position *does* get a KV entry (forward_block writes it as
    # part of the speculative tail), so we add it back. Guard on
    # ``num_draft_tokens > 0`` so a disabled request (0, 0) never matches this
    # by coincidence.
    block_draft_extra = (
        1
        if num_draft_tokens > 0
        and num_draft_tokens_per_step == num_draft_tokens
        else 0
    )
    # Avoid allocating blocks for the entire sequence when chunked prefill is used.
    # This is critically important for SWA.
    active_length = ctx.tokens.active_length
    if max_num_input_tokens is not None:
        active_length = min(active_length, max_num_input_tokens)
    seq_len = ctx.tokens.processed_length + active_length
    seq_len += (
        len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
        + 2 * num_draft_tokens
        + 1
        + block_draft_extra
        - 1
    )
    return seq_len


def _resolve_only_use_kv_connector_last_level_cache() -> bool:
    """Resolve whether to only use the KVConnector last level cache.

    When this is set, the device prefix cache will be disabled. All KVCache hits
    will strictly be served from the KVConnector. This is primarily used for
    testing and benchmarking the performance of the KVConnector. Do NOT use this
    flag in production.

    With the local connector, the last level cache is the host memory. With the
    tiered connector, the last level cache is the disk.
    """
    enabled = os.getenv(
        "MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE", "0"
    ).lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if enabled:
        logger.info(
            "Detected MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE flag, only using KVConnector prefix cache."
        )
    return enabled


@dataclass(frozen=True)
class PrefixCacheHits:
    """Per-tier counts of a request's contiguous cached prefix on one replica.

    Counts are blocks, not tokens, and describe one contiguous run from the
    start of the request's block hash chain: the leading ``device_blocks``
    are resident in the device prefix cache, and ``external_blocks``
    continue that run from the connector's tiers.

    The external count is not split by tier. ``KVConnector.lookup`` reports
    whether a connector holds a block, not which of its tiers holds it, and
    which tier a block is read from is the connector's business -- a disk hit
    is promoted through host memory on its way to the device anyway.
    """

    device_blocks: int = 0
    """Leading blocks resident in the device prefix cache."""

    external_blocks: int = 0
    """Blocks continuing the run that the connector's tiers hold."""

    @property
    def total_blocks(self) -> int:
        """Total contiguous cached blocks across all tiers."""
        return self.device_blocks + self.external_blocks


def prompt_tokens_for_context(ctx: TextContext) -> int:
    """Returns the per-step query (prompt) width for ``ctx``.

    The new tokens processed this step plus any draft tokens to verify. Matches
    the decode kernel's ``q_max_seq_len`` (prefill: full prompt; decode:
    ``1 + num_draft_tokens_to_verify``).
    """
    return ctx.tokens.active_length + len(
        ctx.spec_decoding_state.draft_tokens_to_verify
    )


def cache_valid_length_for_context(
    ctx: TextContext, num_draft_tokens: int
) -> int:
    """Returns the maximum valid cache length this forward reads for ``ctx``.

    Already-cached tokens (processed + accepted draft) plus this step's query
    width (:func:`prompt_tokens_for_context`) plus the draft tokens written this
    step (``num_draft_tokens``). Shared with the graph-capture replay path so
    its upper-bound characteristics match what this manager prepares.
    """
    return (
        ctx.tokens.processed_length
        + len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
        + prompt_tokens_for_context(ctx)
        + num_draft_tokens
    )
