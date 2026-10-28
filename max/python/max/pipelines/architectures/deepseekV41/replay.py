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
"""Plans the contiguous suffixes a decoder replay step runs."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from max.nn.legacy.kv_cache import KVCacheParamInterface
from max.pipelines.context import TextContext

from .model_config import DeepseekV41Config

REPLAY_MIN_TOKENS_ENV = "MODULAR_DEEPSEEK_V41_REPLAY_MIN_TOKENS"


def requested_min_tokens() -> int:
    """Returns the replay threshold the environment asks for; 0 is off."""
    value = int(os.environ.get(REPLAY_MIN_TOKENS_ENV, "0"))
    if value < 0:
        raise ValueError(f"{REPLAY_MIN_TOKENS_ENV} must be nonnegative")
    return value


def replay_min_tokens(
    config: DeepseekV41Config, device_graph_capture: object
) -> tuple[int, str | None]:
    """Returns the replay threshold and why replay is off, if it was asked for.

    The threshold is 0 when replay is off. Replay needs every DSpark tap
    after the cut layer (the last KV-source layer), so the verify can hand
    the drafter retained rows, no Engram layer after it, whose operands cover
    every row, and certified KV write coverage, which is tracked only once
    device graph capture is resolved to on or off.
    """
    value = requested_min_tokens()
    if value == 0:
        return 0, None
    taps = config.dspark_target_layer_ids
    if (
        not taps
        or min(taps) <= max(config.kv_source_layer_ids)
        or max(taps) >= config.num_hidden_layers
    ):
        return 0, "dspark_target_layer_ids are not after the KV source layers"
    cut = max(config.kv_source_layer_ids)
    if any(layer > cut for layer in config.engram_layer_ids):
        return 0, "an Engram layer comes after the KV source layers"
    if device_graph_capture is not True and device_graph_capture is not False:
        return 0, "device graph capture is unresolved"
    return value, None


def sliding_window_leaf_id(kv_params: KVCacheParamInterface) -> str:
    """Returns the id the block manager registers the window leaf under."""
    leaf_ids = [
        leaf_id
        for leaf_id, leaf in kv_params.leaves().items()
        if leaf.group_id.is_sliding_window()
    ]
    assert len(leaf_ids) == 1, leaf_ids
    return leaf_ids[0]


@dataclass(frozen=True)
class ReplayPlan:
    """Host rows and per-request front skips for a replay graph invocation.

    Indices address the merged row space: each request's prompt rows followed
    by its verification draft rows. A request's start position and SWA floor
    are derived on device from the same ``cache_lengths`` the full graph
    reads, so only the relative ``kept_skip`` crosses from the host.
    """

    kept_indices: npt.NDArray[np.int64]
    kept_offsets: npt.NDArray[np.uint32]
    kept_skip: npt.NDArray[np.int32]
    verify_width: int = 0

    def kept_longest(
        self, boundaries: Sequence[int] | npt.NDArray[np.int64]
    ) -> npt.NDArray[np.uint32]:
        """Longest retained row count per replica; 0 for an empty replica.

        ``boundaries`` are the replicas' request boundaries, one more than
        the replica count.
        """
        return np.asarray(
            [
                np.diff(self.kept_offsets[first : last + 1]).max(initial=0)
                for first, last in pairwise(boundaries)
            ],
            dtype=np.uint32,
        )


def build_replay_plan(
    contexts: Sequence[TextContext],
    *,
    window_size: int,
    min_tokens: int = 0,
    verify_width: int = 0,
    return_hidden_states: bool = False,
    enable_echo: bool = False,
) -> ReplayPlan | None:
    """Builds per-request suffixes when replay is explicitly enabled.

    A positive ``min_tokens`` enables replay for prefill batches that have
    at least that many active tokens in a request. Every request retains
    its final window plus the alignment remainder; a decode row (one active
    token) never drops anything, so it is kept whole. Prefix admission must
    already certify the cache's retained sliding window. ``verify_width``
    draft rows follow every request's prompt rows in the merged row space and
    are always kept.
    """
    return build_replay_plan_with_reason(
        contexts,
        window_size=window_size,
        min_tokens=min_tokens,
        verify_width=verify_width,
        return_hidden_states=return_hidden_states,
        enable_echo=enable_echo,
    )[0]


def build_replay_plan_with_reason(
    contexts: Sequence[TextContext],
    *,
    window_size: int,
    min_tokens: int = 0,
    verify_width: int = 0,
    return_hidden_states: bool = False,
    enable_echo: bool = False,
) -> tuple[ReplayPlan | None, str | None]:
    """Like :func:`build_replay_plan`, also naming why no plan was built.

    The reason is ``None`` exactly when a plan is returned.
    """
    if window_size <= 0:
        raise ValueError("Replay window must be positive")
    if verify_width < 0:
        raise ValueError("Replay verify width must be nonnegative")
    if min_tokens <= 0:
        return None, "disabled"
    if return_hidden_states:
        return None, "hidden_states"
    if enable_echo:
        return None, "echo"
    if not contexts:
        return None, "no_contexts"
    if any(ctx.log_probabilities_echo for ctx in contexts):
        return None, "logprob_echo"
    if not any(ctx.is_initial_prompt for ctx in contexts):
        return None, "no_initial_prompt"
    if max(ctx.tokens.active_length for ctx in contexts) < min_tokens:
        return None, "below_threshold"

    indices: list[int] = []
    offsets = [0]
    skips: list[int] = []
    original_offset = 0
    removed = False
    for ctx in contexts:
        start = ctx.tokens.processed_length
        length = ctx.tokens.active_length
        end = start + length
        # An in-flight row's device position runs ahead of processed_length by
        # its accepted drafts, which the host cannot see. Keeping it whole
        # makes its rebased start equal the full graph's, and guarantees that
        # every skipped row has device cache_lengths == processed_length.
        in_flight = bool(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
        skip = (
            0
            if in_flight
            else max(0, end // window_size * window_size - window_size - start)
        )
        removed |= skip > 0
        merged = length + verify_width
        indices.extend(range(original_offset + skip, original_offset + merged))
        offsets.append(len(indices))
        skips.append(skip)
        original_offset += merged
    if not removed:
        return None, "nothing_to_remove"
    plan = ReplayPlan(
        kept_indices=np.asarray(indices, dtype=np.int64),
        kept_offsets=np.asarray(offsets, dtype=np.uint32),
        kept_skip=np.asarray(skips, dtype=np.int32),
        verify_width=verify_width,
    )
    return plan, None


def log_replay_not_built(
    logger: logging.Logger, reason: str, contexts: Sequence[TextContext]
) -> None:
    """Logs a skipped plan for a step that has prompt rows; decode is silent."""
    initial = sum(ctx.is_initial_prompt for ctx in contexts)
    if not initial:
        return
    active = [ctx.tokens.active_length for ctx in contexts]
    logger.info(
        "DSV41 decoder replay plan not built: reason=%s requests=%d "
        "initial=%d decode=%d max_active=%d total_rows=%d",
        reason,
        len(contexts),
        initial,
        len(contexts) - initial,
        max(active),
        sum(active),
    )
