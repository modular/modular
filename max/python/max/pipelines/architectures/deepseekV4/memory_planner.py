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

"""Memory planner for DeepSeek-V4."""

from __future__ import annotations

import logging
from math import ceil
from typing import Any

from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.support.human_readable_formatter import to_human_readable_bytes

logger = logging.getLogger("max.pipelines")

# The lightning indexer runs on the ratio-4 layers only.
_INDEXER_RATIO = 4

# Per-device bytes one prefill token adds to the forward's activation arena,
# fitted to memory-manager allocation logs (MXSERV-569): DeepSeek-V4-Flash-0731
# and its 8-layer minimized variant, TP=2 on 2x B200, max_length 1024 to 4096,
# 500 to 4000 CE tokens. The fit is linear in all three terms to within 0.1%
# (11.6 B per candidate, 444 B per layer, 1.083 MB base); the constants round
# it up by under 1%. The indexer scores straight from the paged leaf, so no
# per-token candidate table is built and the arena barely grows with either
# the candidate count or depth.
_ARENA_BYTES_PER_TOKEN_BASE = 1_090_000
_ARENA_BYTES_PER_TOKEN_PER_CANDIDATE = 12
_ARENA_BYTES_PER_TOKEN_PER_LAYER = 450


def prefill_arena_bytes_per_token(max_length: int, num_layers: int) -> int:
    """Per-device activation bytes one prefill token costs.

    Args:
        max_length: The model's maximum sequence length, which sizes the
            indexer's candidate axis (``2 * ceil(max_length / 4)``).
        num_layers: Trunk layers.

    Returns:
        Bytes per CE token on each device.
    """
    n_cand = 2 * ceil(max_length / _INDEXER_RATIO)
    return (
        _ARENA_BYTES_PER_TOKEN_BASE
        + _ARENA_BYTES_PER_TOKEN_PER_CANDIDATE * n_cand
        + _ARENA_BYTES_PER_TOKEN_PER_LAYER * num_layers
    )


class DeepseekV4MemoryPlanner(PagedMemoryPlanner):
    """Reserves the prefill activation arena out of the KV budget.

    Without it the arena (~8.6 GiB per device for an 8192-token CE batch at
    max_length 4096) lives in whatever ``device_memory_utilization`` leaves
    over, and the first full CE batch OOMs whenever that slack is smaller.
    """

    def estimate_activation_memory(
        self,
        pipeline_config: Any,
        huggingface_config: Any,
    ) -> int:
        """Estimates the largest CE forward's arena, summed over devices.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            Activation memory in bytes across all devices.
        """
        runtime = pipeline_config.runtime
        if runtime.pipeline_role == "decode_only":
            return 0
        max_length = pipeline_config.model.max_length
        # Without chunked prefill a whole prompt is one CE forward.
        ce_tokens = runtime.max_num_input_tokens or max(
            runtime.max_batch_input_tokens, max_length
        )
        per_token = prefill_arena_bytes_per_token(
            max_length, huggingface_config.num_hidden_layers
        )
        n_devices = len(pipeline_config.model.device_specs)
        activation = ce_tokens * per_token * n_devices
        logger.info(
            "Estimated DeepSeek-V4 prefill activation memory: %s "
            "(%d CE tokens x %s per token x %d devices)",
            to_human_readable_bytes(activation),
            ce_tokens,
            to_human_readable_bytes(per_token),
            n_devices,
        )
        return activation
