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

from __future__ import annotations

from unittest.mock import MagicMock, NonCallableMock

import pytest
from max.driver import DeviceSpec
from max.pipelines.architectures.deepseekV4.arch import deepseekV4_arch
from max.pipelines.architectures.deepseekV4.memory_planner import (
    DeepseekV4MemoryPlanner,
    prefill_arena_bytes_per_token,
)
from max.pipelines.kv_cache.memory_planner import ModelConfigWithKVCache
from max.pipelines.lib import PipelineConfig

# Memory-manager allocation logs, TP=2 on 2x B200, per device (MXSERV-569):
# (max_length, trunk layers, CE tokens, arena bytes).
_MEASURED = [
    (4096, 43, 1024, 5_133_310_464),
    (4096, 43, 7168, 35_888_315_392),
    (4096, 8, 1024, 4_733_510_144),
    (2048, 8, 1000, 2_876_710_400),
    (1024, 8, 1000, 2_002_646_528),
]


def _pipeline_config(
    *,
    role: str = "prefill_and_decode",
    chunked: bool = True,
    capture: bool = True,
) -> NonCallableMock:
    config = NonCallableMock(spec=PipelineConfig)
    config.model = MagicMock()
    config.runtime = MagicMock()
    config.model.max_length = 4096
    config.model.device_specs = [
        NonCallableMock(spec=DeviceSpec) for _ in range(2)
    ]
    config.runtime.pipeline_role = role
    config.runtime.max_batch_input_tokens = 8192
    config.runtime.max_num_input_tokens = 8192 if chunked else None
    config.runtime.device_graph_capture = capture
    return config


def _huggingface_config() -> MagicMock:
    config = MagicMock()
    config.num_hidden_layers = 43
    return config


def _planner() -> DeepseekV4MemoryPlanner:
    config = MagicMock(spec=ModelConfigWithKVCache)
    config.get_kv_params.return_value = MagicMock()
    return DeepseekV4MemoryPlanner(config)


def test_arch_plans_through_the_v4_planner() -> None:
    assert deepseekV4_arch.memory_planner is DeepseekV4MemoryPlanner


@pytest.mark.parametrize("max_length,layers,tokens,measured", _MEASURED)
def test_estimate_covers_the_measured_arena(
    max_length: int, layers: int, tokens: int, measured: int
) -> None:
    """Never below what the allocator was asked for, and within 2% of it."""
    estimate = tokens * prefill_arena_bytes_per_token(max_length, layers)
    assert measured <= estimate <= measured * 1.02


def test_reserves_the_full_ce_batch_on_every_device() -> None:
    estimate = _planner().estimate_activation_memory(
        _pipeline_config(), _huggingface_config()
    )
    assert estimate == 2 * 8192 * prefill_arena_bytes_per_token(4096, 43)
    # The batch that OOMed run 36376133383 asked for 38.40 GB on cuda[0].
    assert estimate // 2 > 38.40 * 1000**3


def test_decode_only_reserves_nothing() -> None:
    estimate = _planner().estimate_activation_memory(
        _pipeline_config(role="decode_only"), _huggingface_config()
    )
    assert estimate == 0


def test_unchunked_prefill_reserves_a_whole_prompt() -> None:
    config = _pipeline_config(chunked=False)
    config.runtime.max_batch_input_tokens = 1024
    estimate = _planner().estimate_activation_memory(
        config, _huggingface_config()
    )
    assert estimate == 2 * 4096 * prefill_arena_bytes_per_token(4096, 43)


def test_graph_capture_does_not_move_the_estimate() -> None:
    """Capture holds nothing inside the memory manager's budget.

    Measured: the same 121.56 GiB in use at the first CE batch with capture
    on and off, so the arena is the only term to reserve.
    """
    planner = _planner()
    assert planner.estimate_activation_memory(
        _pipeline_config(capture=True), _huggingface_config()
    ) == planner.estimate_activation_memory(
        _pipeline_config(capture=False), _huggingface_config()
    )
