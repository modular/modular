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
"""Unit tests for ``Gemma4MemoryPlanner.estimate_activation_memory``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from max.pipelines.architectures.gemma4.memory_planner import (
    _ACTIVATION_SAFETY_FACTOR,
    _PREFILL_TOKENS_PER_STEP,
    Gemma4MemoryPlanner,
)

_GIB = 1024**3


def _pipeline_config(
    *,
    kv_bytes: int = 2,
    max_batch_size: int = 1,
    num_devices: int = 1,
    graph_capture: bool = False,
) -> MagicMock:
    pc = MagicMock()
    pc.model.device_specs = [object()] * num_devices
    pc.model.kv_cache.cache_dtype.size_in_bytes = kv_bytes
    pc.runtime.max_batch_size = max_batch_size
    pc.runtime.device_graph_capture = graph_capture
    return pc


def _hf(hidden_size: int, intermediate_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=hidden_size, intermediate_size=intermediate_size
        )
    )


def _planner() -> Gemma4MemoryPlanner:
    # ``estimate_activation_memory`` reads only its arguments, never ``self``,
    # so skip ``__init__`` (and its model-config validation) entirely.
    return Gemma4MemoryPlanner.__new__(Gemma4MemoryPlanner)


def test_scaled_estimate_below_flat_for_small_batch() -> None:
    """31B-class dims: the scaled estimate is far below the 15 GiB flat value."""
    flat = (30 // 2) * _GIB
    width = 21504
    got = _planner().estimate_activation_memory(
        _pipeline_config(), _hf(5376, width)
    )
    expected = _ACTIVATION_SAFETY_FACTOR * _PREFILL_TOKENS_PER_STEP * width * 2
    assert got == expected
    assert got < flat


def test_capped_at_previous_flat_value() -> None:
    """A pathologically wide model is clamped to the old flat reservation."""
    flat = (30 // 2) * _GIB
    got = _planner().estimate_activation_memory(
        _pipeline_config(), _hf(1_000_000, 1_000_000)
    )
    assert got == flat


def test_fp8_kv_cache_uses_larger_flat_cap() -> None:
    """The flat cap still scales with the KV cache dtype (30 GiB at 1 byte)."""
    flat_fp8 = (30 // 1) * _GIB
    got = _planner().estimate_activation_memory(
        _pipeline_config(kv_bytes=1), _hf(1_000_000, 1_000_000)
    )
    assert got == flat_fp8


def test_falls_back_to_flat_when_dims_missing() -> None:
    """Missing model dimensions fall back to the conservative flat value."""
    flat = (30 // 2) * _GIB
    got = _planner().estimate_activation_memory(
        _pipeline_config(), SimpleNamespace()
    )
    assert got == flat


def test_batch_term_inert_below_prefill_floor() -> None:
    """Below the prefill floor, ``max_batch_size`` does not change the estimate
    (the per-step token count is dominated by ``_PREFILL_TOKENS_PER_STEP``)."""
    small = _planner().estimate_activation_memory(
        _pipeline_config(max_batch_size=1), _hf(5376, 21504)
    )
    big = _planner().estimate_activation_memory(
        _pipeline_config(max_batch_size=_PREFILL_TOKENS_PER_STEP - 1),
        _hf(5376, 21504),
    )
    assert small == big


def test_graph_capture_adds_headroom() -> None:
    """Enabling device graph capture adds a fixed 2 GiB headroom per device."""
    without = _planner().estimate_activation_memory(
        _pipeline_config(graph_capture=False), _hf(5376, 21504)
    )
    with_capture = _planner().estimate_activation_memory(
        _pipeline_config(graph_capture=True), _hf(5376, 21504)
    )
    assert with_capture - without == 2 * _GIB


def test_scales_with_device_count() -> None:
    """The total reservation is the per-device value times the device count."""
    one = _planner().estimate_activation_memory(
        _pipeline_config(num_devices=1), _hf(5376, 21504)
    )
    four = _planner().estimate_activation_memory(
        _pipeline_config(num_devices=4), _hf(5376, 21504)
    )
    assert four == one * 4
