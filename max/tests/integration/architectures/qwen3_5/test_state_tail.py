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
"""Tests the state tail a speculative Qwen3.5 graph declares.

An engine driving the exported graph binds this tail by position, so its
order and arity are pinned here.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import RecurrentStateRegion
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.spec_adapters import state_tail_types
from max.pipelines.architectures.qwen3_5.state_cache import (
    COMPILED_RING_LENS,
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    RING_LEAF_ID,
    linear_state_regions,
    ring_len_for_window,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.speculative import SpeculativeConfig

NUM_DRAFTS = 3


def _regions(
    ring_len: int, num_devices: int = 1
) -> tuple[RecurrentStateRegion, ...]:
    """The published geometry's regions, the ring last."""
    return linear_state_regions(
        num_linear_layers=48,
        key_head_dim=128,
        num_key_heads=16,
        value_head_dim=128,
        num_value_heads=48,
        conv_kernel_dim=4,
        dtype=DType.bfloat16,
        num_devices=num_devices,
        ring_len=ring_len,
    )


def _input_types() -> list[TensorType | BufferType]:
    return state_tail_types(
        _regions(ring_len_for_window(1 + NUM_DRAFTS)), [DeviceRef.GPU()]
    )


def test_the_tail_is_the_live_leaves_then_the_ring() -> None:
    """Checks the pools come in declaration order, the ring last."""
    pools = [
        str(t.shape[0]) for t in _input_types() if isinstance(t, BufferType)
    ]

    def rows(leaf: str) -> str:
        return f"{leaf.replace('/', '_')}_rows"

    assert pools == [
        rows(CONV_LEAF_ID),
        rows(RECURRENT_LEAF_ID),
        rows(RING_LEAF_ID),
    ]


def test_every_leaf_takes_a_row_table() -> None:
    """Checks the ring takes engine rows, like the live leaves."""
    tables = [
        t
        for t in _input_types()
        if isinstance(t, TensorType) and str(t.shape[1]) == "batch_size"
    ]
    assert len(tables) == 3, f"got {tables}"


def test_the_ring_is_float32_whatever_the_pool_dtype() -> None:
    """Checks the ring is float32."""
    rings = [
        t.dtype
        for t in _input_types()
        if isinstance(t, BufferType) and str(t.shape[0]).endswith("_ring_rows")
    ]
    assert rings == [DType.float32]


def test_the_verify_window_rounds_up_to_a_compiled_ring() -> None:
    """Checks the verify window rounds up to a compiled ring length."""
    assert ring_len_for_window(1 + 1) == 2
    assert ring_len_for_window(1 + 2) == 4
    assert ring_len_for_window(1 + NUM_DRAFTS) == 4
    assert ring_len_for_window(1 + 7) == 8
    with pytest.raises(ValueError, match="longer than any"):
        ring_len_for_window(max(COMPILED_RING_LENS) + 1)


@pytest.mark.parametrize("ring_len", [2, 4, 8])
@pytest.mark.parametrize("num_devices", [1, 2])
def test_the_ring_page_divides_the_live_pages(
    ring_len: int, num_devices: int
) -> None:
    """Checks the ring's padded page leaves the huge block unchanged.

    At the published geometry a record holds the raw key and three value
    heads' delta rows and decays, 515 elements, padded to 640.
    """
    *live, ring = _regions(ring_len, num_devices)
    assert ring.leaf_id == RING_LEAF_ID and ring.scratch
    assert ring.row_shape == (16 // num_devices, ring_len, 640)
    live_lcm = math.lcm(*(region.bytes_per_page for region in live))
    assert live_lcm % ring.bytes_per_page == 0


def test_a_speculative_cache_holds_the_ring() -> None:
    """Checks the config sizes the ring a speculative verify records into,
    and none without speculation."""
    speculative = SpeculativeConfig(
        speculative_method="mtp", num_speculative_tokens=NUM_DRAFTS
    )
    with_spec = cast("PipelineConfig", SimpleNamespace(speculative=speculative))
    without = cast("PipelineConfig", SimpleNamespace(speculative=None))
    assert Qwen3_5Config._verify_ring_len(with_spec) == ring_len_for_window(
        1 + NUM_DRAFTS
    )
    assert Qwen3_5Config._verify_ring_len(without) == 0
