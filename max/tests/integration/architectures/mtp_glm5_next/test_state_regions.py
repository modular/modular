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
"""The verify ring a speculative GLM-5.3-Flash cache declares.

A cache that does not speculate keeps its three live leaves exactly, and a
speculative one appends a scratch ring sized to the verify window whose page
never widens the pool's block.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import (
    MultiKVCacheParams,
    RecurrentStateParams,
    recurrent_leaf,
)
from max.nn.legacy.state_space import kda_ring_record_elements
from max.pipelines.architectures.glm5_next.model_config import Glm5NextConfig
from max.pipelines.architectures.glm5_next.state_cache import (
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    STATE_CACHE_KEY,
    TAIL_RING_LEAF_ID,
    VERIFY_RING_LEAF_ID,
    with_verify_ring,
)
from max.pipelines.architectures.qwen3_5.state_cache import RING_DTYPE
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    compute_jenga_ratios,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative import SpeculativeConfig

HEADS = 64
HEAD_DIM = 128
KERNEL = 4
KPOOL = 4
INDEX_HEAD_DIM = 128


def _text_config(num_hidden_layers: int) -> SimpleNamespace:
    """A GLM-5.3-Flash text config, cut down to what the caches read."""
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=INDEX_HEAD_DIM,
        index_kpool=KPOOL,
        index_topk=2048,
        num_nextn_predict_layers=1,
        layer_types=[
            "deepseek_sparse_attention" if (i % 4) == 3 else "linear_attention"
            for i in range(num_hidden_layers)
        ],
        linear_attn_config={
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": KERNEL,
        },
    )


def _cache_params(
    *,
    num_drafts: int | None,
    num_devices: int = 1,
    num_hidden_layers: int = 8,
) -> MultiKVCacheParams:
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = (
        None
        if num_drafts is None
        else SpeculativeConfig(
            speculative_method="mtp", num_speculative_tokens=num_drafts
        )
    )
    params = Glm5NextConfig.construct_kv_params(
        huggingface_config=_text_config(num_hidden_layers),
        pipeline_config=pipeline_config,
        devices=[DeviceRef.CPU(i) for i in range(num_devices)],
        kv_cache_config=KVCacheConfig(state_pool_dtype="float32"),
        cache_dtype=DType.bfloat16,
    )
    assert isinstance(params, MultiKVCacheParams)
    return params


def _state(params: MultiKVCacheParams) -> RecurrentStateParams:
    state = recurrent_leaf(params)
    assert isinstance(state, RecurrentStateParams)
    return state


_LIVE_LEAVES = [CONV_LEAF_ID, RECURRENT_LEAF_ID, TAIL_RING_LEAF_ID]


def test_a_cache_that_does_not_speculate_declares_no_ring() -> None:
    """Checks a cache without speculation keeps its three live leaves."""
    state = _state(_cache_params(num_drafts=None))
    assert [region.leaf_id for region in state.regions] == _LIVE_LEAVES


def test_the_speculative_cache_appends_the_ring() -> None:
    """Checks the ring follows the live leaves, a float32 scratch leaf."""
    *live, ring = _state(_cache_params(num_drafts=3)).regions
    assert [region.leaf_id for region in live] == _LIVE_LEAVES
    assert ring.leaf_id == VERIFY_RING_LEAF_ID
    assert ring.scratch
    assert ring.dtype == RING_DTYPE == DType.float32
    assert ring.num_layers == live[0].num_layers


@pytest.mark.parametrize(
    ("num_drafts", "ring_len"), [(1, 2), (2, 4), (3, 4), (7, 8)]
)
def test_the_ring_holds_one_verify_window(
    num_drafts: int, ring_len: int
) -> None:
    """Checks the ring rounds ``K + 1`` up to a compiled ring length."""
    state = _state(_cache_params(num_drafts=num_drafts))
    assert state.regions[-1].row_shape[1] == ring_len


@pytest.mark.parametrize("num_devices", [1, 2, 8])
def test_a_ring_row_is_one_record_per_head_and_slot(num_devices: int) -> None:
    """Checks the row is ``[heads, RING_LEN, stride]`` per device, with room
    for the record the kernel writes."""
    state = _state(_cache_params(num_drafts=3, num_devices=num_devices))
    heads, ring_len, stride = state.regions[-1].row_shape
    assert heads == HEADS // num_devices
    assert ring_len == 4
    assert stride >= kda_ring_record_elements(HEAD_DIM, HEAD_DIM)


@pytest.mark.parametrize("num_devices", [1, 8])
@pytest.mark.parametrize("num_hidden_layers", [8, 45])
def test_the_ring_never_widens_the_pool_block(
    num_devices: int, num_hidden_layers: int
) -> None:
    """Checks the ring's page divides the block the other leaves define.

    A block is the least common multiple of every leaf's page, so a ring page
    off that grid would multiply it. 45 layers is the published schedule.
    """
    params = _cache_params(
        num_drafts=3,
        num_devices=num_devices,
        num_hidden_layers=num_hidden_layers,
    )
    pages = {
        leaf_id: leaf.bytes_per_page
        for leaf_id, leaf in params.leaves().items()
    }
    without_ring = {
        leaf_id: size
        for leaf_id, size in pages.items()
        if leaf_id != VERIFY_RING_LEAF_ID
    }
    budget = 64 * 1024**3
    _, block, _ = compute_jenga_ratios(budget, pages)
    _, bare_block, _ = compute_jenga_ratios(budget, without_ring)
    assert block == bare_block


def test_the_ring_is_a_scratch_group_leaf() -> None:
    """Checks the cache allocates the ring per request and never publishes it."""
    leaves = _cache_params(num_drafts=3).leaves()
    assert leaves[VERIFY_RING_LEAF_ID].group_id.is_scratch()
    for leaf_id in (CONV_LEAF_ID, RECURRENT_LEAF_ID, TAIL_RING_LEAF_ID):
        assert leaves[leaf_id].group_id.is_recurrent()


def test_a_request_is_priced_its_ring() -> None:
    """Checks the bytes memory planning divides by include the ring.

    The planner reads ``per_request_state_bytes``, which sums the state
    child's leaves, so the ring is counted without an extra hook.
    """
    spec = _cache_params(num_drafts=3, num_devices=2)
    state = _state(spec)
    *live, ring = state.regions
    base = MultiKVCacheParams.from_params(
        {
            **spec.children,
            STATE_CACHE_KEY: replace(state, regions=tuple(live)),
        }
    )

    def priced(params: MultiKVCacheParams) -> int:
        holder = SimpleNamespace(kv_params=params, devices=[None, None])
        return Glm5NextConfig.per_request_state_bytes(
            cast(Glm5NextConfig, holder)
        )

    assert priced(spec) - priced(base) == 2 * ring.bytes_per_page > 0


def test_the_published_record_needs_no_padding() -> None:
    """Checks the published geometry's record needs no padding to fit."""
    state = _state(_cache_params(num_drafts=3, num_hidden_layers=45))
    *live, ring = state.regions
    live_lcm = math.lcm(*(region.bytes_per_page for region in live))
    assert live_lcm % ring.bytes_per_page == 0
    assert ring.row_shape[2] == kda_ring_record_elements(HEAD_DIM, HEAD_DIM)


def test_a_ring_length_the_kernels_lack_is_refused() -> None:
    """Checks a ring length outside the compiled set raises."""
    state = _state(_cache_params(num_drafts=None))
    with pytest.raises(ValueError, match="ring_len must be one of"):
        with_verify_ring(state.regions, 3)
