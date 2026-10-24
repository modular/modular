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


"""The cache a Qwen3.5 target and a DFlash2 draft share.

The pairing allocates one tree, the target's cache with its recurrent state
and the draft's own windowed leaf. A mismatch between what the tree holds and
what the graph and the engine bind is a silent wrong answer, not an error, so
the leaves' geometry is pinned here against the code that derives it.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

from dflash2_pairing_harness import (
    DRAFT_ARCH,
    TARGET_ARCH,
    Geometry,
    kv_tree,
    state_regions,
    target_config,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
    recurrent_leaf,
)
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.dflash2_qwen3_5.model_config import (
    DRAFT_SLIDING_WINDOW,
)
from max.pipelines.architectures.qwen3_5.memory_planner import (
    Qwen3_5MemoryPlanner,
)
from max.pipelines.architectures.qwen3_5.state_cache import (
    ATTN_CACHE_KEY,
    RING_LEAF_ID,
    STATE_CACHE_KEY,
    ring_len_for_window,
)
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _kv_tree,
    _Pairing,
    _without_tail,
)
from max.pipelines.speculative import SpeculativeConfig, spec_adapters

BLOCK = 8
DRAFT_LAYERS = 5
DRAFT_KV_HEADS = 2
DRAFT_HEAD_DIM = 16
PAGE_SIZE = 32


def _speculative() -> SpeculativeConfig:
    return SpeculativeConfig(
        speculative_method="dflash2", num_speculative_tokens=BLOCK - 1
    )


def _draft_tree(target_kv: MHAKVCacheParams) -> MultiKVCacheParams:
    """Derives the tree the way a served pairing does, from the draft
    checkpoint's own config."""
    target = PIPELINE_REGISTRY.retrieve_architecture(TARGET_ARCH)
    draft = PIPELINE_REGISTRY.retrieve_architecture(DRAFT_ARCH)
    assert target is not None and draft is not None
    kv_cache = KVCacheConfig(enable_prefix_caching=False)
    # A served target's leaf takes its connector config from the same
    # KVCacheConfig as its draft's, and the shared pool requires them equal.
    target_kv = replace(
        target_kv, kv_connector_config=kv_cache.kv_connector_config
    )
    pipeline_config = SimpleNamespace(
        speculative=_speculative(),
        model=SimpleNamespace(
            kv_cache=kv_cache,
            data_parallel_degree=1,
        ),
        draft_model=SimpleNamespace(
            huggingface_config=SimpleNamespace(
                num_key_value_heads=DRAFT_KV_HEADS,
                head_dim=DRAFT_HEAD_DIM,
                num_hidden_layers=DRAFT_LAYERS,
            )
        ),
    )
    return _kv_tree(
        cast("PipelineConfig", pipeline_config),
        _Pairing(TARGET_ARCH, target, draft),
        target_kv,
    )


def _target_leaf(dtype: DType = DType.bfloat16) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=dtype,
        devices=[DeviceRef.CPU()],
        n_kv_heads=1,
        head_dim=16,
        num_layers=2,
        page_size=PAGE_SIZE,
    )


def test_the_served_cache_holds_the_state_and_its_ring() -> None:
    """Checks the allocated cache carries the state child, ring included.

    MAX serves this graph from that cache, and the graph's signature drops
    the child because its tail declares the state instead.
    """
    tree = kv_tree(Geometry(), target_config(Geometry()))
    assert set(tree.children) == {"target", "draft"}
    target = tree.children["target"]
    assert isinstance(target, MultiKVCacheParams)
    assert set(target.children) == {ATTN_CACHE_KEY, STATE_CACHE_KEY}
    state = recurrent_leaf(target)
    assert state is not None
    (ring,) = (r for r in state.regions if r.leaf_id == RING_LEAF_ID)
    assert ring.scratch and ring.row_shape[1] == ring_len_for_window(BLOCK)

    tail = spec_adapters.tail_kv(TARGET_ARCH)
    assert tail == STATE_CACHE_KEY
    declared = _without_tail(tree, tail)
    declared_target = declared.children["target"]
    assert isinstance(declared_target, MultiKVCacheParams)
    assert set(declared_target.children) == {ATTN_CACHE_KEY}
    assert set(declared.children) == {"target", "draft"}


def test_the_draft_leaf_is_the_drafters_own_windowed_geometry() -> None:
    """Five drafter layers, bf16, bounded at the drafter's window.

    Every one of these is a silent-wrong-answer if it drifts from the Mach
    registry's draft group: a wrong layer count reads another layer's K/V, a
    wrong dtype reinterprets the bytes, and an unbounded leaf turns a
    per-slot constant into a cost that scales with ``--max-length``.
    """
    tree = _draft_tree(_target_leaf())
    draft = tree.children["draft"]
    assert isinstance(draft, MHAKVCacheParams)
    assert draft.num_layers == DRAFT_LAYERS
    assert draft.n_kv_heads == DRAFT_KV_HEADS
    assert draft.head_dim == DRAFT_HEAD_DIM
    assert draft.dtype == DType.bfloat16
    assert draft.window_size == DRAFT_SLIDING_WINDOW
    assert draft.group_id.is_sliding_window()
    # One page size across the tree, or the manager and the graph disagree.
    target = tree.children["target"]
    assert isinstance(target, MHAKVCacheParams)
    assert draft.page_size == target.page_size


def test_a_quantized_target_leaf_does_not_quantize_the_draft_leaf() -> None:
    """``--kv-cache-dtype float8_e4m3fn`` must not reach the drafter's cache.

    The drafter fills it with its own unquantized projections, so a rewritten
    dtype is not a precision trade -- it is a reinterpretation of the bytes.
    """
    tree = _draft_tree(_target_leaf(DType.float8_e4m3fn))
    target, draft = tree.children["target"], tree.children["draft"]
    assert isinstance(target, MHAKVCacheParams)
    assert isinstance(draft, MHAKVCacheParams)
    assert target.dtype == DType.float8_e4m3fn
    assert draft.dtype == DType.bfloat16


def test_the_draft_leaf_reserves_the_targets_draft_width() -> None:
    """Checks the drafter's leaf takes the draft width from the pairing.

    A tree's leaves must agree on it, and the drafter's checkpoint does not
    say what the target verifies.
    """
    target_kv = replace(
        _target_leaf(DType.float8_e4m3fn),
        n_kv_heads=4,
        head_dim=256,
        num_layers=16,
        page_size=256,
    )
    tree = _draft_tree(target_kv)
    draft = tree.children["draft"]
    assert isinstance(draft, MHAKVCacheParams)
    assert draft.num_draft_tokens == BLOCK - 1
    assert draft.page_size == target_kv.page_size


def test_a_request_is_priced_its_block_long_ring() -> None:
    """Checks the planner prices the ring the served cache holds."""
    config = target_config(Geometry())
    planner = Qwen3_5MemoryPlanner(config)
    pipeline_config = cast(
        "PipelineConfig", SimpleNamespace(speculative=_speculative())
    )

    ring_bytes = config._per_request_ring_bytes(ring_len_for_window(BLOCK))
    assert ring_bytes > 0
    assert planner.spec_state_bytes(pipeline_config) == ring_bytes
    assert {r.leaf_id for r in state_regions(config) if r.scratch} == {
        RING_LEAF_ID
    }
