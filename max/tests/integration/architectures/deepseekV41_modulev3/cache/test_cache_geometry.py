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
"""Cache-geometry tests for the V4.1 ModuleV3 interface.

The interface must build the canonical current-main CSA2 record leaves --
the geometry the registered V2 cache, attention and indexer kernels address
-- not the draft interface's packed-float8 rows. Every record constant here
is imported from the V2 modules the kernels' wrappers live in, so a drift in
either direction fails these tests.
"""

from __future__ import annotations

from typing import Any

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.deepseekV41 import kv_leaves as v2_kv_leaves
from max.pipelines.architectures.deepseekV41.layers.csa2 import (
    COMPRESSED_DATA_BYTES,
    COMPRESSED_RECORD_BYTES,
    WINDOW_DATA_BYTES,
    WINDOW_RECORD_BYTES,
    cache_page_bytes,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
    kv_leaf_specs,
)
from max.pipelines.lib import KVCacheConfig


def _hf_config(**overrides: Any) -> dict[str, Any]:
    """A small synthetic V4.1 text config, as ``to_dict`` emits it."""
    text_config: dict[str, Any] = {
        "vocab_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "head_dim": 512,
        "sliding_window": 16,
        "compress_ratios": [0, 2, 1, 2],
        "kv_source_layer_ids": [1, 2],
        "index_head_dim": 128,
        "engram_num_embeddings": [2],
        "engram_head_dim": 256,
        "dspark_target_layer_ids": [3],
    }
    text_config.update(overrides)
    return {"text_config": text_config}


def _params(config: V41Config | None = None) -> dict[str, KVCacheParams]:
    config = config or V41Config.from_hf(_hf_config())
    params = build_kv_params(
        config,
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [DeviceRef.CPU()],
    )
    return {
        key: child
        for key, child in params.children.items()
        if isinstance(child, KVCacheParams)
    }


def test_record_leaves_match_the_registered_ops() -> None:
    """The window, compressed and index leaves hold exactly the opaque
    ``uint8`` records the V2 store, attention and indexer kernels read."""
    specs = {s.key: s for s in kv_leaf_specs(V41Config.from_hf(_hf_config()))}
    assert specs["swa"].record_bytes == WINDOW_RECORD_BYTES
    assert specs["c2a"].record_bytes == COMPRESSED_RECORD_BYTES
    assert specs["c1a"].record_bytes == COMPRESSED_RECORD_BYTES
    assert specs["idx_c2a"].record_bytes == v2_kv_leaves.MXFP4_INDEX_KEY_BYTES
    for key in ("swa", "c2a", "c1a", "idx_c2a", "idx_c1a"):
        assert specs[key].dtype == DType.uint8, key
        assert specs[key].row_bytes == specs[key].record_bytes, key


@pytest.mark.parametrize("ratio", [1, 2])
def test_both_compression_ratios_pool_one_record_per_group(
    ratio: int,
) -> None:
    """A compressed leaf holds one record per ``ratio`` tokens and its
    index-key leaf one key per group, at both supported ratios."""
    specs = {s.key: s for s in kv_leaf_specs(V41Config.from_hf(_hf_config()))}
    comp, idx = specs[f"c{ratio}a"], specs[f"idx_c{ratio}a"]
    assert comp.slots_per_page == idx.slots_per_page == PAGE_SIZE // ratio
    params = _params()
    assert params[f"c{ratio}a"].slots_per_page == PAGE_SIZE // ratio
    assert params[f"idx_c{ratio}a"].slots_per_page == PAGE_SIZE // ratio
    # The other ratio's leaves are untouched by this one.
    assert params["swa"].slots_per_page == PAGE_SIZE


@pytest.mark.parametrize(
    ("key", "data"),
    [
        ("swa", WINDOW_DATA_BYTES),
        ("c2a", COMPRESSED_DATA_BYTES),
        ("c1a", COMPRESSED_DATA_BYTES),
    ],
)
def test_record_pages_are_readably_laid_out(key: str, data: int) -> None:
    """Each layer's share of a record leaf page is exactly the packed page
    the CSA2 ops address: a whole number of the records' value rows, and
    16-byte aligned."""
    specs = {s.key: s for s in kv_leaf_specs(V41Config.from_hf(_hf_config()))}
    spec = specs[key]
    slots, record = spec.slots_per_page, spec.record_bytes
    assert record is not None
    assert cache_page_bytes(slots, record, data) == slots * record
    page = slots * record
    assert page % 16 == 0
    assert page % data == 0
    # The index-key pages need only the stride alignment the indexer ops
    # take; check the same invariant for them at both ratios.
    for idx_spec in (specs["idx_c2a"], specs["idx_c1a"]):
        assert idx_spec.record_bytes is not None
        idx_page = idx_spec.slots_per_page * idx_spec.record_bytes
        assert idx_page % v2_kv_leaves.INDEX_KEY_STRIDE_ALIGNMENT == 0


def test_built_blocks_stay_within_page_and_layer_bounds() -> None:
    """The built params allocate exactly one record (or one row) per slot of
    each of a leaf's layers: no slack rows, no missing layers."""
    specs = {s.key: s for s in kv_leaf_specs(V41Config.from_hf(_hf_config()))}
    params = _params()
    for key, spec in specs.items():
        built = params[key]
        assert built.shape_per_block == [
            1,
            spec.num_layers,
            spec.slots_per_page,
            1,
            spec.row_bytes,
        ], key
        assert built.bytes_per_block == (
            spec.num_layers
            * spec.slots_per_page
            * spec.row_bytes
            * built.dtype.size_in_bytes
        ), key
        assert built.page_size == PAGE_SIZE, key
        assert built.num_layers == spec.num_layers, key


def test_ratio2_state_is_a_scratch_ring() -> None:
    """The ratio-2 compressor carry is a per-request scratch ring of raw
    float32 rows -- a token's KV row then its score row -- two rows while
    no layer runs speculative decoding, not a token-paged float8 leaf."""
    specs = {s.key: s for s in kv_leaf_specs(V41Config.from_hf(_hf_config()))}
    state = specs["c2a_state"]
    assert state.dtype == DType.float32
    assert state.row_bytes == 2 * 512
    assert state.slots_per_page == v2_kv_leaves.state_ring_rows(0)
    assert state.stride_alignment == v2_kv_leaves.STATE_STRIDE_ALIGNMENT
    built = _params()["c2a_state"]
    assert built.scratch
    assert built.dtype == DType.float32
    assert built.head_dim == 2 * 512
    assert built.window_size is None


def test_only_the_csa2_leaves_stay_packed() -> None:
    """The CSA2 attention raises on strided blocks; the indexer and
    compressor take a stride, so their leaves declare one."""
    params = _params()
    for key, leaf in params.items():
        if key in ("swa", "c2a", "c1a"):
            assert leaf.stride_alignment is None, key
        else:
            assert leaf.stride_alignment == 16, key


def test_sliding_window_is_honored() -> None:
    """The window leaf's size follows the config's sliding window, not a
    constant."""
    config = V41Config.from_hf(_hf_config(sliding_window=256))
    specs = {s.key: s for s in kv_leaf_specs(config)}
    assert specs["swa"].window_size == 256
    assert _params(config)["swa"].window_size == 256


def test_index_only_layers_allocate_no_k_planes() -> None:
    """Cache ownership follows ``kv_source_layer_ids``: a layer whose ratio
    is set but that is not a KV source (an index-only refresher) reads its
    source's leaves and allocates no compressed or index-key planes of its
    own."""
    config = V41Config.from_hf(_hf_config())
    # Layer 3 carries ratio 2 but only refreshes the indexer; layers 1 and
    # 2 are the ratio-2 and ratio-1 KV source owners.
    assert config.ratio(3) == 2
    assert 3 not in config.kv_source_layer_ids
    specs = {s.key: s for s in kv_leaf_specs(config)}
    assert specs["c2a"].num_layers == 1
    assert specs["idx_c2a"].num_layers == 1
    assert specs["c2a_state"].num_layers == 1
    params = _params(config)
    assert params["c2a"].num_layers == 1
    assert params["idx_c2a"].num_layers == 1


def test_cache_keys_are_preserved() -> None:
    """The interface's published leaf keys and their order are unchanged."""
    assert list(_params()) == [
        "swa",
        "c2a",
        "c2a_state",
        "idx_c2a",
        "c1a",
        "idx_c1a",
    ]
