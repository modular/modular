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
"""Tests the DeepSeek-V4.1-Flash ModuleV3 config, KV and input interface."""

from __future__ import annotations

from typing import Any

import pytest
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.kv_cache import MLAKVCacheParams
from max.pipelines.architectures.deepseekV41.kv_leaves import (
    MXFP4_INDEX_KEY_BYTES,
    state_ring_rows,
)
from max.pipelines.architectures.deepseekV41.layers.csa2 import (
    COMPRESSED_RECORD_BYTES,
    WINDOW_RECORD_BYTES,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
    extra_input_names,
    extra_inputs,
    kv_leaf_specs,
)
from max.pipelines.lib import KVCacheConfig


def _hf_config(**overrides: Any) -> dict[str, Any]:
    """A small synthetic V4.1 config mapping, ``to_dict`` shaped.

    ``head_dim`` and ``index_head_dim`` keep the checkpoint's packed widths
    (512 and 128 channels) so the record byte counts line up with them.
    """
    text_config: dict[str, Any] = {
        "vocab_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "head_dim": 512,
        "sliding_window": 16,
        # Layer 0 is SWA-only, 1 a ratio-2 KV source, 2 a ratio-1 KV
        # source, 3 a ratio-2 layer outside the KV-source set (its keys
        # live in no leaf).
        "compress_ratios": [0, 2, 1, 2],
        "kv_source_layer_ids": [1, 2],
        "index_head_dim": 128,
        "engram_num_embeddings": [2],
        "engram_head_dim": 256,
        "dspark_target_layer_ids": [3],
    }
    text_config.update(overrides)
    return {"text_config": text_config}


def test_from_hf_reads_text_config() -> None:
    config = V41Config.from_hf(_hf_config())
    assert config.vocab_size == 128
    assert config.num_hidden_layers == 4
    assert config.head_dim == 512
    assert config.compress_ratios == (0, 2, 1, 2)
    assert config.kv_source_layer_ids == (1, 2)
    assert config.engram_num_embeddings == (2,)
    assert config.engram_head_dim == 256
    # The draft-stage keys resolve the way V2's ``num_dspark_stages``
    # resolves them: ``n_mtp_layers`` first, then
    # ``num_nextn_predict_layers``, then the default. A checkpoint without
    # them still resolves.
    assert config.num_dspark_stages == 3
    assert (
        V41Config.from_hf(
            _hf_config(num_nextn_predict_layers=2)
        ).num_dspark_stages
        == 2
    )
    assert (
        V41Config.from_hf(
            _hf_config(n_mtp_layers=1, num_nextn_predict_layers=2)
        ).num_dspark_stages
        == 1
    )


def test_kv_leaf_geometry() -> None:
    config = V41Config.from_hf(_hf_config())
    specs = {spec.key: spec for spec in kv_leaf_specs(config)}
    assert list(specs) == [
        "swa",
        "c2a",
        "c2a_state",
        "idx_c2a",
        "c1a",
        "idx_c1a",
    ]
    # DSpark draft stages share the window leaf: three by default, the
    # count V2 resolves for a config that names none.
    assert specs["swa"].num_layers == 7
    assert specs["swa"].window_size == 16
    assert specs["swa"].slots_per_page == PAGE_SIZE
    # The window leaf holds one opaque MXFP8 record per token, the width
    # the registered V2 CSA2 kernels read.
    assert specs["swa"].row_bytes == WINDOW_RECORD_BYTES
    assert specs["swa"].dtype == DType.uint8
    # One ratio-2 and one ratio-1 KV source layer; their compressed leaves
    # hold one NVFP4 record per closed compression group.
    assert specs["c2a"].num_layers == 1
    assert specs["c2a"].slots_per_page == PAGE_SIZE // 2
    assert specs["c2a"].row_bytes == COMPRESSED_RECORD_BYTES
    assert specs["c2a"].dtype == DType.uint8
    # The index-key leaves hold MXFP4 keys the indexer kernels address.
    assert specs["idx_c2a"].row_bytes == MXFP4_INDEX_KEY_BYTES
    assert specs["idx_c2a"].stride_alignment is not None
    assert specs["c1a"].slots_per_page == PAGE_SIZE
    # The ratio-2 compressor state: a per-request scratch ring of float32
    # rows, each wide enough for the pair's KV row then its score row.
    assert specs["c2a_state"].ring_rows == state_ring_rows(0)
    assert specs["c2a_state"].slots_per_page == state_ring_rows(0)
    assert specs["c2a_state"].row_bytes == 2 * config.head_dim
    assert specs["c2a_state"].dtype == DType.float32
    assert specs["c2a_state"].stride_alignment is not None


def test_swa_leaf_counts_dspark_stages_not_taps() -> None:
    """Sizes the window leaf by the DSpark stage count, not the tap list.

    The two counts are different quantities that only coincide on the
    published checkpoint (three taps, three stages); V2 sizes the same
    leaf by the stage count, so this interface must too.
    """
    config = V41Config.from_hf(
        _hf_config(
            dspark_target_layer_ids=[3, 2, 1],
            num_nextn_predict_layers=2,
        )
    )
    specs = {spec.key: spec for spec in kv_leaf_specs(config)}
    assert specs["swa"].num_layers == config.num_hidden_layers + 2


def test_build_kv_params_matches_leaf_geometry() -> None:
    config = V41Config.from_hf(_hf_config())
    params = build_kv_params(
        config,
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [DeviceRef.CPU()],
    )
    leaves = {spec.key: spec for spec in kv_leaf_specs(config)}
    assert set(params.children) == set(leaves)
    for key, leaf in leaves.items():
        built = params.children[key]
        # Every leaf is MLA-shaped (``kv_dim == 1``), so none asks for
        # head replication.
        assert isinstance(built, MLAKVCacheParams)
        assert built.dtype == leaf.dtype
        assert built.head_dim == leaf.row_bytes
        assert built.num_layers == leaf.num_layers
        assert built.window_size == leaf.window_size
        # The logical ratio must hold: one slot per `ratio` tokens of a
        # PAGE_SIZE-token page.
        assert built.page_size == PAGE_SIZE
        assert built.slots_per_page == leaf.slots_per_page
        # Only the state ring is scratch, and only the record leaves the
        # pool may pad.
        assert built.scratch == (leaf.ring_rows is not None)
        assert built.stride_alignment == leaf.stride_alignment


def test_build_kv_params_rejects_other_page_sizes() -> None:
    config = V41Config.from_hf(_hf_config())
    with pytest.raises(ValueError, match="page size"):
        build_kv_params(
            config,
            KVCacheConfig(kv_cache_page_size=256),
            [DeviceRef.CPU()],
        )


def test_extra_input_names() -> None:
    config = V41Config.from_hf(_hf_config())
    assert extra_input_names(config) == [
        "swa_begin",
        "global_write_floor",
        "image_token_mask",
        "engram_lookback",
        "engram_lookback_dead",
        "engram_table_0_weight",
        "engram_table_0_scale",
    ]


def test_engram_tables_size_from_config_head_dim() -> None:
    """Declares the Engram tables with the checkpoint's ``engram_head_dim``.

    V2 sizes the same tables from that config field, so a checkpoint with
    a head dim other than the published 256 must flow into the shapes
    and dtypes declared here.
    """
    config = V41Config.from_hf(
        _hf_config(engram_head_dim=192, engram_num_embeddings=[3, 5])
    )
    declared = dict(extra_inputs(config, [DeviceRef.GPU(0)]))
    assert declared["engram_table_0_weight"] == TensorType(
        DType.float8_e4m3fn, [3, 192], DeviceRef.CPU()
    )
    assert declared["engram_table_1_scale"] == TensorType(
        DType.float8_e8m0fnu, [5, 192 // 32], DeviceRef.CPU()
    )
