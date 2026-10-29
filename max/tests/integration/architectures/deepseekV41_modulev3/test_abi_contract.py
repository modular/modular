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
"""Tests the V4.1 stub's input ABI against the recorded contract.

Mach binds every graph input by position and the sliced fixtures append to it,
so the order, dtypes, shapes and devices of the inputs are frozen in
``testdata/abi_contract.json``. Each layout is staged afresh on SM100 virtual
devices and must equal the record.
"""

from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

import abi_contract
import pytest
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.legacy.kv_cache import MLAKVCacheParams
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    STATE_RING_ROWS,
    V41Config,
    build_kv_params,
    extra_inputs,
    kv_cache_config,
    kv_leaf_specs,
)
from max.pipelines.lib import KVCacheConfig

_INPUT_COUNTS = {1: 62, 2: 112, 4: 210, 8: 406}
_KV_INPUTS_PER_DEVICE = 48
_EXTRAS: list[tuple[str, str, list[int | str | None], str]] = [
    ("swa_begin", "DType.int32", ["replica_0_batch_size"], "gpu:0"),
    ("global_write_floor", "DType.int32", ["replica_0_batch_size"], "gpu:0"),
    ("image_token_mask", "DType.uint8", ["total_seq_len"], "gpu:0"),
    ("engram_lookback", "DType.int32", ["replica_0_batch_size", 3], "gpu:0"),
    (
        "engram_lookback_dead",
        "DType.uint8",
        ["replica_0_batch_size", 3],
        "gpu:0",
    ),
    ("engram_table_0_weight", "DType.float8_e4m3fn", [None, 256], "cpu"),
    ("engram_table_0_scale", "DType.float8_e8m0fnu", [None, 8], "cpu"),
    ("engram_table_1_weight", "DType.float8_e4m3fn", [None, 256], "cpu"),
    ("engram_table_1_scale", "DType.float8_e8m0fnu", [None, 8], "cpu"),
    ("engram_token_map", "DType.int32", [129280], "gpu:0"),
    ("decoder_rows", "DType.uint8", ["decoder_rows"], "gpu:0"),
]
# Page sizes of the KV leaves, bytes per page per device on the full model:
# packed rows of the record's raw payload -- 528 bytes (window), 288
# (compressed) and 68 (index) -- with no alignment slack; the state ring is
# eight rows of one 1024-element float32 pair row per layer.
_BYTES_PER_PAGE = {
    "swa": 2_906_112,
    "c2a": 55_296,
    "c2a_state": 98_304,
    "idx_c2a": 13_056,
    "c1a": 36_864,
    "idx_c1a": 8_704,
}


@cache
def _recorded() -> dict[str, list[dict[str, Any]]]:
    return json.loads(
        (abi_contract.testdata_dir() / "abi_contract.json").read_text()
    )["layouts"]


@cache
def _staged(model: str, tp: int) -> tuple[list[dict[str, Any]], list[str]]:
    """The contract and the subgraph names of ``model`` staged at ``tp``."""
    with abi_contract.virtual_devices(tp):
        staged, config, kv_params = abi_contract.stage_stub(model, tp)
        records = abi_contract.contract(staged, config, kv_params, tp)
    bodies = re.findall(r"mo.graph @(\w+)", str(staged))
    return records, bodies[1:]


_LAYOUTS = [
    (m, n) for m in abi_contract.MODELS for n in abi_contract.TP_DEGREES
]


@pytest.mark.parametrize(("model", "tp"), _LAYOUTS)
def test_staged_inputs_equal_the_recorded_contract(model: str, tp: int) -> None:
    assert _staged(model, tp)[0] == _recorded()[f"{model}_tp{tp}"]


@pytest.mark.parametrize(("model", "tp"), _LAYOUTS)
def test_input_order_and_counts(model: str, tp: int) -> None:
    records = _recorded()[f"{model}_tp{tp}"]
    n_kv = _KV_INPUTS_PER_DEVICE * tp
    assert len(records) == _INPUT_COUNTS[tp]
    assert [r["name"] for r in records[:3]] == [
        "tokens",
        "return_n_logits",
        "input_row_offsets",
    ]
    assert all(r["name"].startswith("kv.") for r in records[3 : 3 + n_kv])
    extras = records[3 + n_kv : 3 + n_kv + len(_EXTRAS)]
    assert len(extras) == len(_EXTRAS)
    for record, (name, dtype, shape, device) in zip(
        extras, _EXTRAS, strict=True
    ):
        assert record["name"] == name
        assert record["dtype"] == dtype
        assert record["device"] == device
        assert all(
            want is None or want == got
            for want, got in zip(shape, record["symbolic_shape"], strict=True)
        )
    # Signal buffers come last, one uint8 buffer per device, and only when the
    # graph spans several.
    signals = records[3 + n_kv + len(_EXTRAS) :]
    assert len(signals) == (tp if tp > 1 else 0)
    for i, record in enumerate(signals):
        assert record["name"] == f"signal_buffer.{i}"
        assert (record["kind"], record["dtype"], record["device"]) == (
            "buffer",
            "DType.uint8",
            f"gpu:{i}",
        )


@pytest.mark.parametrize(("model", "tp"), _LAYOUTS)
def test_each_engram_table_is_one_cpu_input(model: str, tp: int) -> None:
    tables = [
        r
        for r in _recorded()[f"{model}_tp{tp}"]
        if r["name"].startswith("engram_table_")
    ]
    assert len(tables) == 4
    assert {r["device"] for r in tables} == {"cpu"}
    assert len({r["name"] for r in tables}) == 4


@pytest.mark.parametrize(("model", "tp"), [("f1", 2), ("full", 8)])
def test_layers_share_seven_subgraphs(model: str, tp: int) -> None:
    assert sorted(_staged(model, tp)[1]) == sorted(
        [
            "swa_only",
            "full_r2",
            "reuse_r2",
            "full_r1_cand",
            "reindex_r1",
            "reuse_r1",
            "engram",
        ]
    )


def test_kv_leaves_have_the_frozen_geometry() -> None:
    config = V41Config.from_hf(abi_contract.hf_config("full"))
    # One device, so that a block's bytes are one device's.
    params = build_kv_params(config, kv_cache_config(), [DeviceRef.GPU(0)])
    assert [s.key for s in kv_leaf_specs(config)] == list(_BYTES_PER_PAGE)
    assert {
        k: p.bytes_per_block for k, p in params.children.items()
    } == _BYTES_PER_PAGE


def test_build_kv_params_rejects_other_page_sizes() -> None:
    # The interface fixes the page size the leaves' slots-per-page ratios
    # are recorded under; a boot flag naming any other page must be refused
    # before a pool is ever planned.
    config = V41Config.from_hf(abi_contract.hf_config("full"))
    wrong_page = KVCacheConfig(kv_cache_page_size=64)
    with pytest.raises(ValueError, match="page size of 128"):
        build_kv_params(config, wrong_page, [DeviceRef.GPU(0)])


def test_build_kv_params_plans_draft_rows_on_every_leaf() -> None:
    """A speculative config sizes every leaf's pool for the draft rows."""
    config = V41Config.from_hf(abi_contract.hf_config("full"))
    params = build_kv_params(
        config,
        kv_cache_config(),
        [DeviceRef.CPU()],
        speculative_method="dspark",
        num_draft_tokens=5,
    )
    for child in params.children.values():
        assert child.num_draft_tokens == 5
        assert child.speculative_method == "dspark"
    # The speculative width plans the pool, never the leaf geometry: the
    # state ring keeps the fixed depth both boot modes share.
    state = params.children["c2a_state"]
    assert isinstance(state, MLAKVCacheParams)
    assert state.slots_per_page == STATE_RING_ROWS


def test_build_kv_params_refuses_a_depth_beyond_the_ring() -> None:
    """A verify depth deeper than the ring's fixed rows corrupts it."""
    config = V41Config.from_hf(abi_contract.hf_config("full"))
    with pytest.raises(ValueError, match="compressor-state ring"):
        build_kv_params(
            config,
            kv_cache_config(),
            [DeviceRef.CPU()],
            speculative_method="dspark",
            num_draft_tokens=9,
        )


def test_window_leaf_counts_dspark_stages_not_taps() -> None:
    """Sizes the window leaf by the DSpark stage count, not the tap list.

    The two counts are different quantities that only coincide on the
    published checkpoint (three taps, three stages); V2 sizes the same
    leaf by the stage count, so the ABI must resolve the same depth.
    """
    config = V41Config.from_hf(abi_contract.hf_config("full"))
    assert config.num_dspark_stages == 3
    specs = {spec.key: spec for spec in kv_leaf_specs(config)}
    assert specs["swa"].num_layers == config.num_hidden_layers + 3

    shallow = json.loads(
        (abi_contract.testdata_dir() / "config_full.json").read_text()
    )
    shallow["text_config"]["num_nextn_predict_layers"] = 1
    config = V41Config.from_hf(shallow)
    assert config.num_dspark_stages == 1
    assert kv_leaf_specs(config)[0].num_layers == config.num_hidden_layers + 1


def test_engram_tables_size_from_config_head_dim() -> None:
    """Declares the Engram tables with the checkpoint's ``engram_head_dim``.

    V2 sizes the same tables from that config field, so a checkpoint
    with a head dim other than the published 256 must flow into the
    shapes and dtypes declared here.
    """
    variant = json.loads(
        (abi_contract.testdata_dir() / "config_full.json").read_text()
    )
    variant["text_config"]["engram_head_dim"] = 192
    variant["text_config"]["engram_num_embeddings"] = [3, 5]
    declared = dict(
        extra_inputs(V41Config.from_hf(variant), [DeviceRef.GPU(0)])
    )
    assert declared["engram_table_0_weight"] == TensorType(
        DType.float8_e4m3fn, [3, 192], DeviceRef.CPU()
    )
    assert declared["engram_table_1_scale"] == TensorType(
        DType.float8_e8m0fnu, [5, 192 // 32], DeviceRef.CPU()
    )
