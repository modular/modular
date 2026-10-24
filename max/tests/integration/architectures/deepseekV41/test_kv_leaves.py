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
"""Tests building the KV cache parameters from the real config."""

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.deepseekV41 import kv_leaves
from max.pipelines.architectures.deepseekV41.layers.csa2 import (
    COMPRESSED_DATA_BYTES,
    COMPRESSED_RECORD_BYTES,
    WINDOW_DATA_BYTES,
    WINDOW_RECORD_BYTES,
    cache_page_bytes,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative.config import SpeculativeConfig
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"

_LEAF_KEYS = ("swa", "c2a", "c2a_state", "idx_c2a", "c1a", "idx_c1a")


def _real_text_config() -> AutoConfig:
    config = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    return DeepseekV41Config.get_text_config(config)


def _mock_pipeline_config(
    *, speculative: object = None, data_parallel_degree: int = 1
) -> Mock:
    model = Mock()
    model.data_parallel_degree = data_parallel_degree
    pipeline_config = Mock()
    pipeline_config.model = model
    pipeline_config.speculative = speculative
    return pipeline_config


def _build_params(
    cache_dtype: DType = DType.bfloat16,
    *,
    devices: Sequence[DeviceRef] = (DeviceRef.CPU(),),
    csa_records: bool = False,
) -> MultiKVCacheParams:
    params = kv_leaves.construct_kv_params(
        huggingface_config=_real_text_config(),
        pipeline_config=_mock_pipeline_config(),
        devices=list(devices),
        kv_cache_config=KVCacheConfig(),
        cache_dtype=cache_dtype,
        csa_records=csa_records,
    )
    assert isinstance(params, MultiKVCacheParams)
    return params


def _leaf(params: MultiKVCacheParams, key: str) -> KVCacheParams:
    leaf = params.children[key]
    assert isinstance(leaf, KVCacheParams)
    return leaf


def test_construct_kv_params_succeeds_on_the_real_config() -> None:
    params = _build_params()
    assert set(params.children) == set(_LEAF_KEYS)
    for key in _LEAF_KEYS:
        _leaf(params, key)


def test_construct_kv_params_succeeds_with_four_gpu_devices() -> None:
    """Every leaf carries all four devices."""
    devices = [DeviceRef("gpu", i) for i in range(4)]
    params = _build_params(devices=devices)
    assert set(params.children) == set(_LEAF_KEYS)
    for key in _LEAF_KEYS:
        assert list(_leaf(params, key).devices) == devices


def test_state_leaf_is_a_per_request_float32_ring() -> None:
    """``c2a_state`` is a scratch block a request of ``ring_rows`` slots,
    each a token's float32 KV then score row."""
    state = _leaf(_build_params(), "c2a_state")
    head_dim = _real_text_config().head_dim
    assert state.dtype == DType.float32
    assert state.kv_dim == 1
    assert state.scratch
    assert state.shape_per_block == [
        1,
        state.num_layers,
        kv_leaves.state_ring_rows(0),
        1,
        2 * head_dim,
    ]
    assert state.bytes_per_block == (
        state.num_layers * kv_leaves.state_ring_rows(0) * 2 * head_dim * 4
    )


@pytest.mark.parametrize(
    ("drafts", "rows"),
    [(0, 2), (1, 4), (2, 4), (3, 8), (5, 8), (6, 8), (7, 16)],
)
def test_state_ring_rows(drafts: int, rows: int) -> None:
    """The ring holds a verify step's ``drafts + 1`` rows with a row of
    margin, in a power of two, and never fewer than a pair."""
    assert kv_leaves.state_ring_rows(drafts) == rows
    assert rows >= drafts + 2


def _spec_params(num_speculative_tokens: int | None) -> object:
    speculative = SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=num_speculative_tokens,
    )
    return kv_leaves.construct_kv_params(
        huggingface_config=_real_text_config(),
        pipeline_config=_mock_pipeline_config(speculative=speculative),
        devices=[DeviceRef.CPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
    )


def test_state_ring_follows_the_speculative_draft_count() -> None:
    params = _spec_params(5)
    assert isinstance(params, MultiKVCacheParams)
    assert _leaf(params, "c2a_state").slots_per_page == 8


def test_unresolved_draft_count_is_rejected() -> None:
    """An unset width would size a two-row ring that verify steps overrun."""
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        _spec_params(None)


def _records_params(
    csa_records: bool, devices: Sequence[DeviceRef] = (DeviceRef.CPU(),)
) -> tuple[list[kv_leaves.V41KVLeafSpec], MultiKVCacheParams]:
    specs = kv_leaves.kv_leaf_specs(
        _real_text_config(), csa_records=csa_records
    )
    params = kv_leaves.build_kv_params(
        specs,
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
        devices=list(devices),
        num_q_heads=64,
    )
    return specs, params


def test_default_leaves_store_csa_records() -> None:
    """By default, the window and compressed leaves hold CSA records, the
    only layout the attention reads."""
    assert kv_leaves.CSA_KV_RECORDS
    for spec in kv_leaves.kv_leaf_specs(_real_text_config()):
        if spec.kind in ("swa", "comp"):
            assert spec.record_bytes is not None


def test_bf16_leaves_store_rows() -> None:
    """Without records, the window and compressed leaves keep one bf16 row
    per raw token."""
    params = _build_params()
    for key in ("swa", "c2a", "c1a"):
        leaf = _leaf(params, key)
        assert leaf.dtype == DType.bfloat16
        assert leaf.head_dim == 512
        assert leaf.slots_per_page == 128


def test_switch_selects_record_leaves() -> None:
    params = _build_params(csa_records=True)
    assert _leaf(params, "swa").dtype == DType.uint8
    assert _leaf(params, "c2a").slots_per_page == 64


@pytest.mark.parametrize(
    ("key", "slots", "record", "data"),
    [
        ("swa", 128, WINDOW_RECORD_BYTES, 512),
        ("c2a", 64, COMPRESSED_RECORD_BYTES, 256),
        ("c1a", 128, COMPRESSED_RECORD_BYTES, 256),
    ],
)
def test_record_leaf_pages_hold_the_attention_cache_page(
    key: str, slots: int, record: int, data: int
) -> None:
    """Each layer's share of a record leaf page is exactly the packed page
    the CSA2 ops address, ``slots`` records (one per token or compression
    group) of ``record`` bytes."""
    specs, params = _records_params(True)
    spec = next(s for s in specs if s.key == key)
    leaf = _leaf(params, key)
    assert spec.record_bytes == record
    assert leaf.dtype == DType.uint8
    assert leaf.head_dim == record
    assert leaf.slots_per_page == slots
    assert cache_page_bytes(slots, record, data) == slots * record
    assert leaf.bytes_per_block == spec.num_layers * slots * record
    assert leaf.bytes_per_block == spec.page_bytes(128, DType.bfloat16)


@pytest.mark.parametrize("page_size", [128, 256, 512])
def test_record_leaf_pages_are_readable_at_every_page_size(
    page_size: int,
) -> None:
    """At every page size the KV cache accepts, a multiple of 128, a layer's
    page of a record leaf is one the CSA2 ops read: 16-byte aligned and a
    whole number of the records' value rows."""
    for spec in kv_leaves.kv_leaf_specs(_real_text_config()):
        if spec.kind not in ("swa", "comp"):
            continue
        assert spec.record_bytes is not None
        slots = spec.device_slots_per_page(page_size)
        page = slots * spec.record_bytes
        data = (
            WINDOW_DATA_BYTES if spec.kind == "swa" else COMPRESSED_DATA_BYTES
        )
        assert page % 16 == 0, (spec.key, page_size, page)
        assert page % data == 0, (spec.key, page_size, page)


def test_record_leaves_leave_the_other_leaves_unchanged() -> None:
    old_specs, old = _records_params(False)
    _, new = _records_params(True)
    for spec in old_specs:
        if spec.kind in ("state", "idx_comp"):
            assert (
                _leaf(new, spec.key).bytes_per_block
                == _leaf(old, spec.key).bytes_per_block
            )


def test_record_leaves_build_on_four_devices() -> None:
    devices = [DeviceRef("gpu", i) for i in range(4)]
    _, params = _records_params(True, devices)
    for key in _LEAF_KEYS:
        assert list(_leaf(params, key).devices) == devices


def test_only_csa2_leaves_stay_packed() -> None:
    """The CSA2 ops raise on strided blocks; the indexer and compressor
    take a stride."""
    _, params = _records_params(True)
    for leaf_id, leaf in params.leaves().items():
        key = leaf_id.split(".")[0]
        if key in ("swa", "c2a", "c1a"):
            assert leaf.stride_alignment is None, key
        else:
            assert leaf.stride_alignment == 16, key
