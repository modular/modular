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
"""Tests loading Engram tables stored MXFP4-packed.

A checkpoint may store each Engram weight table as packed E2M1 pairs (uint8
``[rows, 128]``, low nibble first) beside the UE8M0 scales an FP8 table has,
and declare it with ``quantization_config.engram_embed_*``. The loader unpacks
the codes to E4M3 as it pins them. Every E2M1 value is exact in E4M3, so the
pinned bytes must equal an FP8 table of the same values bit for bit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.deepseekV41_modulev3 import model as model_mod
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    EngramTableFormat,
)
from max.pipelines.architectures.deepseekV41_modulev3.weight_adapters import (
    _engram_tables,
    convert_safetensor_state_dict,
    unpack_e2m1_to_e4m3,
)
from transformers import PretrainedConfig

# Not a multiple of the rows per pinning chunk the tests set.
ROWS = 37
DIM = 256
SCALES = DIM // 32
LAYER = 1
# E2M1 magnitudes by 3-bit code; bit 3 is the sign.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)

MXFP4_EXPERTS_QUANTIZATION_CONFIG = {
    "quant_method": "fp8",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}
# What the MXFP4-table checkpoint's quantization_config adds.
ENGRAM_EMBED_KEYS = {
    "engram_embed_quant": "mxfp4",
    "engram_embed_block_size": 32,
    "engram_embed_scale_dtype": "ue8m0",
    "engram_embed_layers": [LAYER],
}


def _e4m3_value(code: int) -> float:
    """Decodes one ``float8_e4m3fn`` byte, independently of the loader."""
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 0xF, code & 0x7
    if exponent == 0:
        return sign * mantissa / 8 * 2.0**-6
    return sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)


def _e4m3_codes(nibbles: np.ndarray) -> np.ndarray:
    """Each E2M1 code's E4M3 byte, found among all 256 decoded bytes."""
    table = np.empty(16, np.uint8)
    for code in range(16):
        value = -E2M1[code & 7] if code & 8 else E2M1[code & 7]
        (byte,) = [
            b
            for b in range(256)
            if b not in (0x7F, 0xFF)
            and _e4m3_value(b) == value
            and (b >> 7) == (code >> 3)
        ]
        table[code] = byte
    return table[nibbles]


def _tables(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The same weight table, MXFP4-packed and as FP8 codes."""
    rng = np.random.default_rng(seed)
    nibbles = rng.integers(0, 16, (ROWS, DIM), dtype=np.uint8)
    # Every code at both an even (low nibble) and an odd column.
    nibbles[0] = np.arange(DIM) % 16
    nibbles[1] = np.arange(DIM) // 16
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return packed, _e4m3_codes(nibbles)


def _wd(array: np.ndarray, name: str, dtype: DType) -> WeightData:
    """``array`` as a checkpoint tensor that shares its memory."""
    buffer = Buffer.from_numpy(array).view(dtype)
    return WeightData(buffer, name, dtype, Shape(list(array.shape)))


def _state(weight: np.ndarray, dtype: DType) -> dict[str, WeightData]:
    prefix = f"layers.{LAYER}.engram.embed"
    scales = np.full((len(weight), SCALES), 127, np.uint8)
    return {
        f"{prefix}.weight": _wd(weight, f"{prefix}.weight", dtype),
        f"{prefix}.scale": _wd(scales, f"{prefix}.scale", DType.float8_e8m0fnu),
    }


def test_unpack_matches_the_fp8_codes() -> None:
    packed, fp8 = _tables()
    out = np.zeros_like(fp8)
    unpack_e2m1_to_e4m3(packed, out)
    np.testing.assert_array_equal(out, fp8)


def test_packed_table_is_a_view_of_the_checkpoint() -> None:
    packed, _ = _tables()
    (table,) = _engram_tables(_state(packed, DType.uint8), (LAYER,))
    assert table.packed
    assert table.rows == ROWS
    assert table.weight_codes.shape == (ROWS, DIM // 2)
    # Never a heap copy: at production size each table is ~49 GB packed.
    assert np.shares_memory(table.weight_codes, packed)
    (fp8_table,) = _engram_tables(
        _state(_tables()[1], DType.float8_e4m3fn), (LAYER,)
    )
    assert not fp8_table.packed


@pytest.mark.parametrize(
    ("width", "dtype"),
    [(DIM // 2, DType.float8_e4m3fn), (DIM // 4, DType.uint8)],
)
def test_other_weight_widths_are_refused(width: int, dtype: DType) -> None:
    weight = np.zeros((ROWS, width), np.uint8)
    with pytest.raises(ValueError, match=r"engram\.embed\.weight"):
        _engram_tables(_state(weight, dtype), (LAYER,))


def test_pinned_packed_table_matches_fp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Four rows a chunk: ragged against ROWS, so the tail chunk is short.
    monkeypatch.setattr(model_mod, "_PIN_CHUNK_BYTES", 4 * DIM)
    packed, fp8 = _tables()
    got = np.zeros((ROWS, DIM), np.uint8)
    model_mod._fill_pinned(packed, got)
    np.testing.assert_array_equal(got, fp8)
    copied = np.zeros((ROWS, DIM), np.uint8)
    model_mod._fill_pinned(fp8, copied)
    np.testing.assert_array_equal(copied, fp8)


def test_packed_digest_is_the_pinned_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload of packed codes dedupes against the unpacked pin."""
    monkeypatch.setattr(model_mod, "_PIN_CHUNK_BYTES", 4 * DIM)
    packed, fp8 = _tables()
    digest = model_mod._table_digest(packed, DIM)
    assert digest == model_mod._table_digest(fp8, DIM)
    changed = packed.copy()
    changed[-1, -1] ^= 0x10
    assert model_mod._table_digest(changed, DIM) != digest


def test_pinning_refuses_a_mismatched_shape() -> None:
    """The shape check runs before any page-locked allocation."""
    packed, _ = _tables()
    device = Mock()
    with pytest.raises(ValueError, match="engram_table_0_scale"):
        model_mod._pinned_table(
            device,
            "engram_table_0_scale",
            ROWS,
            SCALES,
            packed[:, :4],
            DType.float8_e8m0fnu,
        )
    with pytest.raises(ValueError, match="engram_table_0_weight"):
        model_mod._pinned_table(
            device,
            "engram_table_0_weight",
            ROWS,
            DIM,
            packed[:, :64],
            DType.float8_e4m3fn,
        )


@pytest.mark.parametrize(
    ("keys", "want"),
    [
        ({}, EngramTableFormat.FP8),
        (ENGRAM_EMBED_KEYS, EngramTableFormat.MXFP4),
        (
            {**ENGRAM_EMBED_KEYS, "engram_embed_quant": "MXFP4"},
            EngramTableFormat.MXFP4,
        ),
        (
            {**ENGRAM_EMBED_KEYS, "engram_embed_quant": "fp8"},
            EngramTableFormat.FP8,
        ),
    ],
)
def test_engram_embed_keys_resolve(
    keys: Mapping[str, Any], want: EngramTableFormat
) -> None:
    config = {**MXFP4_EXPERTS_QUANTIZATION_CONFIG, **keys}
    assert EngramTableFormat.from_quantization_config(config, [LAYER]) is want


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("engram_embed_quant", "nvfp4"),
        ("engram_embed_block_size", 16),
        ("engram_embed_scale_dtype", "e4m3"),
        ("engram_embed_layers", [LAYER, 14]),
    ],
)
def test_engram_embed_keys_are_checked(field: str, value: object) -> None:
    config = {**ENGRAM_EMBED_KEYS, field: value}
    with pytest.raises(ValueError, match=field):
        EngramTableFormat.from_quantization_config(config, [LAYER])


def _hf_config(engram_keys: Mapping[str, Any]) -> PretrainedConfig:
    return PretrainedConfig(
        text_config=PretrainedConfig(
            num_hidden_layers=1, engram_layer_ids=[LAYER]
        ),
        quantization_config={
            **MXFP4_EXPERTS_QUANTIZATION_CONFIG,
            **engram_keys,
        },
    )


def _checkpoint(state: Mapping[str, WeightData]) -> dict[str, Any]:
    weights: dict[str, Any] = {}
    for name, data in state.items():
        weights[name] = Mock()
        weights[name].data.return_value = data
    return weights


def test_pipeline_adapter_hands_over_the_declared_packed_table() -> None:
    packed, _ = _tables()
    result = convert_safetensor_state_dict(
        _checkpoint(_state(packed, DType.uint8)),
        _hf_config(ENGRAM_EMBED_KEYS),
    )
    (table,) = result.engram_tables
    assert table.packed
    np.testing.assert_array_equal(table.weight_codes, packed)


@pytest.mark.parametrize("declared_mxfp4", [True, False])
def test_pipeline_adapter_refuses_an_undeclared_table_format(
    declared_mxfp4: bool,
) -> None:
    """The memory planner sizes the tables from the declared format."""
    packed, fp8 = _tables()
    state = (
        _state(fp8, DType.float8_e4m3fn)
        if declared_mxfp4
        else _state(packed, DType.uint8)
    )
    with pytest.raises(ValueError, match="Engram tables"):
        convert_safetensor_state_dict(
            _checkpoint(state),
            _hf_config(ENGRAM_EMBED_KEYS if declared_mxfp4 else {}),
        )
