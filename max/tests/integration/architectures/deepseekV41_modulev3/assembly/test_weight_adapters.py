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
"""The ModuleV3 weight adapter's ``wo_a`` expansion, end to end.

The one deliberate dequant in the adapter: each ``attn.wo_a`` FP8 pair
(E4M3 codes plus compact E8M0 block scales) becomes a single bf16 leaf,
the official BF16 baseline the model's grouped projection runs. The
regression covers multiple layers' pairs at once -- the transform must
consume each pair exactly once (the group's two tensors map to one
path, so a per-name walk would raise for the second tensor's
already-consumed partner) -- and pins the expanded keys, shapes and
values against the reference E4M3 x E8M0 arithmetic, plus the no-vision
gate drop on the same pass. The stacked-projection tests cover the
``w13`` fold directly: the TP1 concatenation and its scale-block
placement, the per-rank TP2 interleave each rank's slice loads, the
never-interleaved dense pair, zero-byte stand-ins, and the scale-block
rank-split rejection.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.deepseekV41_modulev3.weight_adapters import (
    _engram_tables,
    convert_safetensor_state_dict,
    convert_weight_data,
    stack_projections,
)
from transformers import PretrainedConfig


def _e4m3_table() -> np.ndarray:
    b = np.arange(256)
    e, m = (b >> 3) & 15, (b & 7).astype(np.float64)
    v = np.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * 2.0 ** (e - 7))
    v[(e == 15) & ((b & 7) == 7)] = np.nan
    return np.where(b & 0x80, -v, v)


_E4M3 = _e4m3_table()


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16 << 16
    return rounded.astype(np.uint32)


def _wd(array: np.ndarray, name: str, dtype: DType) -> WeightData:
    return WeightData(
        Buffer.from_numpy(np.ascontiguousarray(array)),
        name,
        dtype,
        Shape(list(array.shape)),
    )


def _woa_pair(
    layer: int, rows: int, cols: int, seed: int
) -> tuple[dict[str, WeightData], np.ndarray]:
    """One layer's E4M3/E8M0 pair and its expected bf16 expansion."""
    rng = np.random.default_rng(seed)
    weight_u8 = rng.integers(1, 250, size=(rows, cols), dtype=np.uint8)
    # Exponent bytes in a safe E8M0 range (2^-3 .. 2^3).
    scale_u8 = rng.integers(124, 131, size=(rows // 32, cols // 32)).astype(
        np.uint8
    )
    values = _E4M3[weight_u8]
    blocks = np.power(2.0, scale_u8.astype(np.float64) - 127.0).astype(
        np.float32
    )
    expanded = np.repeat(np.repeat(blocks, 32, axis=0), 32, axis=1)[
        :rows, :cols
    ]
    expected_bits = _bf16_bits(values.astype(np.float32) * expanded)
    return {
        f"layers.{layer}.attn.wo_a.weight": _wd(
            weight_u8,
            f"layers.{layer}.attn.wo_a.weight",
            DType.float8_e4m3fn,
        ),
        f"layers.{layer}.attn.wo_a.scale": _wd(
            scale_u8,
            f"layers.{layer}.attn.wo_a.scale",
            DType.float8_e8m0fnu,
        ),
    }, expected_bits


def test_convert_weight_data_expands_woa_pairs() -> None:
    """Two layers' pairs expand once each: keys, shapes and exact values.

    Layer 0 carries a 64x64 pair (four 32x32 blocks), layer 1 a 32x64
    pair (two blocks beside two whole ones) -- the multi-group walk a
    per-name (non-deduplicated) iteration would break on the second
    tensor of the first group.
    """
    state: dict[str, WeightData] = {}
    expected: dict[int, np.ndarray] = {}
    for layer, (rows, cols) in ((0, (64, 64)), (1, (32, 64))):
        pair, bits = _woa_pair(layer, rows, cols, seed=layer + 5)
        state.update(pair)
        expected[layer] = bits
    # A couple of bystander tensors prove the pass keeps them intact.
    state["layers.0.attn_norm.weight"] = _wd(
        np.ones(8, np.float32), "layers.0.attn_norm.weight", DType.float32
    )
    result = convert_weight_data(state, has_vision=False)
    out = result.state_dict
    for layer, (rows, cols) in ((0, (64, 64)), (1, (32, 64))):
        name = f"layers.{layer}.attn.wo_a.weight"
        assert f"layers.{layer}.attn.wo_a.scale" not in out, layer
        leaf = out[name]
        assert leaf.dtype == DType.bfloat16, layer
        assert [int(d) for d in leaf.shape] == [rows, cols], layer
        buffer = (
            leaf.data
            if isinstance(leaf.data, Buffer)
            else Buffer.from_dlpack(leaf.data)
        )
        raw = np.from_dlpack(buffer.view(DType.uint16))
        got = (raw.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(got.view(np.uint32), expected[layer])
    # Bystanders pass through untouched.
    np.testing.assert_array_equal(
        np.from_dlpack(out["layers.0.attn_norm.weight"].data),
        np.ones(8, np.float32),
    )
    # No Engram layers: no tables.
    assert result.engram_tables == []


def test_engram_tables_are_views_of_the_checkpoint() -> None:
    """The adapter hands the loader zero-copy views, never heap copies.

    The loader copies the tables into pinned memory and drops them, so a
    bulk ``bytes`` copy here would be a second full copy of every table --
    ~98 GB per layer at production size -- for the model's lifetime.
    """
    codes = np.arange(3 * 256, dtype=np.uint8).reshape(3, 256)
    scales = np.arange(3 * 8, dtype=np.uint8).reshape(3, 8)
    state = {
        "layers.5.engram.embed.weight": _wd(
            codes,
            "layers.5.engram.embed.weight",
            DType.float8_e4m3fn,
        ),
        "layers.5.engram.embed.scale": _wd(
            scales,
            "layers.5.engram.embed.scale",
            DType.float8_e8m0fnu,
        ),
    }
    tables = _engram_tables(state, (5,))
    assert len(tables) == 1
    table = tables[0]
    assert table.layer == 5
    assert table.rows == 3
    assert np.shares_memory(table.weight_codes, codes)
    assert np.shares_memory(table.scale_codes, scales)
    np.testing.assert_array_equal(table.weight_codes, codes)
    np.testing.assert_array_equal(table.scale_codes, scales)


def test_convert_weight_data_drops_bias_vl_without_vision() -> None:
    """Without ``vision_config`` the gate's vision bias is dropped."""
    state = {
        "layers.0.ffn.gate.weight": _wd(
            np.zeros((4, 8), np.float32),
            "layers.0.ffn.gate.weight",
            DType.float32,
        ),
        "layers.0.ffn.gate.bias": _wd(
            np.zeros(4, np.float32),
            "layers.0.ffn.gate.bias",
            DType.float32,
        ),
        "layers.0.ffn.gate.bias_vl": _wd(
            np.zeros(4, np.float32),
            "layers.0.ffn.gate.bias_vl",
            DType.float32,
        ),
    }
    out = convert_weight_data(state, has_vision=False).state_dict
    assert "layers.0.ffn.gate.bias_vl" not in out
    assert "layers.0.ffn.gate.bias" in out


def test_pipeline_adapter_filters_vision_before_materialization() -> None:
    hf = PretrainedConfig(
        text_config=PretrainedConfig(num_hidden_layers=1, engram_layer_ids=[]),
        vision_config=PretrainedConfig(hidden_size=64),
    )
    keep = Mock()
    keep.data.return_value = _wd(
        np.ones(8, np.float32), "layers.0.attn_norm.weight", DType.float32
    )
    skip = Mock()
    skip.data.side_effect = AssertionError("unsupported tensor materialized")
    adapted = convert_safetensor_state_dict(
        {
            "layers.0.attn_norm.weight": keep,
            "layers.0.ffn.gate.bias_vl": skip,
            "vision.patch_embedding.weight": skip,
            "mtp.0.head.weight": skip,
        },
        hf,
    )
    assert set(adapted) == {"layers.0.attn_norm.weight"}
    skip.data.assert_not_called()
    keep.data.assert_called_once_with()


def _leaf_u8(leaf: WeightData) -> np.ndarray:
    buffer = (
        leaf.data
        if isinstance(leaf.data, Buffer)
        else Buffer.from_dlpack(leaf.data)
    )
    return np.from_dlpack(buffer.view(DType.uint8)).reshape(
        [int(d) for d in leaf.shape]
    )


def _shared_pair(
    prefix: str, seed: int
) -> tuple[dict[str, WeightData], dict[str, np.ndarray]]:
    """One shared-expert ``w1``/``w3`` pair: E4M3 weights, E8M0 scales."""
    rng = np.random.default_rng(seed)
    arrays = {
        "w1.weight": rng.integers(1, 250, size=(64, 32), dtype=np.uint8),
        "w1.scale": rng.integers(124, 131, size=(2, 1)).astype(np.uint8),
        "w3.weight": rng.integers(1, 250, size=(64, 32), dtype=np.uint8),
        "w3.scale": rng.integers(124, 131, size=(2, 1)).astype(np.uint8),
    }
    state = {
        f"{prefix}.{name}": _wd(array, f"{prefix}.{name}", dtype)
        for name, array in arrays.items()
        for dtype in (
            DType.float8_e8m0fnu
            if name.endswith("scale")
            else DType.float8_e4m3fn,
        )
    }
    return state, arrays


def test_convert_weight_data_folds_shared_experts_one_device() -> None:
    """The ``w13`` fold runs on the same pass as the ``wo_a`` expansion.

    One device stacks the pair's own rows in order -- ``[w1; w3]`` -- with
    each scale block of ``w3`` landing at block ``N1 / 32 + i``, exactly
    where its weight rows do, and the parts leave the dict.
    """
    prefix = "layers.2.ffn.shared_experts"
    pair, arrays = _shared_pair(prefix, seed=7)
    out = convert_weight_data(pair, has_vision=False).state_dict
    weight = out[f"{prefix}.w13.weight"]
    assert [int(d) for d in weight.shape] == [128, 32]
    np.testing.assert_array_equal(
        _leaf_u8(weight),
        np.concatenate([arrays["w1.weight"], arrays["w3.weight"]]),
    )
    scale = out[f"{prefix}.w13.scale"]
    assert [int(d) for d in scale.shape] == [4, 1]
    np.testing.assert_array_equal(
        _leaf_u8(scale),
        np.concatenate([arrays["w1.scale"], arrays["w3.scale"]]),
    )
    assert not [name for name in out if ".w1." in name or ".w3." in name]


def test_stack_projections_interleaves_per_rank_two_devices() -> None:
    """Two devices: each rank's contiguous slice holds its two halves.

    The stacked leaf shards along N over the mesh, so a plain
    concatenation would leave rank 0 holding only ``w1``'s rows; the
    interleave places ``[w1_r; w3_r]`` per rank instead, which the generic
    host split loads per rank unchanged.
    """
    prefix = "layers.2.ffn.shared_experts"
    pair, arrays = _shared_pair(prefix, seed=11)
    stack_projections(pair, n_devices=2)
    weight = pair[f"{prefix}.w13.weight"]
    assert [int(d) for d in weight.shape] == [128, 32]
    np.testing.assert_array_equal(
        _leaf_u8(weight),
        np.concatenate(
            [
                arrays["w1.weight"][:32],
                arrays["w3.weight"][:32],
                arrays["w1.weight"][32:],
                arrays["w3.weight"][32:],
            ]
        ),
    )
    np.testing.assert_array_equal(
        _leaf_u8(pair[f"{prefix}.w13.scale"]),
        np.concatenate(
            [
                arrays["w1.scale"][:1],
                arrays["w3.scale"][:1],
                arrays["w1.scale"][1:],
                arrays["w3.scale"][1:],
            ]
        ),
    )


def test_stack_projections_dense_pair_and_stand_ins() -> None:
    """The dense pair never interleaves; stand-ins stack by shape alone.

    ``wqa_wkv`` does not shard, so it stays the plain ``[wq_a; wkv]``
    concatenation at any device count, and a pair of zero-byte stand-ins
    (shaped headers carrying no data) folds to the summed shape with no
    bytes.
    """
    rng = np.random.default_rng(3)
    wq_a = rng.integers(1, 250, size=(32, 32), dtype=np.uint8)
    wkv = rng.integers(1, 250, size=(32, 32), dtype=np.uint8)
    state: dict[str, WeightData] = {}
    for name, array in (
        ("layers.1.attn.wq_a.weight", wq_a),
        ("layers.1.attn.wkv.weight", wkv),
    ):
        state[name] = _wd(array, name, DType.float8_e4m3fn)
    for name in ("layers.1.attn.wq_a.scale", "layers.1.attn.wkv.scale"):
        state[name] = _wd(
            np.full((1, 1), 127, np.uint8), name, DType.float8_e8m0fnu
        )
    for side in ("w1", "w3"):
        for part, shape in (("weight", (0, 32)), ("scale", (0, 1))):
            name = f"layers.0.ffn.shared_experts.{side}.{part}"
            state[name] = WeightData(
                Buffer.from_dlpack(np.zeros((0,), np.uint8)),
                name,
                DType.float8_e4m3fn
                if part == "weight"
                else DType.float8_e8m0fnu,
                Shape(list(shape)),
            )
    stack_projections(state, n_devices=2)
    dense = state["layers.1.attn.wqa_wkv.weight"]
    assert [int(d) for d in dense.shape] == [64, 32]
    np.testing.assert_array_equal(_leaf_u8(dense), np.concatenate([wq_a, wkv]))
    stand_in = state["layers.0.ffn.shared_experts.w13.weight"]
    assert [int(d) for d in stand_in.shape] == [0, 32]
    assert _leaf_u8(stand_in).size == 0


def test_stack_projections_refuses_rank_splitting_a_scale_block() -> None:
    """A rank boundary that would split a scale block fails loudly."""
    prefix = "layers.2.ffn.shared_experts"
    pair, _ = _shared_pair(prefix, seed=13)
    with pytest.raises(ValueError, match="whole"):
        stack_projections(pair, n_devices=4)
