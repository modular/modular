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
"""Tests the MiMo-V2 weight adapter on a synthetic NVFP4 export.

The fixture starts from Xiaomi-format tensors (E8M0 experts, FP8 dense in
chunk order) and derives the export from them with a copy of the export's
converter (``tools/convert_exact_nvfp4.py`` in the NVFP4 repository). The
adapter must give back the Xiaomi bytes, which it never sees.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import SafetensorWeights, WeightData
from max.nn.quant_config import QuantFormat
from max.pipelines.architectures.mimo_v2.quant import parse_quant_scheme
from max.pipelines.architectures.mimo_v2.weight_adapters import (
    QkvChunkLayout,
    convert_safetensor_state_dict,
    fp8_block_scaled_from_float32,
    interleaved_scale_shape,
    qkv_chunk_layout,
)
from max.pipelines.weights._fp8 import e4m3fn_lut
from transformers.configuration_utils import PretrainedConfig

HIDDEN = 128
INTERMEDIATE = 256
# Whole 128-row by 4-column E8M0 granules, which the adapter interleaves.
MOE_INTERMEDIATE = 128
EXPERTS = 2
# Layer 0: full attention, dense MLP. Layer 1: sliding, MoE. Layer 2: full, MoE.
PATTERN = [0, 1, 0]
MOE_FREQ = [0, 1, 1]
# (q, k, v) rows per chunk for 8 query heads in 4 chunks. Full attention pads
# 704 rows to 768, like the real model's 3,392 -> 3,456; sliding needs none.
FULL_ROWS = (384, 192, 128)
SLIDING_ROWS = (384, 384, 256)
CHUNKS = 4

Tensors = dict[str, tuple[str, np.ndarray]]


def _config(**overrides: Any) -> PretrainedConfig:
    quantized = {
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
        }
        for layer in (1, 2)
        for expert in range(EXPERTS)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    values: dict[str, Any] = dict(
        num_hidden_layers=len(PATTERN),
        hybrid_layer_pattern=PATTERN,
        moe_layer_freq=MOE_FREQ,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        moe_intermediate_size=MOE_INTERMEDIATE,
        num_attention_heads=8,
        swa_num_attention_heads=8,
        num_key_value_heads=4,
        swa_num_key_value_heads=8,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        n_routed_experts=EXPERTS,
        num_nextn_predict_layers=1,
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        conversion_metadata={"qkv_layout": "global_q_k_v"},
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "quantized_layers": quantized,
        },
    )
    values.update(overrides)
    return PretrainedConfig(**values)


def _export_scales(e8m0: np.ndarray, global_exponent: int) -> np.ndarray:
    """The converter's ``encode_scales``: E8M0 per 32 -> E4M3 per 16."""
    exponents = e8m0.astype(np.int16) - 127 - global_exponent
    assert exponents.min() >= -9 and exponents.max() <= 8
    encoded = np.empty(e8m0.shape, dtype=np.uint8)
    normal = exponents >= -6
    encoded[normal] = ((exponents[normal] + 7) << 3).astype(np.uint8)
    encoded[~normal] = np.left_shift(1, exponents[~normal] + 9).astype(np.uint8)
    return np.repeat(encoded, 2, axis=-1)


def _dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """FP8 codes times 128x128 block scales, in float32 as the converter."""
    grid = np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)
    return e4m3fn_lut()[codes] * grid[: codes.shape[0], : codes.shape[1]]


def _fp8_block(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    """Random finite E4M3 codes whose block maximum is 448, as amax / 448
    quantization produces."""
    codes = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x80
    codes[::128, ::128] = 0x7E
    return codes


def _chunk_rows(layout: tuple[int, int, int]) -> tuple[int, int]:
    rows = sum(layout)
    return rows, -(-rows // 128) * 128


def _upstream_qkv(
    rng: np.random.Generator, layout: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Chunk-order codes (pad rows zero) and scales."""
    rows, padded = _chunk_rows(layout)
    codes = _fp8_block(rng, CHUNKS * padded, HIDDEN)
    codes.reshape(CHUNKS, padded, HIDDEN)[:, rows:] = 0
    scales = rng.uniform(2**-12, 2**-6, (CHUNKS * padded // 128, 1))
    return codes, scales.astype(np.float32)


def _export_qkv(
    codes: np.ndarray, scales: np.ndarray, layout: tuple[int, int, int]
) -> np.ndarray:
    """The converter's dequantize-then-reorder to global ``[Q; K; V]``."""
    q, k, _ = layout
    rows, padded = _chunk_rows(layout)
    chunks = _dequantize(codes, scales).reshape(CHUNKS, padded, HIDDEN)
    chunks = chunks[:, :rows]
    return np.concatenate(
        [
            chunks[:, :q].reshape(-1, HIDDEN),
            chunks[:, q : q + k].reshape(-1, HIDDEN),
            chunks[:, q + k :].reshape(-1, HIDDEN),
        ]
    )


def _bf16(rng: np.random.Generator, *shape: int) -> tuple[str, np.ndarray]:
    return "BF16", rng.integers(0, 2**16, size=shape, dtype=np.uint16)


def _checkpoint() -> tuple[Tensors, dict[str, tuple[DType, np.ndarray]]]:
    """Returns the export's tensors and the expected adapted tensors.

    The expected tensors are keyed by MAX name, as raw numpy arrays with the
    dtype MAX should report, and the expert scale stacks in row-major order.
    """
    rng = np.random.default_rng(0)
    tensors: Tensors = {}
    expected: dict[str, tuple[DType, np.ndarray]] = {}

    def passthrough(name: str, value: tuple[str, np.ndarray]) -> None:
        tensors[name] = value
        dtype = {"BF16": DType.bfloat16, "F32": DType.float32}[value[0]]
        expected[name.removeprefix("model.")] = (dtype, value[1])

    def dense(name: str, codes: np.ndarray, scales: np.ndarray) -> None:
        tensors[name] = ("F32", _dequantize(codes, scales))
        base = name.removeprefix("model.").removesuffix(".weight")
        expected[f"{base}.weight"] = (DType.float8_e4m3fn, codes)
        expected[f"{base}.weight_scale"] = (DType.float32, scales)

    def qkv(prefix: str, sliding: bool) -> None:
        layout = SLIDING_ROWS if sliding else FULL_ROWS
        codes, scales = _upstream_qkv(rng, layout)
        name = f"{prefix}self_attn.qkv_proj.weight"
        tensors[name] = ("F32", _export_qkv(codes, scales, layout))
        base = name.removeprefix("model.").removesuffix(".weight")
        expected[f"{base}.weight"] = (DType.float8_e4m3fn, codes)
        expected[f"{base}.weight_scale"] = (DType.float32, scales)

    def mlp(prefix: str) -> None:
        for proj, (n, k) in (
            ("gate_proj", (INTERMEDIATE, HIDDEN)),
            ("up_proj", (INTERMEDIATE, HIDDEN)),
            ("down_proj", (HIDDEN, INTERMEDIATE)),
        ):
            scales = rng.uniform(2**-12, 2**-6, (n // 128, k // 128))
            dense(
                f"{prefix}mlp.{proj}.weight",
                _fp8_block(rng, n, k),
                scales.astype(np.float32),
            )

    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        passthrough(name, _bf16(rng, 32, HIDDEN))
    passthrough("model.norm.weight", _bf16(rng, HIDDEN))
    for layer, (sliding, is_moe) in enumerate(
        zip(PATTERN, MOE_FREQ, strict=True)
    ):
        prefix = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            passthrough(f"{prefix}{norm}.weight", _bf16(rng, HIDDEN))
        passthrough(
            f"{prefix}self_attn.o_proj.weight", _bf16(rng, HIDDEN, 1024)
        )
        if sliding:
            passthrough(f"{prefix}self_attn.attention_sink_bias", _bf16(rng, 8))
        qkv(prefix, bool(sliding))
        if not is_moe:
            mlp(prefix)
            continue
        router = rng.standard_normal((EXPERTS, HIDDEN)).astype(np.float32)
        tensors[f"{prefix}mlp.gate.weight"] = (
            "BF16",
            (router.view(np.uint32) >> 16).astype(np.uint16),
        )
        expected[f"layers.{layer}.mlp.gate.gate_score.weight"] = (
            DType.float32,
            (router.view(np.uint32) & 0xFFFF0000).view(np.float32),
        )
        passthrough(
            f"{prefix}mlp.gate.e_score_correction_bias",
            ("F32", rng.standard_normal(EXPERTS).astype(np.float32)),
        )
        stacks: dict[str, list[np.ndarray]] = {
            "gate_up_proj": [],
            "gate_up_proj_scale": [],
            "down_proj": [],
            "down_proj_scale": [],
        }
        for expert in range(EXPERTS):
            upstream = {}
            for proj, (n, k) in (
                ("gate_proj", (MOE_INTERMEDIATE, HIDDEN)),
                ("up_proj", (MOE_INTERMEDIATE, HIDDEN)),
                ("down_proj", (HIDDEN, MOE_INTERMEDIATE)),
            ):
                base = f"{prefix}mlp.experts.{expert}.{proj}"
                codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
                # 110-112 become the subnormal E4M3 bytes 1, 2 and 4 under
                # the global 2^-8, as in the real layers 3-7.
                e8m0 = rng.integers(110, 128, (n, k // 32), dtype=np.uint8)
                e8m0[:3, 0] = (110, 111, 112)
                tensors[f"{base}.weight"] = ("U8", codes)
                tensors[f"{base}.weight_scale"] = (
                    "F8_E4M3",
                    _export_scales(e8m0, global_exponent=-8),
                )
                tensors[f"{base}.weight_scale_2"] = (
                    "F32",
                    np.array([2.0**-8], dtype=np.float32),
                )
                upstream[proj] = (codes, e8m0)
            for i, suffix in enumerate(("", "_scale")):
                stacks[f"gate_up_proj{suffix}"].append(
                    np.concatenate(
                        [upstream["gate_proj"][i], upstream["up_proj"][i]]
                    )
                )
                stacks[f"down_proj{suffix}"].append(upstream["down_proj"][i])
        for stack, arrays in stacks.items():
            dtype = (
                DType.float8_e8m0fnu
                if stack.endswith("_scale")
                else DType.uint8
            )
            expected[f"layers.{layer}.mlp.experts_{stack}"] = (
                dtype,
                np.stack(arrays),
            )

    prefix = "model.mtp.layers.0."
    for norm in (
        "enorm",
        "hnorm",
        "input_layernorm",
        "pre_mlp_layernorm",
        "final_layernorm",
    ):
        passthrough(f"{prefix}{norm}.weight", _bf16(rng, HIDDEN))
    passthrough(f"{prefix}eh_proj.weight", _bf16(rng, HIDDEN, 2 * HIDDEN))
    passthrough(f"{prefix}self_attn.o_proj.weight", _bf16(rng, HIDDEN, 1024))
    passthrough(f"{prefix}self_attn.attention_sink_bias", _bf16(rng, 8))
    qkv(prefix, sliding=True)
    mlp(prefix)
    # The adapter ignores the MTP layers.
    expected = {k: v for k, v in expected.items() if not k.startswith("mtp.")}

    # Towers the text model never reads.
    tensors["visual.patch_embed.proj.weight"] = _bf16(rng, 4, 4)
    tensors["audio_encoder.projection.mlp.0.weight"] = _bf16(rng, 4, 4)
    tensors["speech_embeddings.0.weight"] = _bf16(rng, 4, 4)
    return tensors, expected


def _write_safetensors(path: Path, tensors: Tensors) -> None:
    header: dict[str, Any] = {}
    blobs = []
    offset = 0
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for raw in blobs:
            f.write(raw)


def _adapt(
    tmp_path: Path,
    tensors: Tensors,
    config: PretrainedConfig | None = None,
) -> dict[str, WeightData]:
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, tensors)
    weights = SafetensorWeights([path])
    return convert_safetensor_state_dict(
        dict(weights.items()), config or _config()
    )


def _raw(data: WeightData) -> np.ndarray:
    """The bytes of ``data`` as unsigned integers of the element width."""
    view = {1: DType.uint8, 2: DType.uint16, 4: DType.uint32}
    assert isinstance(data.data, Buffer)
    return np.from_dlpack(data.data.view(view[data.dtype.size_in_bytes]))


def _assert_adapted(
    adapted: dict[str, WeightData],
    expected: dict[str, tuple[DType, np.ndarray]],
) -> None:
    assert sorted(adapted) == sorted(expected)
    for name, (dtype, array) in expected.items():
        assert adapted[name].dtype == dtype, name
        assert adapted[name].name == name
        want = array.view(
            {1: np.uint8, 2: np.uint16, 4: np.uint32}[array.itemsize]
        )
        np.testing.assert_array_equal(_raw(adapted[name]), want, err_msg=name)


def _kernel_layout(scales: np.ndarray) -> np.ndarray:
    """``[E, N, K/32]`` scales, each placed where the SM100 grouped matmul
    reads it (``set_scale_factor`` in ``fp4_utils.mojo``)."""
    experts, rows, cols = scales.shape
    out = np.zeros((experts, rows // 128, cols // 4, 32, 4, 4), np.uint8)
    for r in range(rows):
        for c in range(cols):
            out[:, r // 128, c // 4, r % 32, (r % 128) // 32, c % 4] = scales[
                :, r, c
            ]
    return out


def _interleaved(
    expected: dict[str, tuple[DType, np.ndarray]],
) -> dict[str, tuple[DType, np.ndarray]]:
    """``expected`` with the expert scale stacks in the kernel layout."""
    return {
        name: (
            (dtype, _kernel_layout(array))
            if ".mlp.experts_" in name and name.endswith("_scale")
            else (dtype, array)
        )
        for name, (dtype, array) in expected.items()
    }


def test_adapts_export_to_upstream_bytes(tmp_path: Path) -> None:
    tensors, expected = _checkpoint()
    _assert_adapted(_adapt(tmp_path, tensors), _interleaved(expected))


@pytest.mark.parametrize("rows, cols", [(64, 4), (128, 2)])
def test_interleave_needs_whole_granules(rows: int, cols: int) -> None:
    with pytest.raises(ValueError, match="interleave granules"):
        interleaved_scale_shape(rows, cols)


def test_qkv_chunk_layout_matches_the_real_checkpoint() -> None:
    config = _config(num_attention_heads=64, swa_num_attention_heads=64)
    full = qkv_chunk_layout(config, sliding=False)
    sliding = qkv_chunk_layout(config, sliding=True)
    assert full == QkvChunkLayout(chunks=4, q_rows=3072, k_rows=192, v_rows=128)
    assert (full.rows, full.padded_rows) == (3392, 3456)
    assert sliding == QkvChunkLayout(
        chunks=4, q_rows=3072, k_rows=384, v_rows=256
    )
    assert (sliding.rows, sliding.padded_rows) == (3712, 3712)


def test_scale_search_recovers_an_off_by_one_ulp_scale() -> None:
    rng = np.random.default_rng(1)
    for scale in rng.uniform(2**-12, 2**-6, 1000).astype(np.float32):
        if np.float32(np.float32(448) * scale) / np.float32(448) != scale:
            break
    else:
        pytest.fail("no scale with fl(448 * s) / 448 != s")
    codes = _fp8_block(rng, 128, 128)
    weight = _dequantize(codes, np.array([[scale]], dtype=np.float32))

    got_codes, got_scale = fp8_block_scaled_from_float32(weight, "w")

    assert got_scale[0, 0] == scale
    np.testing.assert_array_equal(got_codes, codes)


_EXPERT = "model.layers.1.mlp.experts.0.gate_proj"
_EXPERT_RE = re.escape(_EXPERT)


def _non_power_of_two_scale(tensors: Tensors) -> None:
    tensors[f"{_EXPERT}.weight_scale"][1][3, 4:6] = 0x39  # 1.125


def _scale_halves_differ(tensors: Tensors) -> None:
    tensors[f"{_EXPERT}.weight_scale"][1][5, 0:2] = (0x38, 0x40)  # 1, 2


def _zero_scale(tensors: Tensors) -> None:
    tensors[f"{_EXPERT}.weight_scale"][1][0, 6:8] = 0


def _delete_weight_scale_2(tensors: Tensors) -> None:
    del tensors["model.layers.2.mlp.experts.1.down_proj.weight_scale_2"]


def _add_input_scale(tensors: Tensors) -> None:
    tensors[f"{_EXPERT}.input_scale"] = ("F32", np.ones(1, dtype=np.float32))


def _add_unknown_tensor(tensors: Tensors) -> None:
    tensors["model.layers.1.mlp.shared_experts.up_proj.weight"] = (
        "BF16",
        np.zeros((4, 4), dtype=np.uint16),
    )


def _perturb_f32(tensors: Tensors) -> None:
    weight = tensors["model.layers.0.self_attn.qkv_proj.weight"][1]
    weight.view(np.uint32)[300, 7] ^= 1


def _perturb_sliding_qkv(tensors: Tensors) -> None:
    # Chunk 0's V row 100, which sits in chunk-order row block 6.
    weight = tensors["model.layers.1.self_attn.qkv_proj.weight"][1]
    weight.view(np.uint32)[CHUNKS * 768 + 100, 5] ^= 1


def _perturb_straddle_block(tensors: Tensors) -> None:
    # Chunk 1's V row 24. Its block also holds the chunk's last K rows, and in
    # global order those belong to another chunk's scale.
    weight = tensors["model.layers.2.self_attn.qkv_proj.weight"][1]
    weight.view(np.uint32)[CHUNKS * 576 + 128 + 24, 5] ^= 1


def _delete_router_bias(tensors: Tensors) -> None:
    del tensors["model.layers.2.mlp.gate.e_score_correction_bias"]


def _wrong_expert_shape(tensors: Tensors) -> None:
    tensors["model.layers.2.mlp.experts.1.down_proj.weight"] = (
        "U8",
        np.zeros((HIDDEN, 1), dtype=np.uint8),
    )


def _mistype(tensors: Tensors) -> None:
    tensors["model.norm.weight"] = ("F32", np.ones(HIDDEN, dtype=np.float32))


def _delete_sink(tensors: Tensors) -> None:
    del tensors["model.layers.1.self_attn.attention_sink_bias"]


@pytest.mark.parametrize(
    "mutate, config_overrides, match",
    [
        (
            _non_power_of_two_scale,
            {},
            rf"{_EXPERT_RE}: MXFP4 block \(row 3, block 2\) .* power of two",
        ),
        (
            _scale_halves_differ,
            {},
            rf"{_EXPERT_RE}: MXFP4 block \(row 5, block 0\) .* differ",
        ),
        (
            _delete_weight_scale_2,
            {},
            r"missing .*model\.layers\.2\.mlp\.experts\.1\.down_proj"
            r"\.weight_scale_2",
        ),
        (_add_input_scale, {}, rf"input_scale .*{_EXPERT_RE}\.input_scale"),
        (
            None,
            {"conversion_metadata": {"qkv_layout": "tp4_interleaved"}},
            "qkv_layout is 'tp4_interleaved'",
        ),
        (
            _add_unknown_tensor,
            {},
            r"neither read nor ignored.*shared_experts\.up_proj\.weight",
        ),
        (_zero_scale, {}, rf"{_EXPERT_RE}: MXFP4 block \(row 0, block 3\)"),
        (
            _perturb_f32,
            {},
            r"model\.layers\.0\.self_attn\.qkv_proj\.weight: FP8 block "
            r"\(row block 2, column block 0\)",
        ),
        (
            _perturb_sliding_qkv,
            {},
            r"model\.layers\.1\.self_attn\.qkv_proj\.weight: FP8 block "
            r"\(row block 6, column block 0\)",
        ),
        (
            _perturb_straddle_block,
            {},
            r"model\.layers\.2\.self_attn\.qkv_proj\.weight: FP8 block "
            r"\(row block 10, column block 0\)",
        ),
        (
            _delete_router_bias,
            {},
            r"missing .*model\.layers\.2\.mlp\.gate\.e_score_correction_bias",
        ),
        (
            _wrong_expert_shape,
            {},
            r"model\.layers\.2\.mlp\.experts\.1\.down_proj: weight \(128, 1\)",
        ),
        (_mistype, {}, r"model\.norm\.weight is DType\.float32; expected"),
        (
            _delete_sink,
            {},
            r"missing .*model\.layers\.1\.self_attn\.attention_sink_bias",
        ),
    ],
)
def test_defect_fails_loudly(
    tmp_path: Path,
    mutate: Callable[[Tensors], None] | None,
    config_overrides: dict[str, Any],
    match: str,
) -> None:
    tensors, _ = _checkpoint()
    if mutate is not None:
        mutate(tensors)
    with pytest.raises(ValueError, match=match):
        _adapt(tmp_path, tensors, _config(**config_overrides))


@pytest.mark.parametrize(
    "edit, match",
    [
        (
            lambda q: q["quantized_layers"].update(
                {
                    "model.layers.0.self_attn.qkv_proj": {
                        "quant_algo": "W4A16_NVFP4",
                        "group_size": 16,
                    }
                }
            ),
            r"not routed-expert projections.*layers\.0\.self_attn\.qkv_proj",
        ),
        (
            lambda q: q["quantized_layers"].pop(_EXPERT),
            rf"missing e\.g\. \['{_EXPERT_RE}'\]",
        ),
        (
            lambda q: q["quantized_layers"][_EXPERT].update(group_size=32),
            rf"quantized_layers\['{_EXPERT_RE}'\]",
        ),
        (
            lambda q: q.update(kv_cache_quant_algo="FP8"),
            "kv_cache_quant_algo is 'FP8'",
        ),
        (lambda q: q.update(quant_algo="NVFP4"), "quant_algo='NVFP4'"),
        (
            lambda q: q.update(quant_method="fp8", quant_algo=None),
            "quant_method='fp8'.* only NVFP4 exports",
        ),
    ],
)
def test_unsupported_quantization_config_fails(
    tmp_path: Path, edit: Callable[[dict[str, Any]], object], match: str
) -> None:
    tensors, _ = _checkpoint()
    config = _config()
    edit(config.quantization_config)
    with pytest.raises(ValueError, match=match):
        _adapt(tmp_path, tensors, config)


def test_parse_quant_scheme() -> None:
    scheme = parse_quant_scheme(_config())

    assert scheme.dense.format == QuantFormat.BLOCKSCALED_FP8
    assert scheme.dense.weight_scale.block_size == (128, 128)
    assert scheme.dense.mlp_quantized_layers == {0}
    assert scheme.dense.attn_quantized_layers == {0, 1, 2}
    assert scheme.experts.format == QuantFormat.MXFP4
    assert scheme.experts.weight_scale.dtype == DType.float8_e8m0fnu
    assert scheme.experts.mlp_quantized_layers == {1, 2}
