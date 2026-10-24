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
"""Tests the MiMo-V2 weight adapter on tensors named as Xiaomi's checkpoint
names them."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import SafetensorWeights
from max.nn.quant_config import QuantFormat
from max.pipelines.architectures.mimo_v2.quant import parse_quant_scheme
from max.pipelines.architectures.mimo_v2.weight_adapters import (
    convert_safetensor_state_dict,
)
from transformers.configuration_utils import PretrainedConfig

QUANTIZATION = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "store_dtype": "mxfp4",
    "mxfp4_block_size": 32,
    "ignored_layers": ["model.layers.0.self_attn.o_proj"],
}

# Checkpoint name -> (safetensors dtype, shape, MAX name).
TENSORS = {
    "model.embed_tokens.weight": ("BF16", [8, 4], "embed_tokens.weight"),
    "lm_head.weight": ("BF16", [8, 4], "lm_head.weight"),
    "model.layers.0.self_attn.qkv_proj.weight": (
        "F8_E4M3",
        [8, 4],
        "layers.0.self_attn.qkv_proj.weight",
    ),
    "model.layers.0.self_attn.qkv_proj.weight_scale_inv": (
        "F32",
        [1, 1],
        "layers.0.self_attn.qkv_proj.weight_scale",
    ),
    "model.layers.0.mlp.down_proj.weight_scale_inv": (
        "F32",
        [1, 1],
        "layers.0.mlp.down_proj.weight_scale",
    ),
    "model.layers.1.mlp.gate.weight": (
        "BF16",
        [2, 4],
        "layers.1.mlp.gate.gate_score.weight",
    ),
    "model.layers.1.mlp.gate.e_score_correction_bias": (
        "F32",
        [2],
        "layers.1.mlp.gate.e_score_correction_bias",
    ),
    "model.layers.1.mlp.experts.0.up_proj.weight": (
        "U8",
        [4, 16],
        "layers.1.mlp.experts.0.up_proj.weight",
    ),
    "model.layers.1.mlp.experts.0.up_proj.weight_scale": (
        "U8",
        [4, 1],
        "layers.1.mlp.experts.0.up_proj.weight_scale",
    ),
}
IGNORED = (
    "visual.blocks.0.attn.qkv.weight",
    "audio_encoder.projection.mlp.0.weight",
    "speech_embeddings.0.weight",
    "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv",
)
_ITEMSIZE = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
_UINT = {1: DType.uint8, 2: DType.uint16, 4: DType.uint32}


def _config(**quantization: Any) -> PretrainedConfig:
    return PretrainedConfig(
        num_hidden_layers=2,
        moe_layer_freq=[0, 1],
        quantization_config={**QUANTIZATION, **quantization},
    )


def _write(path: Path, rng: np.random.Generator) -> dict[str, bytes]:
    """Writes every tensor as random bytes and returns the bytes."""
    tensors = {
        name: (dtype, shape) for name, (dtype, shape, _) in TENSORS.items()
    }
    tensors |= {name: ("BF16", [2, 2]) for name in IGNORED}
    header: dict[str, Any] = {}
    blobs: dict[str, bytes] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = _ITEMSIZE[dtype] * int(np.prod(shape))
        blobs[name] = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs.values())
    )
    return blobs


def test_renames_the_checkpoint_without_moving_bytes(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    blobs = _write(path, np.random.default_rng(0))

    adapted = convert_safetensor_state_dict(
        dict(SafetensorWeights([path]).items()), huggingface_config=_config()
    )

    assert sorted(adapted) == sorted(name for *_, name in TENSORS.values())
    for source, (_, shape, name) in TENSORS.items():
        data = adapted[name]
        assert data.name == name
        assert list(data.shape) == shape, name
        buffer = Buffer.from_dlpack(data.data)
        raw = np.from_dlpack(buffer.view(_UINT[data.dtype.size_in_bytes]))
        assert raw.tobytes() == blobs[source], name
    assert (
        adapted["layers.1.mlp.experts.0.up_proj.weight_scale"].dtype
        == DType.float8_e8m0fnu
    )
    assert adapted["layers.1.mlp.experts.0.up_proj.weight"].dtype == DType.uint8
    assert (
        adapted["layers.0.self_attn.qkv_proj.weight_scale"].dtype
        == DType.float32
    )


@pytest.mark.parametrize(
    "quantization, match",
    [
        ({"quant_method": "modelopt"}, "quant_method is 'modelopt'"),
        ({"weight_block_size": [64, 64]}, r"weight_block_size is \[64, 64\]"),
        ({"store_dtype": None}, "store_dtype is None"),
        ({"activation_scheme": "static"}, "activation_scheme is 'static'"),
    ],
)
def test_other_quantizations_fail(
    quantization: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_quant_scheme(_config(**quantization))


def test_parse_quant_scheme() -> None:
    scheme = parse_quant_scheme(_config())

    assert scheme.dense.format == QuantFormat.BLOCKSCALED_FP8
    assert scheme.dense.weight_scale.block_size == (128, 128)
    assert scheme.dense.mlp_quantized_layers == {0}
    assert scheme.dense.attn_quantized_layers == {0, 1}
    assert scheme.experts.format == QuantFormat.MXFP4
    assert scheme.experts.weight_scale.dtype == DType.float8_e8m0fnu
    assert scheme.experts.mlp_quantized_layers == {1}
