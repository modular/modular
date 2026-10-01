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
"""Tests that the MiMo-V2 DFlash drafter config and weights load loudly."""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.dflash_mimo_v2 import (
    DFlashMiMoV2Config,
    convert_safetensor_state_dict,
    load_mask_embedding,
)
from max.pipelines.architectures.dflash_mimo_v2.weight_adapters import (
    MASK_EMBEDDING,
    drafter_tensor_shapes,
)

# ``dflash/config.json`` of MiMo-V2.6-Flash-RL, shrunk to two layers.
CONFIG: dict[str, Any] = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "v_head_dim": 16,
    "partial_rotary_factor": 0.5,
    "block_size": 8,
    "dflash_config": {
        "target_layer_ids": [0, 11, 23],
        "mask_token_id": 151675,
        "num_anchors": 4096,
        "block_size": 8,
        "loss_decay_gamma": 7.0,
        "attention_value_scale": 0.612,
        "attention_sink_bias": True,
    },
    "layer_types": ["sliding_attention", "sliding_attention"],
    "sliding_window": 1024,
    "use_sliding_window": True,
    "is_causal": False,
    "num_target_layers": 48,
    "target_hidden_size": 64,
    "vocab_size": 152576,
    "max_position_embeddings": 1048576,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-06,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "add_swa_attention_sink_bias": True,
    "tie_word_embeddings": False,
    "use_cache": True,
}


def _config(raw: dict[str, Any] = CONFIG) -> DFlashMiMoV2Config:
    device = DeviceRef.CPU()
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        num_layers=raw["num_hidden_layers"],
        devices=[device],
    )
    return DFlashMiMoV2Config.from_dflash_config(
        raw, devices=[device], kv_params=kv_params, max_seq_len=4096
    )


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, np.ndarray]]
) -> None:
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


def _tensors(config: DFlashMiMoV2Config) -> dict[str, tuple[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    return {
        name: ("BF16", rng.integers(0, 2**15, shape, dtype=np.uint16))
        for name, shape in drafter_tensor_shapes(config).items()
    }


def _save_mask(path: Path, mask: torch.Tensor, token_id: int = 151675) -> None:
    torch.save({"mask_token_id": token_id, "embedding": mask}, path)


def _load(
    tmp_path: Path,
    tensors: dict[str, tuple[str, np.ndarray]],
    config: DFlashMiMoV2Config,
) -> dict[str, Any]:
    path = tmp_path / "dflash_draft_model.safetensors"
    _write_safetensors(path, tensors)
    mask = load_mask_embedding(tmp_path / "mask_embedding.pt", config)
    return convert_safetensor_state_dict(
        dict(SafetensorWeights([path]).items()), config, mask
    )


def test_config_matches_the_checkpoint() -> None:
    config = _config()
    assert config.rotary_dim == 8
    assert config.target_layer_ids == [0, 11, 23]
    assert config.attention_value_scale == 0.612
    assert config.sliding_window == 1024


@pytest.mark.parametrize(
    "edit",
    [
        {"is_causal": True},
        {"layer_types": ["sliding_attention", "full_attention"]},
        {"block_size": 16},
        {"add_swa_attention_sink_bias": False},
        {"v_head_dim": 8},
        {"target_hidden_size": 32},
        {"rope_scaling": {"rope_type": "yarn"}},
        {"rope_parameters": {"rope_type": "yarn"}},
        {"hidden_act": "gelu"},
        {"dflash_config": {"attention_sink_bias": False}},
        {"dflash_config": {"target_layer_ids": [0, 48]}},
        {"dflash_config": {"attention_value_scale": "0.612"}},
    ],
)
def test_config_refuses_other_conventions(edit: dict[str, Any]) -> None:
    raw = copy.deepcopy(CONFIG)
    for key, value in edit.items():
        if key == "dflash_config":
            raw[key].update(value)
        else:
            raw[key] = value
    with pytest.raises(ValueError, match="DFlash MiMo-V2"):
        _config(raw)


def test_loads_every_tensor_and_the_trained_mask(tmp_path: Path) -> None:
    config = _config()
    tensors = _tensors(config)
    mask = torch.randn(config.hidden_size).to(torch.bfloat16)
    _save_mask(tmp_path / "mask_embedding.pt", mask)

    weights = _load(tmp_path, tensors, config)

    assert weights.keys() == tensors.keys() | {MASK_EMBEDDING}
    for name, (_, array) in tensors.items():
        got = np.from_dlpack(weights[name].to_buffer().view(DType.uint16))
        np.testing.assert_array_equal(got, array)
    got = np.from_dlpack(weights[MASK_EMBEDDING].to_buffer().view(DType.uint16))
    np.testing.assert_array_equal(
        got, mask.view(torch.int16).numpy().view(np.uint16)
    )


@pytest.mark.parametrize(
    "defect",
    ["missing", "unused", "wrong dtype", "wrong shape", "extra layer"],
)
def test_drafter_tensor_defects_raise(tmp_path: Path, defect: str) -> None:
    config = _config()
    tensors = _tensors(config)
    sinks = "layers.1.self_attn.attention_sink_bias"
    if defect == "missing":
        del tensors[sinks]
    elif defect == "unused":
        tensors["embed_tokens.weight"] = ("BF16", np.zeros((8, 64), np.uint16))
    elif defect == "wrong dtype":
        tensors[sinks] = ("F32", np.zeros((4,), np.float32))
    elif defect == "wrong shape":
        tensors[sinks] = ("BF16", np.zeros((8,), np.uint16))
    else:
        tensors["layers.2.self_attn.attention_sink_bias"] = tensors[sinks]
    _save_mask(
        tmp_path / "mask_embedding.pt", torch.zeros(64, dtype=torch.bfloat16)
    )
    with pytest.raises(ValueError, match="DFlash MiMo-V2"):
        _load(tmp_path, tensors, config)


@pytest.mark.parametrize(
    "saved",
    [
        {
            "mask_token_id": 7,
            "embedding": torch.zeros(64, dtype=torch.bfloat16),
        },
        {
            "mask_token_id": 151675,
            "embedding": torch.zeros(32, dtype=torch.bfloat16),
        },
        {"mask_token_id": 151675},
        {
            "mask_token_id": 151675,
            "embedding": torch.zeros(64, dtype=torch.bfloat16),
            "extra": 1,
        },
        # Only the bfloat16 storage the checkpoint uses is admitted.
        {"mask_token_id": 151675, "embedding": torch.zeros(64)},
    ],
)
def test_mask_embedding_defects_raise(
    tmp_path: Path, saved: dict[str, Any]
) -> None:
    path = tmp_path / "mask_embedding.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="DFlash MiMo-V2"):
        load_mask_embedding(path, _config())
