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
"""Tests the ModuleV3 names the Laguna weight adapter emits.

An NVFP4 projection carries its packed data and three scales under the names
the checkpoint uses; a bfloat16 projection carries a plain ``weight``. Which is
which varies per layer in S-2.1, so both kinds appear in one state dict here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData
from max.pipelines.architectures.laguna_modulev3.weight_adapters import (
    to_graph_names,
    to_modulev3_names,
)

_NVFP4_EXPERT = "layers.1.mlp.experts.0.gate_proj"
_BF16_EXPERT = "layers.40.mlp.experts.0.gate_proj"


def _data(arr: np.ndarray, name: str) -> WeightData:
    return WeightData.from_numpy(arr, name)


def _graph_state_dict() -> dict[str, WeightData]:
    """A slice of S-2.1 as the graph adapter names it."""
    return {
        "embed_tokens.weight": _data(np.zeros((4, 2), np.float32), "e"),
        "lm_head.weight": _data(np.zeros((4, 2), np.float32), "l"),
        "layers.1.self_attn.q_proj.weight": _data(
            np.zeros((2, 2), np.float32), "q"
        ),
        "layers.1.self_attn.q_norm.weight": _data(
            np.zeros((2,), np.float32), "qn"
        ),
        f"{_NVFP4_EXPERT}.weight": _data(np.zeros((2, 8), np.uint8), "w"),
        f"{_NVFP4_EXPERT}.weight_scale": _data(
            np.zeros((2, 1), np.uint8), "ws"
        ),
        f"{_NVFP4_EXPERT}.weight_scale_2": _data(
            np.ones((1,), np.float32), "ws2"
        ),
        f"{_NVFP4_EXPERT}.input_scale": _data(np.ones((), np.float32), "is"),
        f"{_BF16_EXPERT}.weight": _data(np.zeros((2, 16), np.float32), "b"),
    }


def test_nvfp4_projections_keep_their_checkpoint_names() -> None:
    names = set(to_modulev3_names(_graph_state_dict()))
    proj = f"language_model.{_NVFP4_EXPERT}"
    assert {
        f"{proj}.weight",
        f"{proj}.weight_scale",
        f"{proj}.weight_scale_2",
        f"{proj}.input_scale",
    } <= names


def test_unquantized_weights_keep_a_plain_weight() -> None:
    names = set(to_modulev3_names(_graph_state_dict()))
    assert {
        "language_model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "language_model.layers.1.self_attn.q_proj.weight",
        "language_model.layers.1.self_attn.q_norm.weight",
        f"language_model.{_BF16_EXPERT}.weight",
    } <= names


def test_global_scales_become_scalars() -> None:
    out = to_modulev3_names(_graph_state_dict())
    proj = f"language_model.{_NVFP4_EXPERT}"
    for leaf in ("weight_scale_2", "input_scale"):
        assert list(out[f"{proj}.{leaf}"].shape) == []
        # The loader re-exports every weight through DLPack.
        Buffer.from_dlpack(out[f"{proj}.{leaf}"])
    # A per-group block scale is a matrix and must keep its shape.
    assert list(out[f"{proj}.weight_scale"].shape) == [2, 1]


def test_relabeled_block_scales_are_stored_as_float8() -> None:
    # The graph adapter reinterprets uint8 block scales as float8 by relabeling
    # the WeightData; the ModuleV3 loader reads the storage dtype instead.
    state_dict = _graph_state_dict()
    name = f"{_NVFP4_EXPERT}.weight_scale"
    state_dict[name] = dataclasses.replace(
        state_dict[name], dtype=DType.float8_e4m3fn
    )
    out = to_modulev3_names(state_dict)
    scale = out[f"language_model.{_NVFP4_EXPERT}.weight_scale"]
    assert scale.dtype == DType.float8_e4m3fn
    assert Buffer.from_dlpack(scale.data).dtype == DType.float8_e4m3fn


def test_a_non_scalar_global_scale_is_rejected() -> None:
    state_dict = _graph_state_dict()
    state_dict[f"{_NVFP4_EXPERT}.weight_scale_2"] = _data(
        np.ones((2,), np.float32), "ws2"
    )
    with pytest.raises(ValueError, match="single per-tensor scale"):
        to_modulev3_names(state_dict)


def test_graph_names_round_trip() -> None:
    graph = _graph_state_dict()
    assert set(to_graph_names(to_modulev3_names(graph))) == set(graph)


def test_tensors_are_not_rewritten() -> None:
    graph = _graph_state_dict()
    out = to_modulev3_names(graph)
    packed = f"language_model.{_NVFP4_EXPERT}.weight"
    assert out[packed] is graph[f"{_NVFP4_EXPERT}.weight"]
    assert out[packed].dtype == DType.uint8
