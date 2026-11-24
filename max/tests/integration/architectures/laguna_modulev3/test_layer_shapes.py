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
"""Tests how Laguna layers are grouped into shared subgraphs.

A subgraph's signature is fixed by the first layer traced into it, so two
layers may share one only when they agree on everything that changes the
block's graph. ``as_subgraph`` also keys the body by its symbol, so two
different shapes under one name would silently run one shape's graph for the
other's layers.
"""

from __future__ import annotations

import json
from pathlib import Path

from max.pipelines.architectures.hf_config_shims import LagunaHFConfig
from max.pipelines.architectures.laguna.model_config import LagunaConfig
from max.pipelines.architectures.laguna_modulev3.layers.transformer_block import (
    LagunaLayerShape,
    layer_shapes,
)

TESTDATA = Path(__file__).parents[1] / "laguna" / "testdata"

# S-2.1 ships NVFP4 routed experts in layers 1-39 and bfloat16 ones in 40-47.
_S21_NVFP4_LAYERS = set(range(1, 40))


def _shapes(config_name: str, quantized: set[int]) -> list[LagunaLayerShape]:
    with (TESTDATA / config_name).open() as f:
        hf_config = LagunaHFConfig(**json.load(f))
    return layer_shapes(LagunaConfig.read_layer_schedule(hf_config), quantized)


def test_s21_resolves_five_block_shapes() -> None:
    shapes = _shapes("s21_config.json", _S21_NVFP4_LAYERS)
    assert len(shapes) == 48
    assert set(shapes) == {
        LagunaLayerShape("full_attention", 48, "dense", False),
        LagunaLayerShape("full_attention", 48, "sparse", True),
        LagunaLayerShape("sliding_attention", 72, "sparse", True),
        LagunaLayerShape("full_attention", 48, "sparse", False),
        LagunaLayerShape("sliding_attention", 72, "sparse", False),
    }


def test_every_shape_has_its_own_subgraph_name() -> None:
    shapes = set(_shapes("s21_config.json", _S21_NVFP4_LAYERS))
    names = {shape.subgraph_name for shape in shapes}
    assert len(names) == len(shapes)


def test_m1_resolves_a_dense_and_a_sparse_shape() -> None:
    num_layers = len(_shapes("m1_config.json", set()))
    shapes = _shapes("m1_config.json", set(range(1, num_layers)))
    assert {(s.attention_type, s.mlp_type) for s in shapes} == {
        ("full_attention", "dense"),
        ("full_attention", "sparse"),
    }
