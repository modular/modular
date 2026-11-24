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
"""Parsing Nemotron-H's per-layer block types."""

from __future__ import annotations

import pytest
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    LayerKind,
    parse_layer_kinds,
)


def test_both_block_type_spellings_parse_to_layer_kinds() -> None:
    assert parse_layer_kinds(
        [
            "mamba",
            "linear_attention",
            "attention",
            "full_attention",
            "moe",
            "mlp",
        ]
    ) == [
        LayerKind.MAMBA,
        LayerKind.MAMBA,
        LayerKind.ATTENTION,
        LayerKind.ATTENTION,
        LayerKind.MOE,
        LayerKind.MLP,
    ]


def test_an_unknown_block_type_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="'mamab'"):
        parse_layer_kinds(["mamba", "mamab"])
