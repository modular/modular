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
"""Checks the Inkling 2 vision tower's input contract, tracing only.

MAX serve binds the packed buffers to the tower positionally, and a serving
engine binds the preprocessing graph's outputs to it by path. Both break
silently if a field moves, changes dtype or rank, or leaves the host.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from max import tree
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.inkling2.batch_vision_inputs import (
    HOST_INPUTS,
)
from max.pipelines.architectures.inkling2.vision import (
    Inkling2Vision,
    Inkling2VisionConfig,
)
from max.pipelines.architectures.inkling2.vision_inputs import (
    VisionInputs,
    vision_inputs,
)

# The published checkpoint's vision geometry; the tower's depth and widths do
# not enter the input contract.
_VISION = Inkling2VisionConfig(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=23,
    num_attention_heads=16,
    patch_size=20,
    spatial_merge_size=2,
    posemb_grid_size=32,
    projection_hidden_dim=4096,
    text_hidden_size=4608,
    layer_norm_eps=1e-5,
    rope_theta=10_000.0,
)


def _input_leaves() -> tuple[list[Any], list[str]]:
    """The tower's input leaves and the field name each one binds."""
    owner = SimpleNamespace(vision_config=_VISION, _device_ref=DeviceRef.GPU())
    leaves, treedef = tree.flatten(Inkling2Vision.input_types(owner))  # type: ignore[arg-type]
    names = [
        str(path).replace("/", ".").split(".")[-1]
        for path in treedef.leaf_paths
    ]
    return leaves, names


def test_vision_inputs_follow_the_tower_tree_order() -> None:
    _, names = _input_leaves()
    assert names == list(VisionInputs._fields)


def test_host_inputs_are_the_tower_cpu_inputs() -> None:
    leaves, names = _input_leaves()
    on_host = {
        name
        for name, leaf in zip(names, leaves, strict=True)
        if leaf.device == DeviceRef.CPU()
    }
    assert on_host == set(HOST_INPUTS)


def test_packed_inputs_match_the_tower_dtypes_and_ranks() -> None:
    leaves, names = _input_leaves()
    side = _VISION.patch_size * _VISION.spatial_merge_size
    grid = np.array([[1, 2, 3], [1, 1, 2]], dtype=np.int64)
    tokens = int(grid.prod(axis=1).sum())
    packed = vision_inputs(
        grid,
        np.zeros((tokens, side, side, 3), dtype=np.int16),
        merge=_VISION.spatial_merge_size,
        posemb_side=_VISION.posemb_grid_size,
    )
    for name, leaf in zip(names, leaves, strict=True):
        array = getattr(packed, name)
        assert DType.from_numpy(array.dtype) == leaf.dtype, name
        assert array.ndim == leaf.rank, name


def test_preprocess_graph_returns_the_tower_inputs_in_order() -> None:
    """The engine binds the preprocessing graph's outputs to the tower in
    order, so each output is the tower's input at that position, on host."""
    leaves, names = _input_leaves()
    config = SimpleNamespace(
        vision_config=_VISION, dtype=DType.bfloat16, devices=[DeviceRef.GPU()]
    )
    spec = Inkling2Vision.vision_preprocess_spec(None, config)  # type: ignore[arg-type]
    outputs, _ = tree.flatten(spec.trace().output_tree)
    assert len(outputs) == len(leaves)
    for name, output, want in zip(names, outputs, leaves, strict=True):
        assert (output.dtype, output.rank) == (want.dtype, want.rank), name
