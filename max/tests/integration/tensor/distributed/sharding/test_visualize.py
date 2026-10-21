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
"""Tests for the mesh and sharding drawings."""

from __future__ import annotations

import math

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
    Sharded,
    TensorLayout,
    Unknown,
    visualize,
)
from max.experimental.sharding.visualize import _grid, _svg
from max.experimental.tensor import Tensor
from max.graph import ShapeLike


def mesh(*shape: int) -> DeviceMesh:
    names = tuple(f"m{i}" for i in range(len(shape)))
    if len(shape) <= 3:
        names = ("pp", "dp", "tp")[-len(shape) :]
    return DeviceMesh(
        tuple(CPU() for _ in range(math.prod(shape))), shape, names
    )


def layout(shape: ShapeLike, *placements: Placement) -> TensorLayout:
    grid = mesh(*(2,) * len(placements)) if len(placements) > 1 else mesh(4)
    return TensorLayout(DType.float32, shape, DeviceMapping(grid, placements))


def shown(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.rstrip("\n").split("\n")


def test_mesh_tree(capsys: pytest.CaptureFixture[str]) -> None:
    visualize(mesh(2, 2), blocks=False)
    assert shown(capsys) == [
        "DeviceMesh(dp=2, tp=2)",
        "├── dp=0",
        "│   ├── tp=0  device 0",
        "│   └── tp=1  device 1",
        "└── dp=1",
        "    ├── tp=0  device 2",
        "    └── tp=1  device 3",
    ]
    visualize(mesh(2, 2, 2, 2), blocks=False)
    assert len(shown(capsys)) == 31


def test_view_flags(capsys: pytest.CaptureFixture[str]) -> None:
    value = layout([4, 8], Sharded(1))
    visualize(value, title="a")
    both = shown(capsys)
    assert both[0].startswith("a: DType.float32 [4, 8] with DeviceMapping(")
    # One header, then a block on the left and its tree node beside it.
    assert both[1].startswith("\u250c") and both[1].endswith("tensor[:, :]")

    visualize(value, title="a", tree=False)
    assert all("tensor[" not in line for line in shown(capsys))

    visualize(value, title="a", blocks=False)
    assert all("\u250c" not in line for line in shown(capsys))

    with pytest.raises(ValueError, match="blocks, tree, or both"):
        visualize(value, blocks=False, tree=False)


def test_layout_tree_carries_the_slice_per_node(
    capsys: pytest.CaptureFixture[str],
) -> None:
    visualize(layout([4, 8], Sharded(0), Replicated()), title="x", blocks=False)
    lines = shown(capsys)
    assert lines[0].startswith("x: DType.float32 [4, 8] with DeviceMapping(")
    assert lines[1] == "tensor[:, :]"
    assert lines[2] == "├── dp=0 - tensor[:2, :]"
    assert lines[-1] == "    └── tp=1  device 3 - tensor[2:, :]"


def test_symbolic_bounds_are_sums_of_per_shard_dims(
    capsys: pytest.CaptureFixture[str],
) -> None:
    visualize(layout(["batch", 8], Sharded(0)), blocks=False)
    lines = shown(capsys)
    assert lines[2].endswith("tensor[:batch_tp_0, :]")
    assert lines[-1].endswith(
        "tensor[batch_tp_0 + batch_tp_1 + batch_tp_2:, :]"
    )


def test_partial_is_marked(capsys: pytest.CaptureFixture[str]) -> None:
    visualize(layout([4, 8], Sharded(1), Partial()), blocks=False)
    lines = shown(capsys)
    assert lines[2] == "├── dp=0 - tensor[:, :4]"
    assert lines[3].endswith("device 0 - partial sum of tensor[:, :4]")
    visualize(layout([4, 8], Partial()))
    assert "[:, :] partial" in "\n".join(shown(capsys))


def test_unknown_is_marked(capsys: pytest.CaptureFixture[str]) -> None:
    visualize(layout([4, 8], Sharded(0), Unknown()), blocks=False)
    lines = shown(capsys)
    assert lines[2] == "├── dp=0 - tensor[:2, :]"
    assert lines[3].endswith("device 0 - unknown tensor[:2, :]")
    visualize(layout([4, 8], Unknown()))
    assert "[:, :] unknown" in "\n".join(shown(capsys))


def test_partial_and_unknown_blocks_are_see_through() -> None:
    for placement in (Partial(), Unknown()):
        grid = _grid(layout([4, 8], placement))
        assert grid is not None
        assert 'fill-opacity=".4"' in _svg(*grid)
    grid = _grid(layout([4, 8], Replicated()))
    assert grid is not None
    assert "fill-opacity" not in _svg(*grid)


def test_grid_of_an_uneven_split(capsys: pytest.CaptureFixture[str]) -> None:
    uneven = layout([7, 16], Sharded(0), Sharded(1))
    visualize(uneven)
    text = "\n".join(shown(capsys))
    assert "[:4, :8]" in text and "[4:, 8:]" in text
    grid = _grid(uneven)
    assert grid is not None
    rows, row_w, col_w = grid
    assert (row_w, col_w) == ([4.0, 3.0], [8.0, 8.0])
    assert [[cell[2] for cell in row] for row in rows] == [
        [[0], [1]],
        [[2], [3]],
    ]
    assert _svg(rows, row_w, col_w).count("<text") == 8


def test_replicated_cell_lists_every_device(
    capsys: pytest.CaptureFixture[str],
) -> None:
    visualize(layout([2, 4, 8], Replicated(), Sharded(1)))
    text = "\n".join(shown(capsys))
    assert "devices 0,2" in text and "[:, 2:, :]" in text


def test_mesh_grid(capsys: pytest.CaptureFixture[str]) -> None:
    visualize(mesh(2, 2), title="m")
    lines = shown(capsys)
    assert lines[0] == "m: DeviceMesh(dp=2, tp=2)"
    assert "device 3" in lines[-3] and "dp=1, tp=1" in lines[-2]


def test_too_many_axes_fall_back_to_the_tree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    deep = mesh(2, 2, 2)
    visualize(deep)
    fallback = shown(capsys)
    visualize(deep, blocks=False)
    assert fallback == shown(capsys)
    visualize(layout([2, 4, 8], Sharded(0), Sharded(1), Sharded(2)))
    lines = shown(capsys)
    assert all(not line.startswith("\u250c") for line in lines)
    assert lines[-1].endswith("device 7 - tensor[1:, 2:, 4:]")


def test_accepted_values(capsys: pytest.CaptureFixture[str]) -> None:
    tensor = Tensor.zeros(
        [4, 8],
        dtype=DType.float32,
        device=DeviceMapping(mesh(4), (Sharded(1),)),
    )
    visualize(tensor, blocks=False)
    assert shown(capsys)[-1].endswith("tensor[:, 6:]")
    with pytest.raises(TypeError, match="Wrap it in a TensorLayout"):
        visualize(DeviceMapping(mesh(4), (Sharded(0),)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DeviceMesh"):
        visualize("not a layout")  # type: ignore[arg-type]


def test_dict_draws_side_by_side(capsys: pytest.CaptureFixture[str]) -> None:
    lhs = layout([4, 8], Sharded(1))
    rhs = layout([8, 8], Sharded(0))
    out = layout([4, 8], Partial())
    visualize({"lhs": lhs, "rhs": rhs, "out": out}, title="row", tree=False)
    lines = shown(capsys)
    assert lines[0] == "row"
    assert lines[1].startswith("lhs: [4, 8] Sharded(axis=1)")
    assert "rhs: [8, 8] Sharded(axis=0)" in lines[1]
    assert lines[1].rstrip().endswith("out: [4, 8] Partial()")
    # The blocks of all three start on the same line.
    assert lines[2].count("\u250c") == 3
    with pytest.raises(ValueError, match="at least one value"):
        visualize({})
