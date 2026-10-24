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
"""Tests for building a CompiledCallable from a compiled graph and for its
artifact serialization."""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from max import tree
from max._artifact.value_types import (
    ArtifactError,
    ArtifactFile,
    ArtifactType,
    treedef_from_json,
    treedef_to_json,
)
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.compilation import CompiledCallable, compile
from max.experimental.sharding import (
    BufferLayout,
    DeviceMapping,
    DeviceMesh,
    Partial,
    Sharded,
    TensorLayout,
)
from max.experimental.sharding.types import as_layout
from max.experimental.tensor import Tensor
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.graph_groups import GraphGroup

X = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())


class _Pair(GraphGroup[str, int]):
    """A kinded group of ints, standing in for a graph's named outputs."""

    kind = "_test_pair"


def _scale(x: Tensor) -> Tensor:
    return x * 2


def _graph(name: str, factor: float) -> Graph:
    with Graph(name, input_types=[X]) as g:
        (x,) = g.inputs
        g.output(x.tensor * factor)
    return g


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


def _wrap(
    session: InferenceSession, graph: Graph
) -> CompiledCallable[..., Any]:
    output_types = graph.output_types
    compiled = session.compile(graph)
    in_layouts = [as_layout(X)]
    out_layouts = [as_layout(t) for t in output_types]
    _, in_tree = tree.flatten((tuple(in_layouts), {}), leaf=TensorLayout)
    _, out_tree = tree.flatten(tuple(out_layouts), leaf=TensorLayout)
    return CompiledCallable.from_compiled(
        compiled,
        session.init(compiled),
        in_layouts,
        in_tree,
        out_layouts,
        out_tree,
    )


def _serialize(
    compiled: CompiledCallable[..., Any], root: Path
) -> Mapping[str, ArtifactType]:
    """Writes the files a serialized callable references, as the artifact
    writer does, and returns the value the reader hands back."""
    value = dict(compiled.__serialize__())
    mef = value["mef"]
    assert isinstance(mef, ArtifactFile)
    dest = root / mef.name
    mef.write_to(dest)
    value["mef"] = ArtifactFile(mef.name, dest)
    return value


def _ones() -> Tensor:
    return Tensor(storage=Buffer.from_numpy(np.ones((2, 4), np.float32)))


def test_from_compiled_runs_like_a_staged_graph(
    session: InferenceSession,
) -> None:
    run = _wrap(session, _graph("scale", 3.0))
    (out,) = run(_ones())
    np.testing.assert_allclose(out.to_numpy(), np.full((2, 4), 3.0))
    (raw,) = run.execute_raw(Buffer.from_numpy(np.ones((2, 4), np.float32)))
    np.testing.assert_allclose(raw.to_numpy(), np.full((2, 4), 3.0))


def test_round_trip(tmp_path: Path, session: InferenceSession) -> None:
    run = _wrap(session, _graph("scale", 3.0))
    value = _serialize(run, tmp_path)
    assert set(value) == {
        "mef",
        "in_layouts",
        "in_tree",
        "out_layouts",
        "out_tree",
    }
    loaded: CompiledCallable[..., Any] = CompiledCallable.__deserialize__(value)
    assert loaded.in_layouts == run.in_layouts
    assert loaded.in_tree == run.in_tree
    assert loaded.out_layouts == run.out_layouts
    assert loaded.out_tree == run.out_tree
    assert set(loaded.__serialize__()) == set(value)
    with pytest.raises(RuntimeError, match="not initialized"):
        loaded(_ones())
    loaded.export_mef(tmp_path / "copy.mef")
    assert (tmp_path / "copy.mef").read_bytes() == (
        tmp_path / "model.mef"
    ).read_bytes()


def test_staged_graph_round_trip(tmp_path: Path) -> None:
    run = compile(_scale)(TensorLayout(DType.float32, [3], CPU()))
    loaded: CompiledCallable[..., Any] = CompiledCallable.__deserialize__(
        _serialize(run, tmp_path)
    )
    assert loaded.in_layouts == run.in_layouts
    assert loaded.out_tree == run.out_tree


_MESH = DeviceMesh((CPU(), CPU()), (2,), ("tp",))


@pytest.mark.parametrize(
    "layout",
    [
        as_layout(BufferType(DType.int32, ["n"], DeviceRef.CPU())),
        TensorLayout(
            DType.bfloat16, ["seq", 8], DeviceMapping.replicated(_MESH)
        ),
        TensorLayout(
            DType.float32, [4, "hidden"], DeviceMapping(_MESH, (Sharded(0),))
        ),
        BufferLayout(DType.float32, [4], DeviceMapping(_MESH, (Partial(),))),
    ],
)
def test_layouts_round_trip(
    tmp_path: Path, session: InferenceSession, layout: TensorLayout
) -> None:
    compiled = session.compile(_graph("scale", 3.0))
    _, in_tree = tree.flatten(((layout,), {}), leaf=TensorLayout)
    _, out_tree = tree.flatten((layout,), leaf=TensorLayout)
    run = CompiledCallable.from_compiled(
        compiled, None, [layout], in_tree, [layout], out_tree
    )
    loaded: CompiledCallable[..., Any] = CompiledCallable.__deserialize__(
        _serialize(run, tmp_path)
    )
    assert loaded.in_layouts == [layout]
    assert type(loaded.in_layouts[0]) is type(layout)
    assert loaded.out_layouts == [layout]


def test_a_sharded_layout_records_its_mesh(
    tmp_path: Path, session: InferenceSession
) -> None:
    layout = TensorLayout(
        DType.float32, [4, "hidden"], DeviceMapping(_MESH, (Sharded(0),))
    )
    compiled = session.compile(_graph("scale", 3.0))
    _, in_tree = tree.flatten(((layout,), {}), leaf=TensorLayout)
    run = CompiledCallable.from_compiled(
        compiled, None, [layout], in_tree, [], tree.flatten(())[1]
    )
    layouts = _serialize(run, tmp_path)["in_layouts"]
    assert isinstance(layouts, list)
    (encoded,) = layouts
    assert isinstance(encoded, Mapping)
    assert encoded["mesh"] == {
        "devices": [{"type": "cpu", "id": 0}, {"type": "cpu", "id": 0}],
        "shape": [2],
        "axis_names": ["tp"],
    }
    assert encoded["placements"] == [{"type": "sharded", "axis": 0}]


@pytest.mark.parametrize(
    "value",
    [
        ((1, [2, {"a": 3}]), {"gain": 0.5, "flag": True, "none": None}),
        [],
        {"x": "static"},
        OrderedDict(b=1, a=[2, 3]),
        [_Pair(logits=1, offsets=2)],
    ],
)
def test_treedef_json_round_trip(value: Any) -> None:
    _, treedef = tree.flatten(value, leaf=int)
    assert treedef_from_json(treedef_to_json(treedef)) == treedef


@tree.dataclass
class _Unkinded:
    x: int


def test_treedef_json_rejects_what_json_cannot_hold() -> None:
    point = namedtuple("point", ["x"])
    for value in [point(1), _Unkinded(1), {1: 1}, (object(),)]:
        _, treedef = tree.flatten(value, leaf=int)
        with pytest.raises(ArtifactError):
            treedef_to_json(treedef)


def test_malformed_section_is_rejected(tmp_path: Path) -> None:
    mef = ArtifactFile("model.mef", tmp_path / "model.mef")
    with pytest.raises(ArtifactError, match="mef is malformed"):
        CompiledCallable.__deserialize__({"mef": "model.mef"})
    with pytest.raises(ArtifactError, match="in_layouts is malformed"):
        CompiledCallable.__deserialize__({"mef": mef, "in_layouts": {}})


def test_malformed_layout_names_its_index(tmp_path: Path) -> None:
    good: dict[str, ArtifactType] = {
        "buffer": False,
        "dtype": "float32",
        "shape": [4],
        "mesh": {
            "devices": [{"type": "cpu", "id": 0}],
            "shape": [1],
            "axis_names": ["_"],
        },
        "placements": [{"type": "replicated"}],
    }
    bad: dict[str, ArtifactType] = {
        **good,
        "placements": [{"type": "scattered"}],
    }
    with pytest.raises(
        ArtifactError,
        match=r"in_layouts\[1\] is malformed: unknown placement 'scattered'",
    ):
        CompiledCallable.__deserialize__(
            {
                "mef": ArtifactFile("model.mef", tmp_path / "model.mef"),
                "in_layouts": [good, bad],
            }
        )
    missing: dict[str, ArtifactType] = {
        key: v for key, v in good.items() if key != "mesh"
    }
    with pytest.raises(
        ArtifactError, match=r"in_layouts\[0\] is malformed: missing field"
    ):
        CompiledCallable.__deserialize__(
            {
                "mef": ArtifactFile("model.mef", tmp_path / "model.mef"),
                "in_layouts": [missing],
            }
        )


def test_malformed_tree_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        ArtifactError,
        match="in_tree is malformed: unknown tree structure kind 'nope'",
    ):
        CompiledCallable.__deserialize__(
            {
                "mef": ArtifactFile("model.mef", tmp_path / "model.mef"),
                "in_layouts": [],
                "in_tree": {"kind": "nope"},
            }
        )
