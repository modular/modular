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

"""Tests reusing precompiled MEFs across sessions, including the guard.

The point of matching artifacts by path rather than by compile key is that a
divergence is loud, so the mismatch cases matter more here than the happy path:
they are what stops a caller from quietly reverting to compiling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType


def _graph(name: str = "add_one", *, width: int = 4) -> Graph:
    dtype = TensorType(DType.float32, [width], device=DeviceRef.CPU())
    with Graph(name, input_types=[dtype]) as graph:
        graph.output(graph.inputs[0].tensor + 1.0)
    return graph


def _execute(model: Any, width: int = 4) -> np.ndarray:
    outputs = model.execute(np.zeros(width, dtype=np.float32))
    return outputs[0].to_numpy()


def test_exports_then_reuses_without_recompiling(tmp_path: Path) -> None:
    exported = InferenceSession(devices=[CPU()], export_mefs=tmp_path).load(
        _graph()
    )
    np.testing.assert_allclose(_execute(exported), np.ones(4))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [entry["name"] for entry in manifest["graphs"]] == ["add_one"]
    assert (tmp_path / manifest["graphs"][0]["key"]).is_file()

    reused = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path).load(
        _graph()
    )
    np.testing.assert_allclose(_execute(reused), np.ones(4))


def test_reusing_a_differently_shaped_graph_raises(tmp_path: Path) -> None:
    InferenceSession(devices=[CPU()], export_mefs=tmp_path).load(
        _graph(width=4)
    )

    # A shape divergence is what a pipeline sizing itself from device memory
    # produces, and it must not be papered over with a stale artifact.
    session = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)
    with pytest.raises(RuntimeError, match="does not match the precompiled"):
        session.load(_graph(width=8))


def test_reusing_a_renamed_graph_raises(tmp_path: Path) -> None:
    InferenceSession(devices=[CPU()], export_mefs=tmp_path).load(
        _graph("add_one")
    )

    session = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)
    with pytest.raises(RuntimeError, match="does not match the precompiled"):
        session.load(_graph("something_else"))


def test_compiling_more_graphs_than_were_exported_raises(
    tmp_path: Path,
) -> None:
    InferenceSession(devices=[CPU()], export_mefs=tmp_path).load(_graph())

    session = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)
    session.load(_graph())
    with pytest.raises(RuntimeError, match="only 1 were precompiled"):
        session.load(_graph())


def test_reusing_a_directory_with_no_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no manifest\.json"):
        InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)


def test_exporting_and_reusing_at_once_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most one of"):
        InferenceSession(
            devices=[CPU()],
            precompiled_mefs=tmp_path,
            export_mefs=tmp_path,
        )


def test_is_inert_by_default() -> None:
    model = InferenceSession(devices=[CPU()]).load(_graph())
    np.testing.assert_allclose(_execute(model), np.ones(4))
