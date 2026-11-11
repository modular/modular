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
"""Tests the kernel analysis that KernelLibrary shares between graphs."""

import os
from pathlib import Path

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def _graph_with_builtin_custom_op(name: str, path: Path | None = None) -> Graph:
    """Builds a graph that adds one custom op from the built-in kernels."""
    offsets = TensorType(DType.uint32, ["rows"], DeviceRef.CPU())
    lengths = TensorType(DType.uint32, ["batch"], DeviceRef.CPU())
    with Graph(name, input_types=[offsets, lengths]) as graph:
        if path is not None:
            graph._kernel_library.add_path(path)
        ops.custom(
            "kv_cache.row_offsets.ragged.paged",
            device=DeviceRef.CPU(),
            values=list(graph.inputs),
            out_types=[offsets],
        )
    return graph


def test_graphs_without_extra_paths_share_an_analysis() -> None:
    first = _graph_with_builtin_custom_op("first")
    second = _graph_with_builtin_custom_op("second")

    assert first.kernel_libraries_paths == []
    assert first._kernel_library._analysis is second._kernel_library._analysis


def test_adding_a_path_leaves_the_shared_analysis_alone() -> None:
    first = _graph_with_builtin_custom_op("first")
    second = _graph_with_builtin_custom_op("second")
    path = Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])

    second._kernel_library.add_path(path)

    assert first.kernel_libraries_paths == []
    assert len(second.kernel_libraries_paths) == 1
    assert "my_add" in second._kernel_library
    assert "my_add" not in first._kernel_library


def test_adding_a_loaded_path_keeps_the_shared_analysis() -> None:
    path = Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])
    graph = _graph_with_builtin_custom_op("loaded", path)
    shared = graph._kernel_library._analysis

    graph._kernel_library.add_path(path)

    assert graph._kernel_library._analysis is shared


def test_a_failed_verify_leaves_the_shared_analysis_alone() -> None:
    first = _graph_with_builtin_custom_op("first")
    path = Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])
    tensor = TensorType(DType.float32, [2], DeviceRef.CPU())

    with Graph("missing", input_types=[tensor, tensor]) as graph:
        with pytest.raises(ValueError):
            ops.custom(
                "my_add",
                device=DeviceRef.CPU(),
                values=list(graph.inputs),
                out_types=[tensor],
            )
        graph._kernel_library.add_path(path)

    assert "my_add" in graph._kernel_library
    assert "my_add" not in first._kernel_library


def test_reading_symbols_shares_the_analysis() -> None:
    first = _graph_with_builtin_custom_op("first")
    with Graph("reader") as graph:
        assert "kv_cache.row_offsets.ragged.paged" in graph._kernel_library

    assert graph._kernel_library._analysis is first._kernel_library._analysis


def test_a_bad_path_raises_when_added_to_a_shared_analysis(
    tmp_path: Path,
) -> None:
    graph = _graph_with_builtin_custom_op("bad")
    shared = graph._kernel_library._analysis
    bad = tmp_path / "bad.mojopkg"
    bad.write_bytes(b"not a package")

    with pytest.raises(ValueError):
        graph._kernel_library.add_path(bad)

    assert graph._kernel_library._analysis is shared
    assert graph.kernel_libraries_paths == []
