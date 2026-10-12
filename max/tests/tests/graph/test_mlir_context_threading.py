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
"""Tests that MLIR contexts compile on the AsyncRT pool, not their own."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from max import mlir
from max._mlir_context import default_mlir_context
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def _thread_names() -> list[str]:
    return [
        (task / "comm").read_text().strip()
        for task in Path("/proc/self/task").iterdir()
    ]


def test_new_context_has_no_thread_pool() -> None:
    assert mlir.Context().get_num_threads() == 1


@pytest.mark.skipif(
    sys.platform != "linux", reason="reads thread names from /proc"
)
def test_compile_runs_on_runtime_thread_pool() -> None:
    session = InferenceSession(devices=[CPU()])
    graph = Graph(
        "add",
        forward=lambda x: ops.add(x, x),
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.CPU())],
    )
    session.load(graph)

    names = _thread_names()
    assert not [name for name in names if name.startswith("llvm-worker")]
    runtime_workers = [name for name in names if name.startswith("🔥 Thread")]
    assert default_mlir_context().get_num_threads() == len(runtime_workers)
