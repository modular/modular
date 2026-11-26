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
"""Op log names for the two scopes a graph kernel launch is logged under."""

import re

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, LogLevel
from max.graph import DeviceRef, Graph, TensorType, ops

# [OP] LAUNCH matmul [id=9] target=cpu:0
_LAUNCH = re.compile(r"^\[OP\] LAUNCH (?P<name>\S+) \[id=\d+\]")


def _single_matmul_graph() -> Graph:
    with Graph(
        "matmul",
        input_types=[
            TensorType(DType.float32, (4, 64), device=DeviceRef.CPU()),
            TensorType(DType.float32, (64, 64), device=DeviceRef.CPU()),
        ],
    ) as graph:
        graph.output(ops.matmul(graph.inputs[0].tensor, graph.inputs[1].tensor))
    return graph


def _launch_names(captured: str) -> list[str]:
    return [
        match.group("name")
        for line in captured.splitlines()
        if (match := _LAUNCH.match(line))
    ]


def test_matmul_launch_names(
    capfd: pytest.CaptureFixture,  # type: ignore[type-arg]
) -> None:
    session = InferenceSession(devices=[CPU()])
    session.set_mojo_log_level(LogLevel.TRACE)
    model = session.load(_single_matmul_graph())

    model(
        Buffer.zeros((4, 64), DType.float32),
        Buffer.zeros((64, 64), DType.float32),
    )

    captured = capfd.readouterr()
    names = _launch_names(captured.out + captured.err)

    # One kernel, two nested op-level scopes: the graph op the compiler wraps
    # the kernel dispatch in, then the kernel's own trace.
    assert names.count("matmul") == 1, names
    assert names.count("matmul(mo.matmul)") == 1, names
