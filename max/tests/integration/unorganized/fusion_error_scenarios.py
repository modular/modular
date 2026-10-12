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
"""Shared GPU crash scenario definitions for fusion error message tests.

Each scenario builds a MAX graph with a specific fusion pattern and an
intentional GPU crash op.  The graph and numpy input arrays are returned
so that callers can run them in their own way (pytest assertions vs.
human-readable diff output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops

SCENARIO_NAMES = [
    "single_crash",
    "elementwise_chain_crash",
    "matmul_epilogue_crash",
    "many_fused_crash",
    "big_matmul_fusion",
    "two_matmuls_crash",
    "sqrt_negative",
]

SCENARIO_DESCRIPTIONS = {
    "single_crash": "Single crash op (baseline)",
    "elementwise_chain_crash": (
        "add -> mul -> relu + crash (3 fused elementwise ops)"
    ),
    "matmul_epilogue_crash": (
        "matmul -> add(bias) -> relu -> mul + crash (epilogue fusion)"
    ),
    "many_fused_crash": (
        "add->mul->relu->add->mul->abs->neg->add + crash (8 fused ops)"
    ),
    "big_matmul_fusion": "matmul + 6 epilogues + crash (large fusion tree)",
    "two_matmuls_crash": ("Two sequential matmuls + crash (per-stub tracking)"),
    "sqrt_negative": "sqrt(negative) in fused kernel",
}

f32 = DType.float32


def _crash_op(graph: Graph, x: TensorValue, shape: list[int]) -> TensorValue:
    """Add an intentional_gpu_crash custom op."""
    return ops.custom(
        name="intentional_gpu_crash",
        device=DeviceRef.GPU(),
        values=[x],
        out_types=[TensorType(f32, shape, device=DeviceRef.GPU())],
    )[0].tensor


def build_scenario(
    scenario_name: str, kernel_ops_path: Path
) -> tuple[Graph, list[np.ndarray]]:
    """Build a graph for the named scenario.

    Returns the compiled graph and the list of numpy input arrays needed
    to execute it.
    """
    if scenario_name == "single_crash":
        with Graph(
            "single_crash",
            input_types=[TensorType(f32, [10], device=DeviceRef.GPU())],
        ) as graph:
            graph._import_kernels([kernel_ops_path])
            x = graph.inputs[0].tensor
            graph.output(_crash_op(graph, x, [10]))
        return graph, [np.zeros(10, dtype=np.float32)]

    elif scenario_name == "elementwise_chain_crash":
        with Graph(
            "ewise_chain_crash",
            input_types=[
                TensorType(f32, [10], device=DeviceRef.GPU()),
                TensorType(f32, [10], device=DeviceRef.GPU()),
            ],
        ) as graph:
            graph._import_kernels([kernel_ops_path])
            x, y = graph.inputs[0].tensor, graph.inputs[1].tensor
            activated = ops.relu(ops.mul(ops.add(x, y), ops.add(x, y)))
            graph.output(_crash_op(graph, activated, [10]))
        return graph, [
            np.ones(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
        ]

    elif scenario_name == "matmul_epilogue_crash":
        with Graph(
            "matmul_epi_crash",
            input_types=[
                TensorType(f32, [20, 30], device=DeviceRef.GPU()),
                TensorType(f32, [30, 40], device=DeviceRef.GPU()),
                TensorType(f32, [1, 40], device=DeviceRef.GPU()),
            ],
        ) as graph:
            graph._import_kernels([kernel_ops_path])
            a, b, bias = (inp.tensor for inp in graph.inputs)
            mm = ops.matmul(a, b)
            scaled = ops.mul(
                ops.relu(ops.add(mm, bias)), ops.relu(ops.add(mm, bias))
            )
            graph.output(_crash_op(graph, scaled.reshape([800]), [800]))
        return graph, [
            np.random.randn(*s).astype(np.float32)
            for s in [(20, 30), (30, 40), (1, 40)]
        ]

    elif scenario_name == "many_fused_crash":
        with Graph(
            "many_fused_crash",
            input_types=[
                TensorType(f32, [10], device=DeviceRef.GPU()),
                TensorType(f32, [10], device=DeviceRef.GPU()),
                TensorType(f32, [10], device=DeviceRef.GPU()),
            ],
        ) as graph:
            graph._import_kernels([kernel_ops_path])
            x, y, z = (inp.tensor for inp in graph.inputs)
            a = ops.add(x, y)
            b = ops.mul(a, z)
            c = ops.relu(b)
            d = ops.add(c, x)
            e = ops.mul(d, y)
            f = ops.abs(e)
            g = ops.mul(f, ops.constant(-1.0, f32, DeviceRef.GPU()))
            h = ops.add(g, z)
            graph.output(_crash_op(graph, h, [10]))
        return graph, [np.ones(10, dtype=np.float32)] * 3

    elif scenario_name == "big_matmul_fusion":
        with Graph(
            "big_matmul_fusion",
            input_types=[
                TensorType(f32, [20, 30], device=DeviceRef.GPU()),
                TensorType(f32, [30, 40], device=DeviceRef.GPU()),
                TensorType(f32, [1, 40], device=DeviceRef.GPU()),
                TensorType(f32, [1, 40], device=DeviceRef.GPU()),
                TensorType(f32, [1, 40], device=DeviceRef.GPU()),
            ],
        ) as graph:
            graph._import_kernels([kernel_ops_path])
            a, b, bias, scale, offset = (inp.tensor for inp in graph.inputs)
            mm = ops.matmul(a, b)
            fp32 = ops.cast(
                ops.cast(
                    ops.mul(
                        ops.add(ops.relu(ops.add(mm, bias)), offset), scale
                    ),
                    DType.float16,
                ),
                f32,
            )
            graph.output(_crash_op(graph, fp32.reshape([800]), [800]))
        return graph, [
            np.random.randn(*s).astype(np.float32)
            for s in [(20, 30), (30, 40), (1, 40), (1, 40), (1, 40)]
        ]

    elif scenario_name == "two_matmuls_crash":
        with Graph(
            "two_matmuls",
            input_types=[
                TensorType(f32, [20, 30], device=DeviceRef.GPU()),
                TensorType(f32, [30, 40], device=DeviceRef.GPU()),
                TensorType(f32, [40, 50], device=DeviceRef.GPU()),
            ],
        ) as graph:
            graph._import_kernels([kernel_ops_path])
            a, b, c = (inp.tensor for inp in graph.inputs)
            act2 = ops.relu(ops.matmul(ops.relu(ops.matmul(a, b)), c))
            graph.output(_crash_op(graph, act2.reshape([1000]), [1000]))
        return graph, [
            np.random.randn(*s).astype(np.float32)
            for s in [(20, 30), (30, 40), (40, 50)]
        ]

    elif scenario_name == "sqrt_negative":
        with Graph(
            "sqrt_neg",
            input_types=[
                TensorType(f32, [10], device=DeviceRef.GPU()),
                TensorType(f32, [10], device=DeviceRef.GPU()),
            ],
        ) as graph:
            x, y = graph.inputs[0].tensor, graph.inputs[1].tensor
            diff = ops.sub(x, y)
            graph.output(ops.sqrt(ops.sub(ops.mul(diff, diff), y)))
        return graph, [
            np.zeros(10, dtype=np.float32),
            np.full(10, 0.1, dtype=np.float32),
        ]

    else:
        raise ValueError(f"Unknown scenario: {scenario_name!r}")
