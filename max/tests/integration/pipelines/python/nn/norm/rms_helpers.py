# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.abc import Callable, Sequence

import torch
from max.driver import Tensor
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.nn import RMSNormV1
from modular_graph_test import are_all_tensor_values, modular_graph_test

SHAPES = (
    ["dim"],
    ["batch", "dim"],
    ["a", "x", "y", "z", "dim"],
)


def torch_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    #   See https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
    return (
        x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)
    ).type_as(x) * weight


def run_test_norm(
    session: InferenceSession,
    input_type: TensorType,
    rtol: float,
    atol: float,
) -> None:
    # Initialize Graph
    dim = input_type.shape[-1]
    weight_type = TensorType(input_type.dtype, [dim], device=input_type.device)
    with Graph("norm", input_types=[input_type, weight_type]) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, weight = graph.inputs
        graph.output(RMSNormV1(weight=weight, multiply_before_cast=False)(x))

        @modular_graph_test(session, graph)
        def test_correctness(
            execute: Callable[[Sequence[Tensor]], Tensor],
            inputs: Sequence[Tensor],
            torch_inputs: Sequence[torch.Tensor],
        ) -> None:
            result = torch.from_dlpack(execute(inputs))
            expected = torch_rms_norm(*torch_inputs)
            torch.testing.assert_close(
                result,
                expected,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                check_device=False,
                msg=lambda msg: f"\n{result=}\n{expected=}\n",
            )
