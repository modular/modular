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

import re

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType, ops

_K = 256
_H = 256


def _graph(*, bias: bool, sliced: bool, constant_expr: bool = False) -> Graph:
    device = DeviceRef.GPU()
    width = _K + 1 if sliced else _K
    with Graph(
        "matmul_swiglu_alignment",
        input_types=[TensorType(DType.bfloat16, [Dim("m"), width], device)],
    ) as graph:
        x = graph.inputs[0].tensor
        if sliced:
            x = ops.slice_tensor(x, [slice(None), slice(1, _K + 1)])
        weight = ops.constant_external(
            "weight", TensorType(DType.bfloat16, [2 * _H, _K], device), align=16
        )
        if constant_expr:
            weight = weight + weight
        y = ops.matmul(x, ops.transpose(weight, 0, 1))
        if bias:
            b = torch.cat([torch.full((_H,), 0.25), torch.full((_H,), 0.5)]).to(
                torch.bfloat16
            )
            y = y + ops.constant(b, device=device)
        gate = ops.slice_tensor(y, [slice(None), slice(0, _H)])
        up = ops.slice_tensor(y, [slice(None), slice(_H, 2 * _H)])
        graph.output(ops.silu(gate) * up)
    return graph


def _reference(
    x: torch.Tensor, weight: torch.Tensor, bias: bool
) -> torch.Tensor:
    gate, up = (x.float() @ weight.float().T).split(_H, dim=-1)
    if bias:
        gate = gate + 0.25
        up = up + 0.5
    return torch.nn.functional.silu(gate) * up


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("input_kind", ["aligned", "offset", "graph_slice"])
def test_matmul_swiglu_inputs(bias: bool, input_kind: str) -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    generator = torch.Generator().manual_seed(42)
    weight_cpu = (torch.randn((2 * _H, _K), generator=generator) / _K**0.5).to(
        torch.bfloat16
    )
    weight = Buffer.from_dlpack(weight_cpu).to(device)
    model = session.load(
        _graph(bias=bias, sliced=input_kind == "graph_slice"),
        weights_registry={"weight": weight},
    )
    expected_op = (
        "mo.composite.matmul_swiglu_bias"
        if bias
        else "mo.composite.matmul_swiglu"
    )
    assert any(
        re.search(rf"{re.escape(expected_op)}\b", s)
        for s in model.kernel_summaries
    ), model.kernel_summaries

    width = _K + 1 if input_kind != "aligned" else _K
    values = torch.linspace(-1, 1, _K).to(torch.bfloat16)
    base_values = (
        torch.cat((torch.zeros(1, dtype=torch.bfloat16), values))
        if input_kind != "aligned"
        else values
    )
    base = Buffer.from_dlpack(base_values).to(device)
    if input_kind == "offset":
        x = base[1:].view(DType.bfloat16, [1, _K])
        assert torch.from_dlpack(x).data_ptr() % 16 != 0
    else:
        x = base.view(DType.bfloat16, [1, width])

    actual = torch.from_dlpack(model.execute(x)[0]).cpu().float()
    expected = _reference(values.reshape(1, _K), weight_cpu, bias)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

    if input_kind == "offset":
        (captured,) = model.capture(0, x)
        updated = (-values).reshape(1, _K)
        x.inplace_copy_from(Buffer.from_dlpack(updated).to(device))
        model.replay(0, x)
        actual = torch.from_dlpack(captured).cpu().float()
        expected = _reference(updated, weight_cpu, bias)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

        second_values = (values * 0.5).to(torch.bfloat16)
        second_base = Buffer.from_dlpack(
            torch.cat((torch.zeros(1, dtype=torch.bfloat16), second_values))
        ).to(device)
        second_x = second_base[1:].view(DType.bfloat16, [1, _K])
        (second_output,) = model.capture(1, second_x)
        model.replay(1, second_x)
        second_expected = _reference(
            second_values.reshape(1, _K), weight_cpu, bias
        )
        second_actual = torch.from_dlpack(second_output).cpu().float()
        torch.testing.assert_close(
            second_actual, second_expected, rtol=0.02, atol=0.02
        )
        model.replay(0, x)
        second_actual = torch.from_dlpack(second_output).cpu().float()
        torch.testing.assert_close(
            second_actual, second_expected, rtol=0.02, atol=0.02
        )

        rows = torch.stack(
            (values, -values, values * 0.5, values * 0.25, values * 0.75)
        ).to(torch.bfloat16)
        rows_base = Buffer.from_dlpack(
            torch.cat((torch.zeros(1, dtype=torch.bfloat16), rows.flatten()))
        ).to(device)
        rows_input = rows_base[1:].view(DType.bfloat16, [5, _K])
        actual = torch.from_dlpack(model.execute(rows_input)[0]).cpu().float()
        torch.testing.assert_close(
            actual, _reference(rows, weight_cpu, bias), rtol=0.02, atol=0.02
        )

    if input_kind == "aligned":
        empty = Buffer.from_dlpack(
            torch.empty((0, _K), dtype=torch.bfloat16)
        ).to(device)
        result = model.execute(empty)[0]
        assert result.shape == (0, _H)


def test_matmul_swiglu_constant_expression_weight() -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    generator = torch.Generator().manual_seed(43)
    weight_cpu = (torch.randn((2 * _H, _K), generator=generator) / _K**0.5).to(
        torch.bfloat16
    )
    weight = Buffer.from_dlpack(weight_cpu).to(device)
    model = session.load(
        _graph(bias=False, sliced=False, constant_expr=True),
        weights_registry={"weight": weight},
    )
    assert any(
        "mo.composite.matmul_swiglu" in summary
        for summary in model.kernel_summaries
    ), model.kernel_summaries

    x = torch.linspace(-1, 1, _K).to(torch.bfloat16).reshape(1, _K)
    actual = (
        torch.from_dlpack(model.execute(Buffer.from_dlpack(x).to(device))[0])
        .cpu()
        .float()
    )
    expected = _reference(x, weight_cpu + weight_cpu, False)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
