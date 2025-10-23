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

from typing import Any

import numpy as np
import pytest
import torch.utils.dlpack
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.ops import div, floor


def build_models(lhs_dtype: DType, rhs_dtype: DType) -> Graph:
    lhs_type = TensorType(lhs_dtype, [], device=DeviceRef.CPU())
    rhs_type = TensorType(rhs_dtype, [], device=DeviceRef.CPU())
    with Graph("div_and_floor", input_types=[lhs_type, rhs_type]) as g:
        q = div(g.inputs[0].tensor, g.inputs[1].tensor)
        g.output(q, floor(q))
    return g


def run_models(
    model: Any,
    lhs_val: int | float,
    rhs_val: int | float,
    lhs_dtype: DType,
    rhs_dtype: DType,
    cpu: Device,
) -> tuple[float, float]:
    lhs = Tensor.from_numpy(np.array(lhs_val, dtype=lhs_dtype.to_numpy())).to(
        cpu
    )
    rhs = Tensor.from_numpy(np.array(rhs_val, dtype=rhs_dtype.to_numpy())).to(
        cpu
    )
    out_div, out_floor = model(lhs, rhs)

    def to_scalar(x: Any) -> float:
        if isinstance(x, Tensor):
            return x.to_numpy().item()
        return torch.utils.dlpack.from_dlpack(x).numpy().item()

    return to_scalar(out_div), to_scalar(out_floor)


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype,cases",
    [
        (
            DType.int32,
            DType.int32,
            [(7, 3), (7, -3), (-7, 3), (-7, -3), (1, 2), (0, 5)],
        ),
        (DType.int64, DType.int64, [(15, 4), (np.iinfo(np.int64).min, -1)]),
        (
            DType.float32,
            DType.float32,
            [(7.5, 2.5), (1.0, 3.0), (-0.0, 3.0)],
        ),
        (DType.float64, DType.float64, [(7.0, 3.0), (1.0, 3.0)]),
    ],
)
def test_div_and_floordiv_match_python(
    lhs_dtype: DType,
    rhs_dtype: DType,
    cases: list[tuple[int | float, int | float]],
) -> None:
    cpu = CPU()
    session = InferenceSession(devices=[cpu])
    graph = build_models(lhs_dtype, rhs_dtype)
    model = session.load(graph)

    for lhs_val, rhs_val in cases:
        py_div = lhs_val / rhs_val
        py_floor = lhs_val // rhs_val

        max_div, max_floor = run_models(
            model, lhs_val, rhs_val, lhs_dtype, rhs_dtype, cpu
        )

        if DType.is_float(lhs_dtype) or DType.is_float(rhs_dtype):
            if lhs_dtype == DType.float64 or rhs_dtype == DType.float64:
                rtol, atol = 1e-12, 0.0
            else:
                rtol, atol = 1e-6, 1e-7
            np.testing.assert_allclose(
                max_div,
                py_div,
                rtol=rtol,
                atol=atol,
                err_msg=f"div({lhs_val}, {rhs_val}) mismatch",
            )
            np.testing.assert_allclose(
                max_floor,
                py_floor,
                rtol=rtol,
                atol=atol,
                err_msg=f"floor(div({lhs_val}, {rhs_val})) mismatch",
            )
        else:
            np.testing.assert_equal(max_div, py_div)
            np.testing.assert_equal(max_floor, py_floor)
