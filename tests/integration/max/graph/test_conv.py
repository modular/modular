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

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.ops import conv2d
from modular_graph_test import modular_graph_test

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


def torch_conv2d(
    x: TensorValue,
    filter: TensorValue,
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
):
    x = torch.permute(x, (0, 3, 1, 2))
    filter = torch.permute(filter, (3, 2, 0, 1))
    out = F.conv2d(
        x,
        filter,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return torch.permute(out, (0, 2, 3, 1))


# TODO(KERN-1066): Fix and enable test
@pytest.mark.skip(reason="Errors are larger than usual (10^-2)")
@pytest.mark.parametrize(
    "input_type, filter_type",
    [
        (
            TensorType(DType.float32, [1, 16, 16, 4], device=device_ref),
            TensorType(DType.float32, [16, 16, 4, 5], device=device_ref),
        ),
    ],
)
def test_conv2d(
    session: InferenceSession, input_type: TensorType, filter_type: TensorType
) -> None:
    with Graph("conv2d", input_types=[input_type, filter_type]) as graph:
        x, filter = graph.inputs
        stride = (16, 16)
        padding = (0, 0)
        dilation = (1, 1)

        conv = conv2d(x.tensor, filter.tensor, stride, dilation, (0, 0, 0, 0))
        graph.output(conv)

        @modular_graph_test(session, graph)
        def test_correctness(
            execute: Callable[[Sequence[Tensor]], Tensor],
            inputs: Sequence[Tensor],
            torch_inputs: Sequence[torch.Tensor],
        ) -> None:
            result = execute(inputs).to_numpy()
            x, w = torch_inputs
            expected = (
                torch_conv2d(x, w, stride, dilation, padding)
                .detach()
                .cpu()
                .numpy()
            )
            ACCURACY_RTOL = 1e-4
            ACCURACY_ATOL = 1e-6
            np.testing.assert_allclose(
                result,
                expected,
                equal_nan=True,
                rtol=ACCURACY_RTOL,
                atol=ACCURACY_ATOL,
            )
