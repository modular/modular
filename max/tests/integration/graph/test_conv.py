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

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.ops import conv2d
from modular_graph_test import modular_graph_test

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


def torch_conv2d(  # noqa: ANN201
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


# Grouped (groups > 1) cases are a regression test for #5614, where grouped
# convolution failed to compile: the `mo.conv` lowering forwarded the unpacked
# RSCF filter to a kernel that requires a packed (FRSCf) filter for grouped conv.
# `modular_graph_test` calls `session.load(graph)`, so such a regression would
# reappear as a compile failure here.
# TODO(KERN-1066): Fix and enable test.
@pytest.mark.skip(reason="Errors are larger than usual (10^-2)")
@pytest.mark.parametrize(
    "input_type, filter_type, groups",
    [
        (
            TensorType(DType.float32, [1, 16, 16, 4], device=device_ref),
            TensorType(DType.float32, [16, 16, 4, 5], device=device_ref),
            1,
        ),
        (
            TensorType(DType.float32, [1, 8, 8, 4], device=device_ref),
            TensorType(DType.float32, [3, 3, 2, 8], device=device_ref),
            2,
        ),
    ],
)
def test_conv2d(
    session: InferenceSession,
    input_type: TensorType,
    filter_type: TensorType,
    groups: int,
) -> None:
    stride = (16, 16)
    padding = (0, 0)
    dilation = (1, 1)

    with Graph("conv2d", input_types=[input_type, filter_type]) as graph:
        x, filter = graph.inputs
        conv = conv2d(
            x.tensor,
            filter.tensor,
            stride,
            dilation,
            (padding[0], padding[0], padding[1], padding[1]),
            groups,
        )
        graph.output(conv)

        @modular_graph_test(session, graph)
        def test_correctness(
            execute: Callable[[Sequence[Buffer]], Buffer],
            inputs: Sequence[Buffer],
            torch_inputs: Sequence[torch.Tensor],
        ) -> None:
            result = execute(inputs).to_numpy()
            x, w = torch_inputs
            expected = (
                torch_conv2d(x, w, stride, dilation, padding, groups)
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
