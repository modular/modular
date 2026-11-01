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
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from shared_conv_impl import (
    conv1d_impl,
    conv1d_tuple_padding_impl,
    conv1d_tuple_padding_nonfcrs_impl,
    conv2d_1x1_impl,
    conv2d_impl,
    conv2d_tuple_padding_impl,
    conv3d_impl,
    conv3d_tuple_padding_impl,
)
from test_common.numerics import pytorch_disable_tf32_dtype


@pytorch_disable_tf32_dtype
def test_conv3d_gpu(gpu_session: InferenceSession) -> None:
    conv3d_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv1d_gpu(gpu_session: InferenceSession) -> None:
    conv1d_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv2d_1x1_gpu(gpu_session: InferenceSession) -> None:
    conv2d_1x1_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv2d_gpu(gpu_session: InferenceSession) -> None:
    conv2d_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv2d_tuple_padding_gpu(gpu_session: InferenceSession) -> None:
    conv2d_tuple_padding_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv1d_tuple_padding_gpu(gpu_session: InferenceSession) -> None:
    conv1d_tuple_padding_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv1d_tuple_padding_nonfcrs_gpu(
    gpu_session: InferenceSession,
) -> None:
    conv1d_tuple_padding_nonfcrs_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv3d_tuple_padding_gpu(gpu_session: InferenceSession) -> None:
    conv3d_tuple_padding_impl(gpu_session)


def _check_conv(
    session: InferenceSession,
    dtype: DType,
    input_shape: tuple[int, ...],
    filter_shape: tuple[int, ...],
    stride: int,
    padding: int,
) -> None:
    """Checks a conv against torch, with an add fused into it.

    The add fuses into the conv as its epilogue. The fused conv has no output
    buffer of its own, so every value reaches the graph output only through
    the epilogue's stores.
    """
    rank = len(input_shape) - 2
    device = DeviceRef.GPU()
    with Graph(
        f"conv{rank}d_fused_epilogue",
        input_types=(
            TensorType(dtype, input_shape, device=device),
            TensorType(dtype, filter_shape, device=device),
        ),
    ) as graph:
        x, w = (value.tensor for value in graph.inputs)
        p = padding
        if rank == 2:
            out = ops.conv2d(
                x, w, stride=(stride, stride), padding=(p, p, p, p)
            )
        else:
            out = ops.conv3d(
                x,
                w,
                stride=(stride, stride, stride),
                padding=(p, p, p, p, p, p),
            )
        graph.output(out + 1.0)
    compiled = session.load(graph)

    torch_dtype = torch.float32 if dtype == DType.float32 else torch.bfloat16
    torch.manual_seed(0)
    x_host = torch.rand(input_shape, dtype=torch_dtype)
    w_host = torch.rand(filter_shape, dtype=torch_dtype)
    result = compiled.execute(
        Buffer.from_dlpack(x_host).to(session.devices[0]),
        Buffer.from_dlpack(w_host).to(session.devices[0]),
    )[0]
    assert isinstance(result, Buffer)
    actual = torch.from_dlpack(result).to("cpu").to(torch.float32)

    # MAX takes channels-last input and an RS...CF filter; torch takes
    # channels-first input and an FC...RS filter.
    to_channels_first = (0, rank + 1, *range(1, rank + 1))
    to_fc_rs = (rank + 1, rank, *range(rank))
    torch_conv = (
        torch.nn.functional.conv2d if rank == 2 else torch.nn.functional.conv3d
    )
    expected = torch_conv(
        x_host.to(torch.float32).permute(to_channels_first),
        w_host.to(torch.float32).permute(to_fc_rs),
        stride=stride,
        padding=padding,
    )
    expected = expected.permute(0, *range(2, rank + 2), 1)
    expected = expected + 1.0

    if dtype == DType.float32:
        torch.testing.assert_close(actual, expected, rtol=2.5e-3, atol=1e-3)
    else:
        torch.testing.assert_close(
            actual,
            expected.to(torch.bfloat16).to(torch.float32),
            rtol=2e-2,
            atol=1e-1,
        )


@pytest.mark.parametrize(
    ("dtype", "stride"),
    [(DType.float32, 1), (DType.bfloat16, 1)],
)
@pytorch_disable_tf32_dtype
def test_conv2d_fused_epilogue_gpu(
    gpu_session: InferenceSession, dtype: DType, stride: int
) -> None:
    _check_conv(gpu_session, dtype, (1, 16, 16, 64), (3, 3, 64, 64), stride, 1)


@pytest.mark.parametrize(
    ("dtype", "kernel"),
    [(DType.bfloat16, 3), (DType.bfloat16, 1)],
)
@pytorch_disable_tf32_dtype
def test_conv3d_fused_epilogue_gpu(
    gpu_session: InferenceSession, dtype: DType, kernel: int
) -> None:
    _check_conv(
        gpu_session,
        dtype,
        (1, 8, 8, 8, 32),
        (kernel, kernel, kernel, 32, 32),
        1,
        kernel // 2,
    )
