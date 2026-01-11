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
"""Smoke tests for ops in `max.experimental.functional`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

from typing import cast

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef

DEVICE = Accelerator() if accelerator_count() else CPU()


def test_allgather() -> None:
    input = Tensor.ones([2, 4], dtype=DType.float32, device=DEVICE)
    signal_buffer = Tensor.zeros([], device=DEVICE)
    [result] = F.allgather([input], [signal_buffer])
    result._sync_realize()
    assert result.real


def test_allreduce() -> None:
    input = Tensor.ones([2, 4], dtype=DType.float32, device=DEVICE)
    signal_buffer = Tensor.zeros([], device=DEVICE)
    [result] = F.allreduce_sum([input], [signal_buffer])
    result._sync_realize()
    assert result.real


def test_as_interleaved_complex() -> None:
    # needs even last dimension
    complex_input = Tensor.ones([2, 4], dtype=DType.float32, device=DEVICE)
    result = F.as_interleaved_complex(complex_input)
    result._sync_realize()
    assert result.real


def test_avg_pool2d() -> None:
    # needs 4D input with NHWC format
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    result = F.avg_pool2d(tensor_4d, kernel_size=(2, 2), stride=1)
    result._sync_realize()
    assert result.real


def test_band_part() -> None:
    # needs at least 2D tensor
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.band_part(tensor_2d, num_lower=-1, num_upper=0)
    result._sync_realize()
    assert result.real


def test_broadcast_to() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.broadcast_to(tensor_2d, [4, 6])
    result._sync_realize()
    assert result.real


def test_cast() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.cast(tensor_2d, DType.int64)
    result._sync_realize()
    assert result.real
    assert result.dtype == DType.int64


def test_chunk() -> None:
    # split into 2 chunks along axis 0
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    chunks = F.chunk(tensor_2d, chunks=2, axis=0)
    for chunk in chunks:
        chunk._sync_realize()
        assert chunk.real


def test_complex_mul() -> None:
    lhs = Tensor.ones([2, 2], dtype=DType.float32, device=DEVICE)
    rhs = Tensor.ones([2, 1, 2], dtype=DType.float32, device=DEVICE)
    result = F.complex_mul(lhs, rhs)
    assert result.shape == [2, 2, 2]
    result._sync_realize()
    assert result.real


def test_concat() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.concat([tensor_2d, tensor_2d], axis=0)
    result._sync_realize()
    assert result.real
    assert result.shape.static_dims == [8, 6]


def test_constant() -> None:
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.constant(1.0, DType.float32, device_ref)
    result._sync_realize()
    assert result.real


def test_conv2d() -> None:
    # NHWC input: [batch, height, width, in_channels]
    # RSCF filter: [height, width, in_channels/groups, out_channels]
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    weight = Tensor.ones(
        [3, 3, 2, 1], dtype=DType.float32, device=DEVICE
    )  # [H, W, in_ch, out_ch]
    result = F.conv2d(tensor_4d, weight)
    result._sync_realize()
    assert result.real


@pytest.mark.skip("KERNELS-1975")
def test_conv2d_transpose() -> None:
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    weight = Tensor.ones(
        [3, 3, 1, 2], dtype=DType.float32, device=DEVICE
    )  # [H, W, out_ch, in_ch]
    result = F.conv2d_transpose(tensor_4d, weight)
    result._sync_realize()
    assert result.real


def test_conv3d() -> None:
    # NDHWC input: [batch, depth, height, width, in_channels]
    # QRSCF filter: [depth, height, width, in_channels/groups, out_channels]
    tensor_5d = Tensor.ones(
        [1, 2, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, D, H, W, C]
    weight_3d = Tensor.ones(
        [2, 3, 3, 2, 1], dtype=DType.float32, device=DEVICE
    )  # [D, H, W, in_ch, out_ch]
    result = F.conv3d(tensor_5d, weight_3d)
    result._sync_realize()
    assert result.real


def test_flatten() -> None:
    tensor_3d = Tensor.ones([2, 3, 4], dtype=DType.float32, device=DEVICE)
    result = F.flatten(tensor_3d, start_dim=1, end_dim=2)
    result._sync_realize()
    assert result.real


def test_fold() -> None:
    # needs shape [N, C * kernel_size[0] * kernel_size[1], L]
    # For kernel_size=[2, 2], we need C * 4 channels
    kernel_size = [2, 2]
    tensor_3d = Tensor.ones(
        [1, 4, 4], dtype=DType.float32, device=DEVICE
    )  # [N, C*kernel_prod, L]
    result = F.fold(tensor_3d, output_size=[3, 3], kernel_size=kernel_size)
    result._sync_realize()
    assert result.real


def test_gather() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices = Tensor.full([2], 0, dtype=DType.int64, device=DEVICE)
    result = F.gather(tensor_2d, indices, axis=0)
    result._sync_realize()
    assert result.real


def test_gather_nd() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices_nd = Tensor.full([2, 2], 0, dtype=DType.int64, device=DEVICE)
    result = F.gather_nd(tensor_2d, indices_nd)
    result._sync_realize()
    assert result.real


def test_hann_window() -> None:
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.hann_window(4, device=device_ref)
    result._sync_realize()
    assert result.real


@pytest.mark.skipif(
    isinstance(DEVICE, CPU), reason="IRFFT only supported on GPU"
)
def test_irfft() -> None:
    tensor_2d = Tensor.ones([4, 8], dtype=DType.float32, device=DEVICE)
    result = F.irfft(tensor_2d, n=14, axis=-1)
    result._sync_realize()
    assert result.real


def test_layer_norm() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    gamma = Tensor.ones(
        [6], dtype=DType.float32, device=DEVICE
    )  # normalization weights
    beta = Tensor.zeros(
        [6], dtype=DType.float32, device=DEVICE
    )  # normalization bias
    result = F.layer_norm(tensor_2d, gamma, beta, epsilon=1e-5)
    result._sync_realize()
    assert result.real


def test_masked_scatter() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    mask = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    source = Tensor.ones([24], dtype=DType.float32, device=DEVICE)
    result = F.masked_scatter(tensor_2d, mask, source, out_dim=24)
    result._sync_realize()
    assert result.real


def test_matmul() -> None:
    a = Tensor.ones([4, 3], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([3, 6], dtype=DType.float32, device=DEVICE)
    result = F.matmul(a, b)
    result._sync_realize()
    assert result.real


def test_max_pool2d() -> None:
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    result = F.max_pool2d(tensor_4d, kernel_size=(2, 2), stride=1)
    result._sync_realize()
    assert result.real


def test_nonzero() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.nonzero(
        tensor_2d, out_dim=24
    )  # assuming all elements are nonzero
    result._sync_realize()
    assert result.real


def test_outer() -> None:
    vec_a = Tensor.ones([3], dtype=DType.float32, device=DEVICE)
    vec_b = Tensor.ones([4], dtype=DType.float32, device=DEVICE)
    result = F.outer(vec_a, vec_b)
    result._sync_realize()
    assert result.real


def test_pad() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.pad(tensor_2d, [1, 1, 2, 2])  # pad left, right, top, bottom
    result._sync_realize()
    assert result.real


def test_permute() -> None:
    tensor_3d = Tensor.ones([2, 3, 4], dtype=DType.float32, device=DEVICE)
    result = F.permute(tensor_3d, [2, 0, 1])
    result._sync_realize()
    assert result.real


def test_arange() -> None:
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.arange(0, 10, 1, dtype=DType.int32, device=device_ref)
    result._sync_realize()
    assert result.real


def test_repeat_interleave() -> None:
    # repeat_interleave not supported on GPU, use CPU
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=CPU())
    result = F.repeat_interleave(tensor_2d, 2, axis=0)
    result._sync_realize()
    assert result.real


def test_reshape() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.reshape(tensor_2d, [6, 4])
    result._sync_realize()
    assert result.real


def test_scatter() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices_scatter = Tensor.full([2, 2], 0, dtype=DType.int64, device=DEVICE)
    source_scatter = Tensor.ones([2, 2], dtype=DType.float32, device=DEVICE)
    result = F.scatter(
        tensor_2d, source_scatter, indices_scatter, axis=0
    )  # updates, indices order
    result._sync_realize()
    assert result.real


def test_scatter_nd() -> None:
    # Create input tensor to scatter into
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices_nd = Tensor.full([2, 2], 0, dtype=DType.int64, device=DEVICE)
    source_scatter = Tensor.ones(
        [2], dtype=DType.float32, device=DEVICE
    )  # match indices shape
    result = F.scatter_nd(tensor_2d, source_scatter, indices_nd)
    result._sync_realize()
    assert result.real


def test_slice_tensor() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.slice_tensor(tensor_2d, [slice(0, 2), slice(1, 4)])
    result._sync_realize()
    assert result.real


def test_split() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    splits = cast(
        list[Tensor], F.split(tensor_2d, [2, 2], axis=0)
    )  # split_sizes as sequence
    for split_tensor in splits:
        split_tensor._sync_realize()
        assert split_tensor.real


def test_squeeze() -> None:
    # needs tensor with size-1 dimension
    tensor_with_one = Tensor.ones([4, 1, 6], dtype=DType.float32, device=DEVICE)
    result = F.squeeze(tensor_with_one, axis=1)
    result._sync_realize()
    assert result.real


def test_stack() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    tensors_to_stack = [tensor_2d, tensor_2d]
    result = F.stack(tensors_to_stack, axis=0)
    result._sync_realize()
    assert result.real


def test_tile() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.tile(tensor_2d, [2, 1])
    result._sync_realize()
    assert result.real


def test_top_k() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    values, indices = F.top_k(tensor_2d, k=3, axis=-1)
    values._sync_realize()
    indices._sync_realize()
    assert values.real
    assert indices.real


def test_transfer_to() -> None:
    # transfer to same device (should be no-op)
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.transfer_to(tensor_2d, device_ref)
    result._sync_realize()
    assert result.real


def test_transpose() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.transpose(tensor_2d, axis_1=0, axis_2=1)
    result._sync_realize()
    assert result.real


def test_unsqueeze() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.unsqueeze(tensor_2d, axis=1)
    result._sync_realize()
    assert result.real


def test_where() -> None:
    condition = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    x = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    y = Tensor.zeros([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.where(condition, x, y)
    result._sync_realize()
    assert result.real


def test_functional_returns_tensor() -> None:
    @F.functional
    def returns_tensor():  # noqa: ANN202
        return Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)

    result = returns_tensor()
    result._sync_realize()
    assert result.real
