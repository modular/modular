# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Smoke tests for ops in `max.experimental.functional`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef

DEVICE = Accelerator() if accelerator_count() else CPU()


def test_as_interleaved_complex():
    # needs even last dimension
    complex_input = Tensor.ones([2, 4], dtype=DType.float32, device=DEVICE)
    result = F.as_interleaved_complex(complex_input)
    result._sync_realize()
    assert result.real


def test_avg_pool2d():
    # needs 4D input with NHWC format
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    result = F.avg_pool2d(tensor_4d, kernel_size=(2, 2), stride=1)
    result._sync_realize()
    assert result.real


def test_band_part():
    # needs at least 2D tensor
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.band_part(tensor_2d, num_lower=-1, num_upper=0)
    result._sync_realize()
    assert result.real


def test_broadcast_to():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.broadcast_to(tensor_2d, [4, 6])
    result._sync_realize()
    assert result.real


def test_cast():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.cast(tensor_2d, DType.int64)
    result._sync_realize()
    assert result.real
    assert result.dtype == DType.int64


def test_chunk():
    # split into 2 chunks along axis 0
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    chunks = F.chunk(tensor_2d, chunks=2, axis=0)
    for chunk in chunks:
        chunk._sync_realize()
        assert chunk.real


def test_constant():
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.constant(1.0, DType.float32, device_ref)
    result._sync_realize()
    assert result.real


def test_conv2d():
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
def test_conv2d_transpose():
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    weight = Tensor.ones(
        [3, 3, 1, 2], dtype=DType.float32, device=DEVICE
    )  # [H, W, out_ch, in_ch]
    result = F.conv2d_transpose(tensor_4d, weight)
    result._sync_realize()
    assert result.real


def test_conv3d():
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


def test_flatten():
    tensor_3d = Tensor.ones([2, 3, 4], dtype=DType.float32, device=DEVICE)
    result = F.flatten(tensor_3d, start_dim=1, end_dim=2)
    result._sync_realize()
    assert result.real


def test_fold():
    # needs shape [N, C * kernel_size[0] * kernel_size[1], L]
    # For kernel_size=[2, 2], we need C * 4 channels
    kernel_size = [2, 2]
    tensor_3d = Tensor.ones(
        [1, 4, 4], dtype=DType.float32, device=DEVICE
    )  # [N, C*kernel_prod, L]
    result = F.fold(tensor_3d, output_size=[3, 3], kernel_size=kernel_size)
    result._sync_realize()
    assert result.real


def test_gather():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices = Tensor.full([2], 0, dtype=DType.int64, device=DEVICE)
    result = F.gather(tensor_2d, indices, axis=0)
    result._sync_realize()
    assert result.real


def test_gather_nd():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices_nd = Tensor.full([2, 2], 0, dtype=DType.int64, device=DEVICE)
    result = F.gather_nd(tensor_2d, indices_nd)
    result._sync_realize()
    assert result.real


def test_hann_window():
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.hann_window(4, device=device_ref)
    result._sync_realize()
    assert result.real


@pytest.mark.skipif(
    isinstance(DEVICE, CPU), reason="IRFFT only supported on GPU"
)
def test_irfft():
    tensor_2d = Tensor.ones([4, 8], dtype=DType.float32, device=DEVICE)
    result = F.irfft(tensor_2d, n=14, axis=-1)
    result._sync_realize()
    assert result.real


def test_layer_norm():
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


def test_masked_scatter():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    mask = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    source = Tensor.ones([24], dtype=DType.float32, device=DEVICE)
    result = F.masked_scatter(tensor_2d, mask, source, out_dim=24)
    result._sync_realize()
    assert result.real


def test_matmul():
    a = Tensor.ones([4, 3], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([3, 6], dtype=DType.float32, device=DEVICE)
    result = F.matmul(a, b)
    result._sync_realize()
    assert result.real


def test_max_pool2d():
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    result = F.max_pool2d(tensor_4d, kernel_size=(2, 2), stride=1)
    result._sync_realize()
    assert result.real


def test_nonzero():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.nonzero(
        tensor_2d, out_dim=24
    )  # assuming all elements are nonzero
    result._sync_realize()
    assert result.real


def test_outer():
    vec_a = Tensor.ones([3], dtype=DType.float32, device=DEVICE)
    vec_b = Tensor.ones([4], dtype=DType.float32, device=DEVICE)
    result = F.outer(vec_a, vec_b)
    result._sync_realize()
    assert result.real


def test_pad():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.pad(tensor_2d, [1, 1, 2, 2])  # pad left, right, top, bottom
    result._sync_realize()
    assert result.real


def test_permute():
    tensor_3d = Tensor.ones([2, 3, 4], dtype=DType.float32, device=DEVICE)
    result = F.permute(tensor_3d, [2, 0, 1])
    result._sync_realize()
    assert result.real


def test_range():
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.range(0, 10, 1, dtype=DType.int32, device=device_ref)
    result._sync_realize()
    assert result.real


def test_repeat_interleave():
    # repeat_interleave not supported on GPU, use CPU
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=CPU())
    result = F.repeat_interleave(tensor_2d, 2, axis=0)
    result._sync_realize()
    assert result.real


def test_reshape():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.reshape(tensor_2d, [6, 4])
    result._sync_realize()
    assert result.real


def test_scatter():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices_scatter = Tensor.full([2, 2], 0, dtype=DType.int64, device=DEVICE)
    source_scatter = Tensor.ones([2, 2], dtype=DType.float32, device=DEVICE)
    result = F.scatter(
        tensor_2d, source_scatter, indices_scatter, axis=0
    )  # updates, indices order
    result._sync_realize()
    assert result.real


def test_scatter_nd():
    # Create input tensor to scatter into
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices_nd = Tensor.full([2, 2], 0, dtype=DType.int64, device=DEVICE)
    source_scatter = Tensor.ones(
        [2], dtype=DType.float32, device=DEVICE
    )  # match indices shape
    result = F.scatter_nd(tensor_2d, source_scatter, indices_nd)
    result._sync_realize()
    assert result.real


def test_slice_tensor():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.slice_tensor(tensor_2d, [slice(0, 2), slice(1, 4)])
    result._sync_realize()
    assert result.real


def test_split():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    splits = F.split(tensor_2d, [2, 2], axis=0)  # split_sizes as sequence
    for split_tensor in splits:
        split_tensor._sync_realize()
        assert split_tensor.real


def test_squeeze():
    # needs tensor with size-1 dimension
    tensor_with_one = Tensor.ones([4, 1, 6], dtype=DType.float32, device=DEVICE)
    result = F.squeeze(tensor_with_one, axis=1)
    result._sync_realize()
    assert result.real


def test_stack():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    tensors_to_stack = [tensor_2d, tensor_2d]
    result = F.stack(tensors_to_stack, axis=0)
    result._sync_realize()
    assert result.real


def test_tile():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.tile(tensor_2d, [2, 1])
    result._sync_realize()
    assert result.real


def test_top_k():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    values, indices = F.top_k(tensor_2d, k=3, axis=-1)
    values._sync_realize()
    indices._sync_realize()
    assert values.real
    assert indices.real


def test_transfer_to():
    # transfer to same device (should be no-op)
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.transfer_to(tensor_2d, device_ref)
    result._sync_realize()
    assert result.real


def test_transpose():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.transpose(tensor_2d, axis_1=0, axis_2=1)
    result._sync_realize()
    assert result.real


def test_unsqueeze():
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.unsqueeze(tensor_2d, axis=1)
    result._sync_realize()
    assert result.real


def test_where():
    condition = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    x = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    y = Tensor.zeros([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.where(condition, x, y)
    result._sync_realize()
    assert result.real


def test_functional_returns_tensor():
    @F.functional
    def returns_tensor():
        return Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)

    result = returns_tensor()
    result._sync_realize()
    assert result.real
