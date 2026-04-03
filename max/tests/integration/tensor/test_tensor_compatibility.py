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
"""Compatibility-lane tensor API tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.experimental import functional as F
from max.experimental.tensor import Tensor


def _devices() -> list[Device]:
    devices: list[Device] = [CPU()]
    if accelerator_count():
        devices.append(Accelerator())
    return devices


@pytest.mark.parametrize("device", _devices())
def test_flatten_size_and_reshape(device) -> None:  # noqa: ANN001
    x = Tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), device=device)
    y = x.reshape(x.size(0), -1, x.size(-1)).flatten(0, 1)

    assert y.shape == [6, 4]
    np.testing.assert_array_equal(
        np.from_dlpack(y.to(CPU())),
        np.arange(24, dtype=np.float32).reshape(6, 4),
    )


def test_unflatten_infers_dimension() -> None:
    x = Tensor(np.arange(24, dtype=np.float32).reshape(2, 12), device=CPU())
    y = x.unflatten(1, (4, -1))

    assert y.shape == [2, 4, 3]
    np.testing.assert_array_equal(
        np.from_dlpack(y),
        np.arange(24, dtype=np.float32).reshape(2, 4, 3),
    )


def test_unbind_returns_tuple() -> None:
    x = Tensor(np.arange(24, dtype=np.int32).reshape(2, 3, 4), device=CPU())
    parts = x.unbind(dim=1)

    assert isinstance(parts, tuple)
    assert len(parts) == 3
    for i, part in enumerate(parts):
        np.testing.assert_array_equal(
            np.from_dlpack(part),
            np.arange(24, dtype=np.int32).reshape(2, 3, 4)[:, i, :],
        )


@pytest.mark.parametrize("device", _devices())
def test_repeat_matches_pytorch(device) -> None:  # noqa: ANN001
    x = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), device=device)
    y = x.repeat(4, 1, 2)
    z = F.repeat(x, 4, 1, 2)

    expected = (
        torch.arange(6, dtype=torch.float32).reshape(2, 3).repeat(4, 1, 2)
    )
    torch.testing.assert_close(torch.from_dlpack(y.to(CPU())), expected)
    torch.testing.assert_close(torch.from_dlpack(z.to(CPU())), expected)


@pytest.mark.parametrize("device", _devices())
def test_repeat_interleave_matches_pytorch(device) -> None:  # noqa: ANN001
    x_np = np.arange(12, dtype=np.float32).reshape(2, 6)
    x = Tensor(x_np, device=device)
    y = F.repeat_interleave(x, 2, axis=1)

    expected = torch.repeat_interleave(
        torch.from_numpy(x_np),
        2,
        dim=1,
    )
    torch.testing.assert_close(torch.from_dlpack(y.to(CPU())), expected)


@pytest.mark.parametrize("device", _devices())
def test_repeat_interleave_zero_matches_pytorch(device) -> None:  # noqa: ANN001
    x_np = np.arange(12, dtype=np.float32).reshape(2, 6)
    x = Tensor(x_np, device=device)
    y = F.repeat_interleave(x, 0, axis=1)

    expected = torch.repeat_interleave(
        torch.from_numpy(x_np),
        0,
        dim=1,
    )
    torch.testing.assert_close(torch.from_dlpack(y.to(CPU())), expected)


def test_repeat_rejects_too_few_dims() -> None:
    x = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), device=CPU())
    with pytest.raises(ValueError):
        x.repeat(2)


def test_masked_fill_broadcasts_and_accepts_float_literals() -> None:
    x = Tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), device=CPU())
    mask = Tensor(
        np.array(
            [
                [[True, False, True, False]],
                [[False, True, False, True]],
            ],
            dtype=np.bool_,
        ),
        device=CPU(),
    )
    y = x.masked_fill(mask, float("-inf"))

    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = np.where(
        np.broadcast_to(np.from_dlpack(mask), expected.shape), -np.inf, expected
    )
    np.testing.assert_array_equal(np.from_dlpack(y), expected)


def test_amin_tuple_axes() -> None:
    x_np = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    x = Tensor(x_np, device=CPU())
    y = x.amin(axis=(1, -2))

    np.testing.assert_array_equal(np.from_dlpack(y), np.amin(x_np, axis=(1, 2)))


def test_swapaxes_flip_and_negative_step_slice() -> None:
    x_np = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    x = Tensor(x_np, device=CPU())

    np.testing.assert_array_equal(
        np.from_dlpack(x.swapaxes(1, 2)),
        np.swapaxes(x_np, 1, 2),
    )
    np.testing.assert_array_equal(
        np.from_dlpack(x[:, ::-1]),
        x_np[:, ::-1],
    )
    np.testing.assert_array_equal(
        np.from_dlpack(x.flip(dims=[1])),
        np.flip(x_np, axis=1),
    )


def test_bitwise_named_ops_and_operators() -> None:
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y_np = np.array([[7, 3, 1], [2, 8, 4]], dtype=np.int32)
    x = Tensor(x_np, device=CPU())
    y = Tensor(y_np, device=CPU())

    np.testing.assert_array_equal(
        np.from_dlpack(F.bitwise_and(x, y)), x_np & y_np
    )
    np.testing.assert_array_equal(
        np.from_dlpack(F.bitwise_or(x, y)), x_np | y_np
    )
    np.testing.assert_array_equal(
        np.from_dlpack(F.bitwise_xor(x, y)), x_np ^ y_np
    )
    np.testing.assert_array_equal(np.from_dlpack(x & y), x_np & y_np)
    np.testing.assert_array_equal(np.from_dlpack(x | y), x_np | y_np)
    np.testing.assert_array_equal(np.from_dlpack(x ^ y), x_np ^ y_np)


def test_bool_operator_overloads_stay_logical() -> None:
    x_np = np.array([True, True, False, False], dtype=np.bool_)
    y_np = np.array([True, False, True, False], dtype=np.bool_)
    x = Tensor(x_np, device=CPU())
    y = Tensor(y_np, device=CPU())

    np.testing.assert_array_equal(
        np.from_dlpack(x & y), np.logical_and(x_np, y_np)
    )
    np.testing.assert_array_equal(
        np.from_dlpack(x | y), np.logical_or(x_np, y_np)
    )
    np.testing.assert_array_equal(
        np.from_dlpack(x ^ y), np.logical_xor(x_np, y_np)
    )


@pytest.mark.parametrize("device", _devices())
def test_interpolate_nearest_scale_factor(device) -> None:  # noqa: ANN001
    x_np = np.arange(1 * 1 * 2 * 3, dtype=np.float32).reshape(1, 1, 2, 3)
    x = Tensor(x_np, device=device)
    y = F.interpolate(x, scale_factor=2.0, mode="nearest")

    expected = torch.nn.functional.interpolate(
        torch.from_numpy(x_np),
        scale_factor=2.0,
        mode="nearest",
    )
    torch.testing.assert_close(torch.from_dlpack(y.to(CPU())), expected)


@pytest.mark.parametrize("device", _devices())
def test_interpolate_nearest_size(device) -> None:  # noqa: ANN001
    x_np = np.arange(1 * 1 * 10 * 6, dtype=np.float32).reshape(1, 1, 10, 6)
    x = Tensor(x_np, device=device)
    y = F.interpolate(x, size=(5, 3), mode="nearest")

    expected = torch.nn.functional.interpolate(
        torch.from_numpy(x_np),
        size=(5, 3),
        mode="nearest",
    )
    torch.testing.assert_close(torch.from_dlpack(y.to(CPU())), expected)


@pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
def test_mul_integer_bfloat16_promotes_to_bfloat16(
    int_dtype: torch.dtype,
) -> None:
    lhs_torch = torch.tensor([[1, 2], [3, 4]], dtype=int_dtype)
    rhs_torch = torch.tensor(
        [[1.5, 2.0], [3.0, 4.5]],
        dtype=torch.bfloat16,
    )

    lhs = Tensor.from_dlpack(lhs_torch)
    rhs = Tensor.from_dlpack(rhs_torch)
    out = lhs * rhs

    expected = lhs_torch * rhs_torch
    torch.testing.assert_close(torch.from_dlpack(out), expected)
