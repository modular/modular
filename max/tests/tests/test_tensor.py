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
"""Tests `max.driver.Buffer` basic behaviors."""

import asyncio

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.tensor import (
    Tensor,
    TensorType,
    _default_device,
    _default_dtype,
    default_device,
    default_dtype,
    defaults_like,
    driver_tensor_type,
)
from max.graph import BufferValue, DeviceRef, Graph


def test_tensor_basic() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    b = a + 1
    assert isinstance(b, Tensor)
    assert b.type == a.type == expected_type
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_tensor_basic_lazy() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    with F.lazy():
        b = a + 1
    assert isinstance(b, Tensor)
    assert not b.real
    assert b.type == a.type == expected_type

    asyncio.run(b.realize)
    assert b.real
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_tensor_with_intermediate() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    b = a + a + 1
    assert isinstance(b, Tensor)
    assert b.type == a.type == expected_type
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_tensor_with_intermediate_lazy() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    with F.lazy():
        b = a + a + 1
    assert isinstance(b, Tensor)
    assert not b.real
    assert b.type == a.type == expected_type

    asyncio.run(b.realize)
    assert b.real
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_compilation_failure() -> None:
    a_data = Buffer.zeros([5, 5], DType.float8_e4m3fn, CPU())
    b_data = Buffer.zeros([5, 5], DType.float32, CPU())
    a = Tensor(storage=a_data)

    with F.lazy():
        # Adding fp8 on cpu is unsupported
        fails_compilation = a + 1

    with pytest.raises(Exception):
        asyncio.run(fails_compilation.realize)

    # Test that new tensor ops can still execucte
    b = Tensor(storage=b_data)
    c = b + 1
    assert c.real


def test_tensor_dlpack() -> None:
    data = Buffer.zeros([5, 5], DType.float32, CPU())
    t = Tensor(storage=data)
    assert t.type == driver_tensor_type(data)
    assert t.real
    npt = np.from_dlpack(t)
    assert npt.dtype == t.dtype.to_numpy()
    assert list(npt.shape) == t.shape


def test_tensor_lazy_dlpack() -> None:
    expected_type = TensorType(DType.float32, [5, 5], DeviceRef.CPU())
    with F.lazy():
        t = random.normal_like(expected_type)
    npt = np.from_dlpack(t)
    assert npt.dtype == t.dtype.to_numpy()
    assert list(npt.shape) == t.shape


def test_tensor_from_dlpack() -> None:
    npt = np.random.normal([5, 5])
    t = Tensor.from_dlpack(npt)
    assert t.real
    assert npt.dtype == t.dtype.to_numpy()
    assert list(npt.shape) == t.shape


def test_functional_in_graph() -> None:
    with Graph("test_functional") as graph:
        graph.output(F.constant(1, dtype=DType.float32, device=DeviceRef.CPU()))


def test_constant_default_dtype() -> None:
    t = Tensor.constant(1, device=CPU())
    assert t.dtype == _default_dtype(CPU())
    assert t.device == CPU()

    assert DType.float64 != _default_dtype(CPU())
    with default_dtype(DType.float64):
        t = Tensor.constant(1, device=CPU())
    assert t.dtype == DType.float64


def test_constant_default_device() -> None:
    t = Tensor.constant(1)
    assert t.device == _default_device()
    assert t.dtype == _default_dtype(_default_device())


def test_defaults_like() -> None:
    t = Tensor.constant(1, dtype=DType.float64)
    with defaults_like(t):
        t2 = Tensor.constant(1)
        assert t.type == t2.type
    with defaults_like(t.type):
        t3 = Tensor.constant(1)
        assert t.type == t3.type


@pytest.mark.skipif(
    not accelerator_count(), reason="requires at least 2 devices"
)
def test_constant_default_device_context() -> None:
    assert _default_device() != CPU()
    with default_device(CPU()):
        t = Tensor.constant(1)

    assert t.device == CPU()
    assert t.dtype == _default_dtype(CPU())


def test_realized_tensor_as_buffer() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    b = Tensor.ones_like(a.type)
    F.buffer_store(a, b)
    assert a.real


def test_realized_tensor_as_buffer_lazy() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.real
    with F.lazy():
        b = Tensor.ones_like(a.type)
        # Woof. `a` is a `Tensor`, not a `LazyTensor`. What does this do?
        F.buffer_store(a, b)
    assert not a.real
    asyncio.run(a.realize)
    assert a.real


def test_unrealized_value_as_buffer() -> None:
    with F.lazy():
        a = Tensor.zeros([5, 5])
        b = Tensor.ones_like(a.type)
        assert not a.real
        F.buffer_store(a, b)
        assert not a.real
    asyncio.run(a.realize)
    assert a.real


def test_buffervalue_on_realized_tensor() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.real
    with F.lazy():
        _ = BufferValue(a)
        # Don't know whether the value was thrown away or used
        # in a mutating op!
        assert not a.real
    asyncio.run(a.realize)
    assert a.real


def test_mutation_op_order() -> None:
    a = Tensor.zeros([1])
    b = Tensor.ones_like(a.type)
    c = a + b
    F.buffer_store(a, b)
    d = a + b
    assert a.item() == 1.0
    assert b.item() == 1.0
    assert c.item() == 1.0
    assert d.item() == 2.0


def test_mutation_op_order_lazy() -> None:
    with F.lazy():
        a = Tensor.zeros([1])
        b = Tensor.ones_like(a.type)
        c = a + b
        F.buffer_store(a, b)
        d = a + b
    asyncio.run(c.realize)
    asyncio.run(d.realize)
    assert a.item() == 1.0
    assert b.item() == 1.0
    assert c.item() == 1.0
    assert d.item() == 2.0
