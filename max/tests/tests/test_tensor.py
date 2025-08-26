# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests `max.Tensor` basic behaviors."""

import asyncio
import warnings

import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.driver import Tensor as DriverTensor
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
    driver_tensor_type,
)
from max.graph import DeviceRef, Graph

DEVICE = Accelerator() if accelerator_count() else CPU()


def test_tensor_basic() -> None:
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = DriverTensor.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    b = a + 1
    assert isinstance(b, Tensor)
    assert not b.real
    assert b.type == a.type == expected_type

    asyncio.run(b.realize)
    assert b.real
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_tensor_with_intermediate() -> None:
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = DriverTensor.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    b = a + a + 1
    assert isinstance(b, Tensor)
    assert not b.real
    assert b.type == a.type == expected_type

    asyncio.run(b.realize)
    assert b.real
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_compilation_failure() -> None:
    a_data = DriverTensor.zeros([5, 5], DType.float8_e4m3fn, CPU())
    b_data = DriverTensor.zeros([5, 5], DType.float32, CPU())
    a = Tensor(storage=a_data)
    a_plus_1 = a + 1
    assert isinstance(a_plus_1, Tensor)

    with pytest.raises(Exception):
        asyncio.run(a_plus_1.realize)

    del a, a_plus_1

    b = Tensor(storage=b_data)
    c = b + 1
    asyncio.run(c.realize)
    assert c.real


def test_tensor_dlpack() -> None:
    data = DriverTensor.zeros([5, 5], DType.float32, CPU())
    t = Tensor(storage=data)
    assert t.type == driver_tensor_type(data)
    assert t.real
    npt = np.from_dlpack(t)
    assert npt.dtype == t.dtype.to_numpy()
    assert list(npt.shape) == t.shape


def test_tensor_eager_dlpack() -> None:
    expected_type = TensorType(DType.float32, [5, 5], DeviceRef.CPU())
    t = random.normal_like(expected_type)
    assert not t.real
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


def test_tensor_warns_on_sync() -> None:
    async def coro():
        t = Tensor.constant(1, dtype=DType.float32, device=CPU())
        return t.item()

    with warnings.catch_warnings(record=True) as warns:
        asyncio.run(coro())

    assert warns


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


@pytest.mark.skipif(
    not accelerator_count(), reason="requires at least 2 devices"
)
def test_constant_default_device_context() -> None:
    assert _default_device() != CPU()
    with default_device(CPU()):
        t = Tensor.constant(1)

    assert t.device == CPU()
    assert t.dtype == _default_dtype(CPU())
