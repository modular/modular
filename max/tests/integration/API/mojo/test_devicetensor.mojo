# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo %s %S/mo_unnamed.mlir

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/engine-device-tensor
# RUN: %mojo-build %s -o %t/engine-device-tensor
# RUN: %t/engine-device-tensor %S/mo_unnamed.mlir

from pathlib import Path
from sys import argv

from max.driver import (
    Device,
    DeviceTensor,
    Tensor,
    accelerator,
    accelerator_count,
    cpu,
)
from max.engine import InferenceSession, SessionOptions
from max.graph import Graph, Symbol, TensorType, Type
from max.tensor import TensorShape, TensorSpec
from testing import assert_equal, assert_true


fn test_model_device_tensor() raises:
    var args = argv()
    assert_equal(len(args), 2)
    assert_true("mo_unnamed.mlir" in args[1])

    var model_path = args[1]
    var device = cpu()

    var options = SessionOptions()
    options._set_device(device)
    var session = InferenceSession(options)
    var model = session.load(Path(model_path))

    var dt = device.allocate(TensorSpec(DType.float32, 5))
    var input_tensor = dt^.to_tensor[DType.float32, 1]()
    for i in range(5):
        input_tensor[i] = 1.0

    var outputs = model.execute(input_tensor^)

    assert_equal(len(outputs), 1)
    var dm_back = outputs[0].take().to_device_tensor()
    var output_tensor = dm_back^.to_tensor[DType.float32, 1]()

    assert_equal(output_tensor[0], 4.0)
    assert_equal(output_tensor[1], 2.0)
    assert_equal(output_tensor[2], -5.0)
    assert_equal(output_tensor[3], 3.0)
    assert_equal(output_tensor[4], 6.0)


fn test_device_graph() raises:
    # Build graph that adds a single constant tensor to its input.
    var g = Graph("add_constant", List[Type](TensorType(DType.float32, 1)))
    var x = g.scalar[DType.float32](1.0, rank=1)
    var y = g.op("mo.add", List[Symbol](g[0], x), TensorType(DType.float32, 1))
    g.output(y)
    g.verify()

    var device = cpu()
    var options = SessionOptions()
    options._set_device(device)
    var session = InferenceSession(options)
    var model = session.load(g)

    var dt = device.allocate(TensorSpec(DType.float32, 1))
    var input_tensor = dt^.to_tensor[DType.float32, 1]()
    input_tensor[0] = 1.0

    var outputs = model.execute(input_tensor^)

    assert_equal(len(outputs), 1)
    var dm_back = outputs[0].take().to_device_tensor()
    var output_tensor = dm_back^.to_tensor[DType.float32, 1]()
    assert_equal(output_tensor[0], 2.0)


def test_model_device_tensor_gpu(
    cpu_dev: Device,
    gpu_dev: Device,
    session: InferenceSession,
    inputs_on_device: Bool,
):
    args = argv()
    assert_equal(len(args), 2)
    assert_true("mo_unnamed.mlir" in args[1])
    model_path = args[1]
    model = session.load(Path(model_path))

    # We have to first allocate the input tensor on CPU and fill it there.
    # Attempting to read or write to a GPU tensor will cause a segfault.
    cpu_input_tensor = Tensor[DType.float32, 1](
        TensorShape(
            5,
        )
    )
    for i in range(5):
        cpu_input_tensor[i] = 1.0

    # We now copy the populated CPU tensor to the GPU device.
    input = cpu_input_tensor^.to_device_tensor()
    if inputs_on_device:
        input = input^.copy_to(gpu_dev)

    outputs = model.execute(input)
    assert_equal(len(outputs), 1)

    # In order to examine the output tensor, we need to copy it from the GPU
    # device to the CPU.
    gpu_output_dt = outputs[0].take().to_device_tensor()
    cpu_output_tensor = gpu_output_dt.copy_to(cpu_dev).to_tensor[
        DType.float32, 1
    ]()

    assert_equal(cpu_output_tensor[0], 4.0)
    assert_equal(cpu_output_tensor[1], 2.0)
    assert_equal(cpu_output_tensor[2], -5.0)
    assert_equal(cpu_output_tensor[3], 3.0)
    assert_equal(cpu_output_tensor[4], 6.0)


def test_device_graph_gpu(
    cpu_dev: Device,
    gpu_dev: Device,
    session: InferenceSession,
    inputs_on_device: Bool,
):
    # Build graph that adds a single constant tensor to its input.
    g = Graph("add_constant", List[Type](TensorType(DType.float32, 1)))
    x = g.scalar[DType.float32](1.0, rank=1)
    y = g.op("mo.add", List[Symbol](g[0], x), TensorType(DType.float32, 1))
    g.output(y)
    g.verify()
    model = session.load(g)

    cpu_input_tensor = Tensor[DType.float32, 1](
        TensorShape(
            1,
        )
    )
    cpu_input_tensor[0] = 1.0

    input = cpu_input_tensor^.to_device_tensor()
    if inputs_on_device:
        input = input^.copy_to(gpu_dev)

    outputs = model.execute(input)
    assert_equal(len(outputs), 1)

    gpu_output_dt = outputs[0].take().to_device_tensor()
    cpu_output_tensor = gpu_output_dt.copy_to(cpu_dev).to_tensor[
        DType.float32, 1
    ]()
    assert_equal(cpu_output_tensor[0], 2.0)


fn main() raises:
    test_model_device_tensor()
    test_device_graph()

    if accelerator_count() > 0:
        cpu_dev = cpu()
        gpu_dev = accelerator()
        options = SessionOptions()
        options._set_device(gpu_dev)
        session = InferenceSession(options)

        test_model_device_tensor_gpu(
            cpu_dev, gpu_dev, session, inputs_on_device=True
        )
        test_model_device_tensor_gpu(
            cpu_dev, gpu_dev, session, inputs_on_device=False
        )
        test_device_graph_gpu(cpu_dev, gpu_dev, session, inputs_on_device=True)
        test_device_graph_gpu(cpu_dev, gpu_dev, session, inputs_on_device=False)
