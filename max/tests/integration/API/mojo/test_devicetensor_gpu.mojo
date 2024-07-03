# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# REQUIRES: cuda
# TODO (MSDK-465): Remove env var
# RUN: TMP_ALLOCATE_ON_DEVICE=1 %mojo %s %S/mo_unnamed.mlir

from sys import argv
from testing import assert_equal, assert_true
from pathlib import Path

from max.engine import InferenceSession, SessionOptions
from max.graph import Graph, Symbol, TensorType, Type
from max.tensor import TensorSpec
from max._driver import cpu_device, cuda_device, Device


fn test_model_device_tensor(
    cpu_dev: Device, cuda_dev: Device, session: InferenceSession
) raises:
    var args = argv()
    assert_equal(len(args), 2)
    assert_true("mo_unnamed.mlir" in args[1])
    var model_path = args[1]
    var model = session.load(Path(model_path))

    # We have to first allocate the input tensor on CPU and fill it there.
    # Attempting to read or write to a CUDA tensor will cause a segfault.
    var cpu_dt = cpu_dev.allocate(TensorSpec(DType.float32, 5))
    var cpu_input_tensor = cpu_dt^.to_tensor[DType.float32, 1]()
    for i in range(5):
        cpu_input_tensor[i] = 1.0

    # We now copy the populated CPU tensor to the CUDA device.
    var cuda_dt = cpu_input_tensor.to_device_tensor().copy_to(cuda_dev)
    var cuda_input_tensor = cuda_dt^.to_tensor[DType.float32, 1]()

    var outputs = model._execute(cuda_input_tensor^)
    assert_equal(len(outputs), 1)

    # In order to examine the output tensor, we need to copy it from the CUDA
    # device to the CPU.
    var cuda_output_dt = outputs[0].take().to_device_tensor()
    var cpu_output_dt = cuda_output_dt.copy_to(cpu_dev)
    var cpu_output_tensor = cpu_output_dt^.to_tensor[DType.float32, 1]()

    assert_equal(cpu_output_tensor[0], 4.0)
    assert_equal(cpu_output_tensor[1], 2.0)
    assert_equal(cpu_output_tensor[2], -5.0)
    assert_equal(cpu_output_tensor[3], 3.0)
    assert_equal(cpu_output_tensor[4], 6.0)


fn test_device_graph(
    cpu_dev: Device, cuda_dev: Device, session: InferenceSession
) raises:
    # Build graph that adds a single constant tensor to its input.
    var g = Graph("add_constant", List[Type](TensorType(DType.float32, 1)))
    var x = g.scalar[DType.float32](1.0, rank=1)
    var y = g.op("mo.add", List[Symbol](g[0], x), TensorType(DType.float32, 1))
    g.output(y)
    g.verify()
    var model = session.load(g)

    var cpu_dt = cpu_dev.allocate(TensorSpec(DType.float32, 1))
    var cpu_input_tensor = cpu_dt^.to_tensor[DType.float32, 1]()
    cpu_input_tensor[0] = 1.0

    var cuda_dt = cpu_input_tensor.to_device_tensor().copy_to(cuda_dev)
    var cuda_input_tensor = cuda_dt^.to_tensor[DType.float32, 1]()

    var outputs = model._execute(cuda_input_tensor^)
    assert_equal(len(outputs), 1)

    var cuda_output_dt = outputs[0].take().to_device_tensor()
    var cpu_output_dt = cuda_output_dt.copy_to(cpu_dev)
    var cpu_output_tensor = cpu_output_dt^.to_tensor[DType.float32, 1]()
    assert_equal(cpu_output_tensor[0], 2.0)


fn main() raises:
    var cpu_dev = cpu_device()
    var cuda_dev = cuda_device()
    var options = SessionOptions()
    options._set_device(cuda_dev)
    var session = InferenceSession(options)

    test_model_device_tensor(cpu_dev, cuda_dev, session)
    test_device_graph(cpu_dev, cuda_dev, session)
