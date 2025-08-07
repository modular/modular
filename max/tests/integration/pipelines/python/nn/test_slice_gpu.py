# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_slice_gpu_ideal_case() -> None:
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    x_input = TensorType(
        shape=("dynamic_dim", 10), dtype=DType.float32, device=device_ref
    )
    x_slice_input = TensorType(shape=(1,), dtype=DType.int64, device=device_ref)

    # TODO(MAXPLAT-363):
    with pytest.raises(TypeError):
        with Graph(
            "slice_gpu", input_types=[x_input, x_slice_input], device=device_ref
        ) as graph:
            x_tensor, x_slice_val = graph.inputs

            # mypy incorrectly flags this as an error, but it should be valid
            out = x_tensor.tensor[x_slice_val.tensor :, :]  # type: ignore

            graph.output(out)


def test_slice_gpu_explicit_devices() -> None:
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    x_input = TensorType(
        shape=("dynamic_dim", 10), dtype=DType.float32, device=device_ref
    )
    x_slice_input = TensorType(shape=(1,), dtype=DType.int64, device=device_ref)

    with Graph(
        "slice_gpu", input_types=[x_input, x_slice_input], device=device_ref
    ) as graph:
        x_tensor, x_slice_val = graph.inputs
        out = ops.slice_tensor(
            x_tensor.tensor,
            [
                (
                    slice(
                        x_slice_val.tensor,
                        ops.constant(-1, DType.int64, device=device_ref),
                        ops.constant(1, DType.int64, device=device_ref),
                    ),
                    "dynamic_slice_dim",
                ),
                (
                    slice(
                        ops.constant(0, DType.int64, device=device_ref),
                        ops.constant(10, DType.int64, device=device_ref),
                        ops.constant(1, DType.int64, device=device_ref),
                    ),
                    10,
                ),
            ],
        )
        graph.output(out)

    device = Accelerator()
    session = InferenceSession(devices=[device])

    # TODO(MAXPLAT-363):
    with pytest.raises(ValueError):
        model = session.load(graph)


def test_slice_gpu_scalar_slice_ideal() -> None:
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    x_input = TensorType(
        shape=("dynamic_dim"), dtype=DType.float32, device=device_ref
    )
    x_slice_input = TensorType(shape=(1,), dtype=DType.int64, device=device_ref)

    # TODO(MAXPLAT-363):
    with pytest.raises(ValueError):
        with Graph(
            "slice_gpu", input_types=[x_input, x_slice_input], device=device_ref
        ) as graph:
            x_tensor, x_slice_val = graph.inputs
            out = x_tensor.tensor[x_slice_val.tensor]
            graph.output(out)


def test_slice_gpu_scalar_slice() -> None:
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    x_input = TensorType(
        shape=("dynamic_dim"), dtype=DType.float32, device=device_ref
    )
    x_slice_input = TensorType(shape=(1,), dtype=DType.int64, device=device_ref)

    # TODO(MAXPLAT-363):
    with pytest.raises(ValueError):
        with Graph(
            "slice_gpu", input_types=[x_input, x_slice_input], device=device_ref
        ) as graph:
            x_tensor, x_slice_val = graph.inputs
            out = ops.slice_tensor(
                x_tensor.tensor,
                [
                    (
                        slice(
                            x_slice_val.tensor,
                            x_slice_val.tensor + 1,
                            ops.constant(1, DType.int64, device=device_ref),
                        ),
                        1,
                    )
                ],
            )
            graph.output(out)
