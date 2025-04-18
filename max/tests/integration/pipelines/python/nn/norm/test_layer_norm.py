# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import max.driver as md
import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn.functional as F
from distance_metrics import is_euclidean_distance_close
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import LayerNormV2
from ndarray_from_tensor_type import ndarray_from_tensor_type


def torch_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return F.layer_norm(x, weight.shape, weight=weight, bias=bias, eps=1e-6)


def _test_layer_norm(
    inputs: npt.NDArray[np.floating],
    weight: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
    input_type: TensorType,
    session: InferenceSession,
) -> None:
    # Phase 1: op staging.
    graph = Graph(
        "layer_norm",
        forward=LayerNormV2(
            dims=weight.shape[0],
            eps=1e-6,
            use_bias=True,
            device=DeviceRef.CPU(),
        ),
        input_types=[input_type],
    )

    # Phase 2: model compilation and weight initialization.
    model = session.load(
        graph, weights_registry={"weight": weight, "bias": bias}
    )

    # Phase 3: execution.
    predicted = model(inputs)[0]
    assert isinstance(predicted, md.Tensor)

    expected = (
        torch_layer_norm(
            torch.from_numpy(inputs),
            torch.from_numpy(weight),
            torch.from_numpy(bias),
        )
        .detach()
        .numpy()
    )

    # Compared the outputs.
    assert is_euclidean_distance_close(
        predicted.to_numpy(), expected, rtol=1e-4
    )


@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(DType.float32, ["batch", "dim"]),
        TensorType(DType.float32, ["x", "y", "z", "dim"]),
    ],
)
def test_layer_norm_random(
    session: InferenceSession, input_type: TensorType
) -> None:
    """Compares the output of a single layer norm op against torch."""
    # Generate random inputs with the given tensor types.
    weight_bias_shape = TensorType(input_type.dtype, [input_type.shape[-1]])
    inputs, weight, bias = ndarray_from_tensor_type(
        input_type, weight_bias_shape, weight_bias_shape
    )

    _test_layer_norm(inputs, weight, bias, input_type, session)


@pytest.mark.parametrize(
    "input_type,inputs,weight,bias",
    [
        (
            TensorType(DType.float32, ["dim"]),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ),
    ],
)
def test_layer_norm_aipipe_153(
    session: InferenceSession,
    input_type: TensorType,
    inputs: npt.NDArray[np.floating],
    weight: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
) -> None:
    """Regression test for AIPIPE-153 layer norm corner case."""
    _test_layer_norm(inputs, weight, bias, input_type, session)
