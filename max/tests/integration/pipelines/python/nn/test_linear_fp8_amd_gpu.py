# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test Linear layer with FP8 quantization on AMD GPUs."""

from typing import Any

import numpy as np
import torch
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Linear,
)
from max.nn.float8_config import Float8WeightScaleSpec


def test_linear_fp8_amd_conversion(gpu_session: InferenceSession) -> None:
    """Test Linear layer applies AMD FP8 conversion when needed."""

    float8_config = Float8Config(
        input_scale=Float8InputScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.TENSOR,
            origin=Float8ScaleOrigin.STATIC,
        ),
        weight_scale=Float8WeightScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.TENSOR,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
    )

    negative_zero = np.float32(-0.0)
    test_values = np.array(
        [1.0, 2.0, -1.0, 0.0, negative_zero], dtype=np.float32
    )
    assert np.signbit(test_values[4]) and test_values[4] == 0.0

    weights_scale = Tensor.from_dlpack(torch.tensor([1.0], dtype=torch.float32))
    input_scale = Tensor.from_dlpack(torch.tensor([1.0], dtype=torch.float32))

    device = Accelerator()
    input_values = torch.from_numpy(test_values).unsqueeze(0).to(torch.bfloat16)
    inputs = Tensor.from_dlpack(input_values).to(device)

    def run_linear_test(
        dtype: DType,
        name_suffix: str,
        test_values: np.ndarray,
        use_float8_config: bool = False,
    ) -> Any:
        linear_layer = Linear(
            name=f"linear_{name_suffix}",
            in_dim=5,
            out_dim=1,
            dtype=dtype,
            device=DeviceRef.GPU(),
            float8_config=float8_config if use_float8_config else None,
        )

        graph = Graph(
            f"linear_{name_suffix}_test",
            linear_layer,
            input_types=[
                TensorType(DType.bfloat16, (1, 5), device=DeviceRef.GPU()),
            ],
        )

        if dtype == DType.float8_e4m3fn:
            weights_torch = torch.from_numpy(test_values).to(
                torch.float8_e4m3fn
            )
            weights_tensor = Tensor.from_dlpack(
                weights_torch.view(torch.uint8)
            ).view(DType.float8_e4m3fn)
            weights_registry = {
                f"linear_{name_suffix}.weight_scale": weights_scale,
                f"linear_{name_suffix}.input_scale": input_scale,
                f"linear_{name_suffix}.weight": weights_tensor,
            }
        else:
            weights_tensor = Tensor.from_dlpack(
                torch.from_numpy(test_values).to(torch.bfloat16)
            )
            weights_registry = {f"linear_{name_suffix}.weight": weights_tensor}

        model = gpu_session.load(graph, weights_registry=weights_registry)
        return model.execute(inputs)[0]

    fp8_result = torch.from_dlpack(
        run_linear_test(
            DType.float8_e4m3fn, "fp8", test_values, use_float8_config=True
        )
    )
    bf16_result = torch.from_dlpack(
        run_linear_test(DType.bfloat16, "bf16", test_values)
    )

    assert torch.isfinite(fp8_result).all(), (
        "FP8 result should be finite (no NaN or Inf)"
    )
    assert torch.allclose(fp8_result, bf16_result, rtol=1e-1, atol=1e-3), (
        "FP8 and BF16 outputs should be similar"
    )


def test_linear_fp8_amd_conversion_dynamic_scale(
    gpu_session: InferenceSession,
) -> None:
    """Test Linear layer applies AMD FP8 conversion with dynamic scaling."""

    float8_config = Float8Config(
        input_scale=Float8InputScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.COLWISE,
            origin=Float8ScaleOrigin.DYNAMIC,
        ),
        weight_scale=Float8WeightScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.ROWWISE,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
    )

    weight_scale = torch.tensor([1.0], dtype=torch.float32)

    negative_zero = np.float32(-0.0)
    base_values = np.array(
        [1.0, 2.0, -1.0, 0.0, negative_zero], dtype=np.float32
    )

    weights_np = np.tile(base_values, (16, 7))[:16, :32].astype(np.float32)
    assert np.signbit(weights_np[0, 4]) and weights_np[0, 4] == 0.0

    np.random.seed(42)
    input_data = np.random.randn(4, 32).astype(np.float32) * 0.5

    device = Accelerator()
    input_values = torch.from_numpy(input_data).to(torch.bfloat16)
    inputs = Tensor.from_dlpack(input_values).to(device)

    def run_linear_dynamic_test(
        dtype: DType,
        name_suffix: str,
        weights_np: np.ndarray,
        use_float8_config: bool = False,
    ) -> Any:
        linear_layer = Linear(
            name=f"linear_{name_suffix}",
            in_dim=32,
            out_dim=16,
            dtype=dtype,
            device=DeviceRef.GPU(),
            float8_config=float8_config if use_float8_config else None,
        )

        graph = Graph(
            f"linear_dynamic_{name_suffix}_test",
            linear_layer,
            input_types=[
                TensorType(DType.bfloat16, (4, 32), device=DeviceRef.GPU()),
            ],
        )

        if dtype == DType.float8_e4m3fn:
            weights_torch = torch.from_numpy(weights_np).to(torch.float8_e4m3fn)
            weights_tensor = Tensor.from_dlpack(
                weights_torch.view(torch.uint8)
            ).view(DType.float8_e4m3fn)
            weights_registry = {
                f"linear_{name_suffix}.weight_scale": weight_scale,
                f"linear_{name_suffix}.weight": weights_tensor,
            }
        else:
            weights_tensor = Tensor.from_dlpack(
                torch.from_numpy(weights_np).to(torch.bfloat16)
            )
            weights_registry = {f"linear_{name_suffix}.weight": weights_tensor}

        model = gpu_session.load(graph, weights_registry=weights_registry)
        return model.execute(inputs)[0]

    fp8_result = torch.from_dlpack(
        run_linear_dynamic_test(
            DType.float8_e4m3fn, "fp8", weights_np, use_float8_config=True
        )
    )

    # TODO: The results of dynamic quantized matmul are quite off
    assert torch.isfinite(fp8_result).all(), (
        "Dynamic scaling: FP8 result should be finite (no NaN or Inf)"
    )
