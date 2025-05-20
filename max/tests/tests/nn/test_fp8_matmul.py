# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for FP8 matmul kernels in max.nn.kernels."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kernels import dynamic_scaled_matmul


class DynamicScaledMatmul:
    return_type: DType
    """Return type of the `dynamic_scaled_matmul` custom op."""

    def __init__(self, return_type: DType) -> None:
        self.return_type = return_type

    def __call__(
        self,
        a: TensorValue,
        b: TensorValue,
        a_scales: TensorValue,
        b_scales: TensorValue,
    ) -> TensorValue:
        return dynamic_scaled_matmul(
            a, b, a_scales, b_scales, out_type=self.return_type
        )


def test_dynamic_scaled_matmul_rowwise():
    """Tests dynamic_scaled_matmul with valid inputs."""
    device = DeviceRef.CPU()
    with Graph(
        "dynamic_scaled_matmul",
        input_types=[
            # a
            TensorType(DType.float8_e4m3fn, shape=(2, 4), device=device),
            # b
            TensorType(DType.float8_e4m3fn, shape=(3, 4), device=device),
            # a_scales
            TensorType(DType.bfloat16, shape=(2, 1), device=device),
            # b_scales
            TensorType(DType.bfloat16, shape=(3, 1), device=device),
        ],
    ) as graph:
        a, b, a_scales, b_scales = (inp.tensor for inp in graph.inputs)

        # Test with row-wise weight scales.
        output_rowwise = dynamic_scaled_matmul(
            a, b, a_scales, b_scales, out_type=DType.bfloat16
        )
        assert output_rowwise.shape == [2, 3]
        assert output_rowwise.dtype == DType.bfloat16


@pytest.mark.parametrize(
    "a_dtype, b_dtype, a_scales_dtype, b_scales_dtype, err_msg_part",
    [
        # a.dtype != b.dtype
        (
            DType.float16,
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            "a and b dtypes",
        ),
        # a.dtype != b.dtype
        (
            DType.float8_e4m3fn,
            DType.float16,
            DType.bfloat16,
            DType.bfloat16,
            "a and b dtypes",
        ),
        # a_scales.dtype != b_scales.dtype
        (
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.float16,
            DType.bfloat16,
            "scales dtypes",
        ),
        # a_scales.dtype != b_scales.dtype
        (
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float16,
            "scales dtypes",
        ),
    ],
)
def test_dynamic_scaled_matmul_dtype_mismatch(
    a_dtype: DType,
    b_dtype: DType,
    a_scales_dtype: DType,
    b_scales_dtype: DType,
    err_msg_part: str,
) -> None:
    """Tests dtype mismatches."""
    device = DeviceRef.CPU()
    with pytest.raises(TypeError, match=err_msg_part):
        Graph(
            "dynamic_scaled_matmul",
            forward=DynamicScaledMatmul(return_type=DType.bfloat16),
            input_types=[
                # a
                TensorType(a_dtype, shape=(2, 4), device=device),
                # b
                TensorType(b_dtype, shape=(3, 4), device=device),
                # a_scales
                TensorType(a_scales_dtype, shape=(2, 1), device=device),
                # b_scales
                TensorType(b_scales_dtype, shape=(3, 1), device=device),
            ],
        )
