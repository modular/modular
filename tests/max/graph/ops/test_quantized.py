# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for quantized operations."""

import pytest
from conftest import shapes, static_dims, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.driver import accelerator_count
from max.dtype import DType
from max.graph import Dim, StaticDim, TensorType, ops
from max.graph.quantization import QuantizationEncoding


# Helper function to create tensor types with specific dtypes
def tensor_types_with_dtype(dtype: DType, **kwargs):
    return tensor_types(dtypes=st.just(dtype), **kwargs)


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="Quantization only supported on cpu currently",
)
@given(
    base_type=tensor_types_with_dtype(
        DType.float32,
        shapes=shapes(
            min_rank=2,
            max_rank=2,
            dims=static_dims(min=1),  # Ensure non-zero dimensions
        ),
    ),
    encoding=st.sampled_from(
        [
            QuantizationEncoding.Q4_0,
            QuantizationEncoding.Q4_K,
            QuantizationEncoding.Q6_K,
        ]
    ),
)
def test_qmatmul(
    graph_builder, base_type: TensorType, encoding: QuantizationEncoding
) -> None:
    """Test qmatmul with basic inputs."""
    *_, m, n = base_type.shape
    # Ensure non-zero dimensions
    assume(m != 0 and n != 0)

    # Create quantized input type
    quantized_type = TensorType(
        DType.uint8,
        (n, m),  # Note: quantized input is transposed
        base_type.device,
    )

    with graph_builder(input_types=[base_type, quantized_type]) as graph:
        out = ops.qmatmul(encoding, None, *[x.tensor for x in graph.inputs])
        assert out.type == base_type
        graph.output(out)


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="Quantization only supported on cpu currently",
)
@given(
    base_type=tensor_types_with_dtype(
        DType.float32,
        shapes=shapes(
            min_rank=2,
            max_rank=2,
            dims=static_dims(
                min=1, max=2**30
            ),  # Limit max dimension to avoid overflow
        ),
    ),
    encoding=st.sampled_from(
        [
            QuantizationEncoding.Q4_0,
            QuantizationEncoding.Q4_K,
            QuantizationEncoding.Q6_K,
        ]
    ),
)
def test_dequantize(
    graph_builder, base_type: TensorType, encoding: QuantizationEncoding
) -> None:
    """Test dequantize with basic inputs."""
    *_, m, n = base_type.shape
    # For static dimensions, ensure they are divisible by block size
    if isinstance(n, StaticDim):
        # Round up to nearest multiple of block size, but ensure we don't overflow
        n_value = int(n)
        max_value = 2**63 - 1
        n_value = min(
            ((n_value + encoding.block_size - 1) // encoding.block_size)
            * encoding.block_size,
            max_value,
        )
        n = StaticDim(n_value)

    # Create quantized input type
    quantized_type = TensorType(
        DType.uint8,
        (m, n),
        base_type.device,
    )

    with graph_builder(input_types=[quantized_type]) as graph:
        out = ops.dequantize(encoding, *[x.tensor for x in graph.inputs])
        if isinstance(n, StaticDim):
            expected_shape = (
                *base_type.shape[:-1],
                (int(n) // encoding.block_size) * encoding.elements_per_block,
            )
            assert out.type == TensorType(
                DType.float32, expected_shape, base_type.device
            )
        graph.output(out)


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="Quantization only supported on cpu currently",
)
@given(
    base_type=tensor_types_with_dtype(
        DType.float32,
        shapes=shapes(
            min_rank=2,
            dims=static_dims(min=1),
        ),
    ),
    encoding=st.sampled_from(
        [
            QuantizationEncoding.Q4_0,
            QuantizationEncoding.Q4_K,
            QuantizationEncoding.Q6_K,
        ]
    ),
)
def test_dequantize__error__nondivisible_block_size(
    graph_builder, base_type: TensorType, encoding: QuantizationEncoding
) -> None:
    """Test that dequantize raises an error when last dimension is not divisible by block size."""
    *_, m, n = base_type.shape
    # Only check divisibility for static dimensions
    if isinstance(n, StaticDim):
        assume(
            int(n) % encoding.block_size != 0
        )  # Ensure NOT divisible by block size

    # Create quantized input type
    quantized_type = TensorType(
        DType.uint8,
        (m, n),
        base_type.device,
    )

    with graph_builder(input_types=[quantized_type]) as graph:
        with pytest.raises(
            ValueError,
            match=f"last dimension \\({n}\\) not divisible by block size \\({encoding.block_size}\\)",
        ):
            ops.dequantize(encoding, *[x.tensor for x in graph.inputs])


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="Quantization only supported on cpu currently",
)
@given(
    base_type=tensor_types_with_dtype(DType.float32, shapes=shapes(min_rank=2)),
    encoding=st.sampled_from(
        [
            QuantizationEncoding.Q4_0,
            QuantizationEncoding.Q4_K,
            QuantizationEncoding.Q6_K,
        ]
    ),
)
def test_dequantize__error__nonstatic_last_dim(
    graph_builder, base_type: TensorType, encoding: QuantizationEncoding
) -> None:
    """Test that dequantize raises an error when last dimension is not static."""
    *_, m = base_type.shape
    dynamic_dim = Dim("dynamic")

    # Create quantized input type with dynamic last dimension
    quantized_type = TensorType(
        DType.uint8,
        (m, dynamic_dim),
        base_type.device,
    )

    with graph_builder(input_types=[quantized_type]) as graph:
        with pytest.raises(
            TypeError,
            match="dequantize only supported with static last dimension",
        ):
            ops.dequantize(encoding, *[x.tensor for x in graph.inputs])
