# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for ops.layer_norm."""

import pytest
from conftest import shapes, static_positive_dims, tensor_types
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Shape, StaticDim, TensorType, ops


@given(
    input_type=tensor_types(
        dtypes=st.sampled_from([DType.float32, DType.float64]),
        shapes=shapes(min_rank=1),
    ),
    epsilon=st.floats(min_value=1e-6, max_value=1.0),
)
def test_layer_norm__valid(
    graph_builder, input_type: TensorType, epsilon: float
) -> None:
    """Test layer_norm with valid inputs."""
    # Create gamma and beta with same shape as last dimension of input
    *_, last_dim = input_type.shape
    gamma_type = TensorType(input_type.dtype, [last_dim], input_type.device)
    beta_type = TensorType(input_type.dtype, [last_dim], input_type.device)

    with graph_builder(
        input_types=[input_type, gamma_type, beta_type]
    ) as graph:
        out = ops.layer_norm(
            graph.inputs[0],
            graph.inputs[1],  # gamma
            graph.inputs[2],  # beta
            epsilon,
        )
        assert out.type == input_type
        graph.output(out)


static_shapes = st.shared(shapes(dims=static_positive_dims))


@given(
    input_type=tensor_types(
        dtypes=st.sampled_from([DType.float32, DType.float64]),
        shapes=static_shapes,
    ),
)
def test_layer_norm__error__gamma_shape_mismatch(
    graph_builder, input_type: TensorType
) -> None:
    """Test that layer_norm raises an error when gamma shape doesn't match the last dimension."""
    # Create a gamma tensor with incorrect shape
    # Use a different value that's still invalid but won't overflow
    gamma_shape = Shape([StaticDim(int(input_type.shape[-1]) - 1)])
    gamma_type = TensorType(input_type.dtype, gamma_shape, input_type.device)
    beta_type = TensorType(
        input_type.dtype,
        Shape([StaticDim(int(input_type.shape[-1]))]),
        input_type.device,
    )

    with graph_builder(
        input_types=[input_type, gamma_type, beta_type]
    ) as graph:
        with pytest.raises(
            ValueError, match="does not match dimension of reduction"
        ):
            ops.layer_norm(
                graph.inputs[0],
                graph.inputs[1],  # gamma
                graph.inputs[2],  # beta
                1e-5,
            )


@given(
    input_type=tensor_types(
        dtypes=st.sampled_from([DType.float32, DType.float64]),
        shapes=static_shapes,
    ),
)
def test_layer_norm__error__beta_shape_mismatch(
    graph_builder, input_type: TensorType
) -> None:
    """Test that layer_norm raises an error when beta shape doesn't match the last dimension."""
    # Create a beta tensor with incorrect shape
    # Use a different value that's still invalid but won't overflow
    beta_shape = Shape([StaticDim(int(input_type.shape[-1]) - 1)])
    gamma_type = TensorType(
        input_type.dtype, Shape([input_type.shape[-1]]), input_type.device
    )
    beta_type = TensorType(input_type.dtype, beta_shape, input_type.device)

    with graph_builder(
        input_types=[input_type, gamma_type, beta_type]
    ) as graph:
        with pytest.raises(
            ValueError, match="does not match dimension of reduction"
        ):
            ops.layer_norm(
                graph.inputs[0],
                graph.inputs[1],  # gamma
                graph.inputs[2],  # beta
                1e-5,
            )


@given(
    input_type=tensor_types(
        dtypes=st.sampled_from([DType.float32, DType.float64]),
        shapes=shapes(min_rank=1),
    ),
    epsilon=st.floats(max_value=0.0),
)
def test_layer_norm__error__non_positive_epsilon(
    graph_builder, input_type: TensorType, epsilon: float
) -> None:
    """Test that layer_norm raises an error when epsilon is not positive."""
    *_, last_dim = input_type.shape
    gamma_type = TensorType(input_type.dtype, [last_dim], input_type.device)
    beta_type = TensorType(input_type.dtype, [last_dim], input_type.device)

    with graph_builder(
        input_types=[input_type, gamma_type, beta_type]
    ) as graph:
        with pytest.raises(ValueError, match="epsilon must be positive"):
            ops.layer_norm(
                graph.inputs[0],
                graph.inputs[1],  # gamma
                graph.inputs[2],  # beta
                epsilon,
            )


@given(
    input_type=tensor_types(
        dtypes=st.sampled_from([DType.float32, DType.float64]),
        shapes=shapes(min_rank=1),
    ),
)
def test_layer_norm__error__zero_last_dim(
    graph_builder, input_type: TensorType
) -> None:
    """Test that layer_norm handles zero-sized last dimension gracefully."""
    *_, last_dim = input_type.shape
    # Create input with zero-sized last dimension
    zero_last_dim_type = TensorType(
        input_type.dtype, [*input_type.shape[:-1], 0], input_type.device
    )
    gamma_type = TensorType(input_type.dtype, [0], input_type.device)
    beta_type = TensorType(input_type.dtype, [0], input_type.device)

    with graph_builder(
        input_types=[zero_last_dim_type, gamma_type, beta_type]
    ) as graph:
        # Should not raise an error for zero-sized last dimension
        out = ops.layer_norm(
            graph.inputs[0],
            graph.inputs[1],  # gamma
            graph.inputs[2],  # beta
            1e-5,
        )
        assert out.type == zero_last_dim_type
        graph.output(out)
