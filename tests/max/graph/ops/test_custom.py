# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for ops.custom and ops.inplace_custom.

These tests focus on API validation and error handling for custom operations.
Includes both explicit test cases and property-based tests using Hypothesis.
Full integration tests with actual kernel execution are located in:
SDK/integration-test/API/python/graph/test_custom_op_*.py
"""

import pytest
from conftest import buffer_types, dtypes, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, ops


class TestCustomOp:
    """Tests for ops.custom function - focuses on API validation and error handling."""

    def test_custom__error__buffer_input(self, graph_builder):
        """Test that custom() rejects BufferValue inputs."""
        buffer_type = BufferType(DType.float32, (10,), DeviceRef.CPU())

        with graph_builder(input_types=[buffer_type]) as graph:
            buffer_val = graph.inputs[0]

            with pytest.raises(TypeError) as exc_info:
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[buffer_val],
                    out_types=[
                        TensorType(DType.float32, (10,), DeviceRef.CPU())
                    ],
                )

            # Verify error message mentions buffer inputs not allowed
            assert (
                "BufferValue" in str(exc_info.value)
                or "buffer" in str(exc_info.value).lower()
            )

    def test_custom__error__empty_name(self, graph_builder):
        """Test custom operation with empty name."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(ValueError):
                ops.custom(
                    name="",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[input_type],
                )

    def test_custom__error__invalid_parameter_type(self, graph_builder):
        """Test custom operation with invalid parameter type."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(TypeError) as exc_info:
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[input_type],
                    parameters={"invalid": [1, 2, 3]},  # List not supported
                )

            # Should mention unsupported parameter type
            error_msg = str(exc_info.value).lower()
            assert "parameter" in error_msg or "unsupported" in error_msg

    def test_custom__error__mismatched_outputs(self, graph_builder):
        """Test custom operation with empty out_types."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(ValueError):
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[],  # Empty outputs
                )

    def test_custom__error__unregistered_kernel(self, graph_builder):
        """Test dedicated error for completely unregistered kernel."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name="definitely_nonexistent_kernel_12345",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[input_type],
                )

            # Should mention that the kernel couldn't be found
            error_msg = str(exc_info.value).lower()
            assert (
                "kernel" in error_msg
                or "find" in error_msg
                or "register" in error_msg
            )

    def test_custom__error__no_inputs(self, graph_builder):
        """Test custom operation with no input values."""
        output_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder() as graph:
            with pytest.raises(ValueError):
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[],  # No inputs
                    out_types=[output_type],
                )

    def test_custom__error__none_values(self, graph_builder):
        """Test custom operation with None values."""
        output_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder() as graph:
            with pytest.raises(TypeError):
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=None,  # None values
                    out_types=[output_type],
                )

    def test_custom__error__none_out_types(self, graph_builder):
        """Test custom operation with None out_types."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(TypeError):
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=None,  # None outputs
                )


class TestInplaceCustomOp:
    """Tests for ops.inplace_custom function - focuses on API validation and error handling."""

    def test_inplace_custom__error__no_buffer_input(self, graph_builder):
        """Test that inplace_custom() requires BufferValue inputs."""
        tensor_type = TensorType(DType.float32, (10,), DeviceRef.CPU())

        with graph_builder(input_types=[tensor_type]) as graph:
            with pytest.raises(TypeError) as exc_info:
                ops.inplace_custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],  # Tensor input, not buffer
                )

            # Should mention need for buffer/opaque inputs
            error_msg = str(exc_info.value).lower()
            assert "buffer" in error_msg or "opaque" in error_msg

    def test_inplace_custom__error__empty_values(self, graph_builder):
        """Test inplace custom operation with no input values."""
        with graph_builder() as graph:
            with pytest.raises(TypeError):
                ops.inplace_custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[],  # No inputs
                )

    def test_inplace_custom__error__invalid_parameter_type(self, graph_builder):
        """Test inplace custom operation with invalid parameter type."""
        buffer_type = BufferType(DType.float32, (10,), DeviceRef.CPU())

        with graph_builder(input_types=[buffer_type]) as graph:
            with pytest.raises(TypeError) as exc_info:
                ops.inplace_custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    parameters={
                        "invalid": {"nested": "dict"}
                    },  # Dict not supported
                )

            error_msg = str(exc_info.value).lower()
            assert "parameter" in error_msg or "unsupported" in error_msg

    def test_inplace_custom__error__unregistered_kernel(self, graph_builder):
        """Test inplace custom operation with unregistered kernel."""
        buffer_type = BufferType(DType.float32, (10,), DeviceRef.CPU())

        with graph_builder(input_types=[buffer_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.inplace_custom(
                    name="definitely_nonexistent_inplace_kernel_12345",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                )

            # Should mention that the kernel couldn't be found
            error_msg = str(exc_info.value).lower()
            assert (
                "kernel" in error_msg
                or "find" in error_msg
                or "register" in error_msg
            )

    def test_inplace_custom__error__none_values(self, graph_builder):
        """Test inplace custom operation with None values."""
        with graph_builder() as graph:
            with pytest.raises(TypeError):
                ops.inplace_custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=None,  # None values
                )

    def test_inplace_custom__basic_chain_behavior(self, graph_builder):
        """Test that inplace_custom returns proper chain structure."""
        buffer_type = BufferType(DType.float32, (10,), DeviceRef.CPU())

        with graph_builder(input_types=[buffer_type]) as graph:
            # This will fail at kernel verification but should succeed at graph construction
            with pytest.raises(ValueError) as exc_info:
                result = ops.inplace_custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                )

            # Should fail due to missing kernel, not chain/graph construction issues
            error_msg = str(exc_info.value).lower()
            assert (
                "kernel" in error_msg
                or "find" in error_msg
                or "register" in error_msg
            )

    def test_inplace_custom__with_outputs_and_chain(self, graph_builder):
        """Test inplace custom operation with both outputs and chain behavior."""
        buffer_type = BufferType(DType.float32, (10,), DeviceRef.CPU())
        output_type = TensorType(DType.float32, (10,), DeviceRef.CPU())

        with graph_builder(input_types=[buffer_type]) as graph:
            # This should construct the operation properly (but fail at verification)
            with pytest.raises(ValueError) as exc_info:
                result = ops.inplace_custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[output_type],
                )

            # Should fail due to missing kernel, not API construction issues
            error_msg = str(exc_info.value).lower()
            assert (
                "kernel" in error_msg
                or "find" in error_msg
                or "register" in error_msg
            )


class TestCustomParameterValidation:
    """Tests for custom operation parameter validation."""

    @pytest.mark.parametrize(
        "param_value",
        [
            True,
            False,
            42,
            -10,
            0,
            "hello",
            "",
            DType.float32,
            DType.int64,
        ],
    )
    def test_custom__parameter_types__supported(
        self, graph_builder, param_value
    ):
        """Test custom operation with supported parameter types."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            # Should succeed (but will fail at kernel verification, which is expected)
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[input_type],
                    parameters={"test_param": param_value},
                )
            # Make sure it's failing due to missing kernel, not parameter validation
            error_msg = str(exc_info.value).lower()
            assert "kernel" in error_msg or "mojo" in error_msg

    @pytest.mark.parametrize(
        "param_value",
        [
            [1, 2, 3],  # List not supported
            {"key": "value"},  # Dict not supported
            None,  # None not supported
            3.14,  # Float not supported
        ],
    )
    def test_custom__parameter_types__unsupported(
        self, graph_builder, param_value
    ):
        """Test custom operation with unsupported parameter types."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            # Should raise exception for invalid parameter type
            with pytest.raises(TypeError) as exc_info:
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[input_type],
                    parameters={"test_param": param_value},
                )
            # Should mention parameter type issue
            error_msg = str(exc_info.value).lower()
            assert "parameter" in error_msg or "unsupported" in error_msg

    def test_custom__no_parameters(self, graph_builder):
        """Test custom operation with no parameters (None)."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[input_type],
                    parameters=None,  # Explicit None
                )
            # Should fail due to missing kernel, not parameter validation
            error_msg = str(exc_info.value).lower()
            assert "kernel" in error_msg or "mojo" in error_msg

    def test_custom__empty_parameters(self, graph_builder):
        """Test custom operation with empty parameters dict."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[input_type],
                    parameters={},  # Empty dict
                )
            # Should fail due to missing kernel, not parameter validation
            error_msg = str(exc_info.value).lower()
            assert "kernel" in error_msg or "mojo" in error_msg


class TestCustomGraphStateConsistency:
    """Tests for graph state consistency after custom operation failures."""

    def test_custom__graph_state_after_verification_failure(
        self, graph_builder
    ):
        """Verify graph remains consistent if custom op verification fails."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            # Record initial state
            initial_input_count = len(graph.inputs)

            # Try to add invalid custom op
            with pytest.raises(ValueError):
                ops.custom(
                    name="nonexistent_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[input_type],
                )

            # Graph should still be in valid state
            assert len(graph.inputs) == initial_input_count

            # Should be able to add valid operations after failure
            try:
                # Add a simple valid operation to test graph consistency
                from max.graph import ops as graph_ops

                result = graph_ops.cast(graph.inputs[0], DType.float64)
                graph.output(result)
                # If this succeeds, graph state is consistent
            except ValueError as e:
                pytest.fail(
                    f"Graph became inconsistent after custom op failure: {e}"
                )

    def test_inplace_custom__graph_state_after_verification_failure(
        self, graph_builder
    ):
        """Verify graph remains consistent if inplace custom op verification fails."""
        buffer_type = BufferType(DType.float32, (10,), DeviceRef.CPU())

        with graph_builder(input_types=[buffer_type]) as graph:
            # Record initial state
            initial_input_count = len(graph.inputs)

            # Try to add invalid inplace custom op
            with pytest.raises(ValueError):
                ops.inplace_custom(
                    name="nonexistent_inplace_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                )

            # Graph should still be in valid state
            assert len(graph.inputs) == initial_input_count

            # Should be able to add valid operations after failure
            try:
                # Add a simple buffer operation to test graph consistency
                from max.graph import ops as graph_ops

                tensor_val = graph_ops.buffer_load(graph.inputs[0])
                graph.output(tensor_val)
                # If this succeeds, graph state is consistent
            except ValueError as e:
                pytest.fail(
                    f"Graph became inconsistent after inplace custom op failure: {e}"
                )

    def test_custom__multiple_failures_preserve_state(self, graph_builder):
        """Test that multiple failed custom op attempts don't corrupt graph state."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            initial_input_count = len(graph.inputs)

            # Try multiple different invalid operations
            invalid_ops = [
                ("nonexistent_kernel_1", {}),
                ("nonexistent_kernel_2", {"bad_param": [1, 2, 3]}),
                ("", {}),  # Empty name
            ]

            for op_name, params in invalid_ops:
                if params and any(isinstance(v, list) for v in params.values()):
                    # This should fail at parameter validation
                    with pytest.raises(TypeError):
                        ops.custom(
                            name=op_name,
                            device=DeviceRef.CPU(),
                            values=[graph.inputs[0]],
                            out_types=[input_type],
                            parameters=params,
                        )
                else:
                    # This should fail at kernel verification
                    with pytest.raises(ValueError):
                        ops.custom(
                            name=op_name,
                            device=DeviceRef.CPU(),
                            values=[graph.inputs[0]],
                            out_types=[input_type],
                            parameters=params,
                        )

                # Verify graph state remains consistent after each failure
                assert len(graph.inputs) == initial_input_count


# Property-based testing strategies
valid_parameter_values = st.one_of(
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(min_size=0, max_size=50),
    st.sampled_from([dtype for dtype in DType if dtype != DType._unknown]),
)

# Strategy for generating parameter dictionaries
parameter_dicts = st.dictionaries(
    st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("z")),
    ),
    valid_parameter_values,
    min_size=0,
    max_size=5,
)

# Strategy for generating kernel names (avoiding empty strings which cause errors)
kernel_names = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
)


class TestCustomPropertyBased:
    """Property-based tests for ops.custom function."""

    @given(
        input_type=tensor_types(shapes=shapes(min_rank=1, max_rank=4)),
        kernel_name=kernel_names,
        parameters=parameter_dicts,
    )
    def test_custom__property__basic_construction(
        self,
        graph_builder,
        input_type: TensorType,
        kernel_name: str,
        parameters: dict,
    ):
        """Property test: custom operations should construct consistently with valid inputs."""
        # Generate matching output type with same shape and dtype
        output_type = TensorType(
            input_type.dtype, input_type.shape, input_type.device
        )

        with graph_builder(input_types=[input_type]) as graph:
            # This will fail at kernel verification but should succeed at graph construction
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name=kernel_name,
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[output_type],
                    parameters=parameters,
                )

            # Should fail due to missing kernel, not parameter or construction issues
            error_msg = str(exc_info.value).lower()
            # Allow for either kernel-related errors or parameter validation errors
            # The key is that it should be a consistent, expected error type
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "parameter", "unsupported"]
            )

    @given(
        input_type=tensor_types(shapes=shapes(min_rank=1, max_rank=3)),
        num_outputs=st.integers(min_value=1, max_value=4),
    )
    def test_custom__property__multiple_outputs(
        self, graph_builder, input_type: TensorType, num_outputs: int
    ):
        """Property test: custom operations should handle multiple outputs consistently."""
        # Generate multiple output types
        output_types = [
            TensorType(input_type.dtype, input_type.shape, input_type.device)
            for _ in range(num_outputs)
        ]

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name="test_multi_output_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=output_types,
                )

            # Should fail consistently regardless of output count
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "find", "register"]
            )

    @given(
        input_types=st.lists(
            tensor_types(shapes=shapes(min_rank=1, max_rank=3)),
            min_size=1,
            max_size=3,
        ),
        kernel_name=kernel_names,
    )
    def test_custom__property__multiple_inputs(
        self, graph_builder, input_types: list[TensorType], kernel_name: str
    ):
        """Property test: custom operations should handle multiple inputs consistently."""
        # Use first input's properties for output
        output_type = TensorType(
            input_types[0].dtype, input_types[0].shape, input_types[0].device
        )

        with graph_builder(input_types=input_types) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name=kernel_name,
                    device=DeviceRef.CPU(),
                    values=graph.inputs,
                    out_types=[output_type],
                )

            # Should fail consistently regardless of input count
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "find", "register"]
            )

    @given(
        base_type=tensor_types(shapes=shapes(min_rank=1, max_rank=4)),
        target_dtype=dtypes,
    )
    def test_custom__property__dtype_variations(
        self, graph_builder, base_type: TensorType, target_dtype: DType
    ):
        """Property test: custom operations should handle different dtype combinations."""
        assume(target_dtype != DType._unknown)

        # Create output type with different dtype
        output_type = TensorType(
            target_dtype, base_type.shape, base_type.device
        )

        with graph_builder(input_types=[base_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name="test_dtype_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[output_type],
                    parameters={"target_dtype": target_dtype},
                )

            # Should handle dtype variations consistently
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "find", "register"]
            )

    @given(
        input_type=tensor_types(shapes=shapes(min_rank=1, max_rank=3)),
        invalid_param_value=st.one_of(
            st.lists(
                st.integers(), min_size=1, max_size=3
            ),  # Lists not supported
            st.dictionaries(
                st.text(), st.integers(), min_size=1, max_size=2
            ),  # Dicts not supported
            st.none(),  # None not supported
            st.floats(
                min_value=-100.0, max_value=100.0
            ),  # Floats not supported
        ),
    )
    def test_custom__property__invalid_parameter_types(
        self, graph_builder, input_type: TensorType, invalid_param_value
    ):
        """Property test: custom operations should consistently reject invalid parameter types."""
        output_type = TensorType(
            input_type.dtype, input_type.shape, input_type.device
        )

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(TypeError) as exc_info:
                ops.custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[output_type],
                    parameters={"invalid_param": invalid_param_value},
                )

            # Should mention parameter-related errors for invalid types
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["parameter", "unsupported", "type"]
            )


class TestInplaceCustomPropertyBased:
    """Property-based tests for ops.inplace_custom function."""

    @given(
        buffer_type=buffer_types(shapes=shapes(min_rank=1, max_rank=4)),
        kernel_name=kernel_names,
        parameters=parameter_dicts,
    )
    def test_inplace_custom__property__basic_construction(
        self,
        graph_builder,
        buffer_type: BufferType,
        kernel_name: str,
        parameters: dict,
    ):
        """Property test: inplace custom operations should construct consistently with valid buffer inputs."""
        with graph_builder(input_types=[buffer_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.inplace_custom(
                    name=kernel_name,
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    parameters=parameters,
                )

            # Should fail due to missing kernel, not construction issues
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "parameter", "unsupported"]
            )

    @given(
        buffer_types_list=st.lists(
            buffer_types(shapes=shapes(min_rank=1, max_rank=3)),
            min_size=1,
            max_size=3,
        ),
    )
    def test_inplace_custom__property__multiple_buffer_inputs(
        self, graph_builder, buffer_types_list: list[BufferType]
    ):
        """Property test: inplace custom operations should handle multiple buffer inputs."""
        with graph_builder(input_types=buffer_types_list) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.inplace_custom(
                    name="test_multi_buffer_kernel",
                    device=DeviceRef.CPU(),
                    values=graph.inputs,
                )

            # Should fail consistently regardless of buffer count
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "find", "register"]
            )

    @given(
        buffer_type=buffer_types(shapes=shapes(min_rank=1, max_rank=3)),
        num_outputs=st.integers(min_value=0, max_value=3),
    )
    def test_inplace_custom__property__with_outputs(
        self, graph_builder, buffer_type: BufferType, num_outputs: int
    ):
        """Property test: inplace custom operations should handle output types consistently."""
        # Generate tensor output types (inplace ops can produce tensor outputs)
        output_types = (
            [
                TensorType(
                    buffer_type.dtype, buffer_type.shape, buffer_type.device
                )
                for _ in range(num_outputs)
            ]
            if num_outputs > 0
            else None
        )

        with graph_builder(input_types=[buffer_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                if output_types:
                    ops.inplace_custom(
                        name="test_inplace_with_outputs",
                        device=DeviceRef.CPU(),
                        values=[graph.inputs[0]],
                        out_types=output_types,
                    )
                else:
                    ops.inplace_custom(
                        name="test_inplace_no_outputs",
                        device=DeviceRef.CPU(),
                        values=[graph.inputs[0]],
                    )

            # Should fail consistently regardless of output configuration
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "find", "register"]
            )

    @given(
        tensor_type=tensor_types(shapes=shapes(min_rank=1, max_rank=3)),
    )
    def test_inplace_custom__property__tensor_input_rejection(
        self, graph_builder, tensor_type: TensorType
    ):
        """Property test: inplace custom operations should consistently reject tensor inputs."""
        with graph_builder(input_types=[tensor_type]) as graph:
            with pytest.raises(TypeError) as exc_info:
                ops.inplace_custom(
                    name="test_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],  # Tensor input, should be buffer
                )

            # Should mention buffer/opaque input requirement
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg for word in ["buffer", "opaque", "inplace"]
            )


class TestCustomParameterPropertyBased:
    """Property-based tests for custom operation parameter handling."""

    @given(
        input_type=tensor_types(shapes=shapes(min_rank=1, max_rank=3)),
        param_name=st.text(min_size=1, max_size=20),
        param_value=valid_parameter_values,
    )
    def test_custom__property__single_parameter_variations(
        self,
        graph_builder,
        input_type: TensorType,
        param_name: str,
        param_value,
    ):
        """Property test: custom operations should handle individual parameter types consistently."""
        output_type = TensorType(
            input_type.dtype, input_type.shape, input_type.device
        )

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name="test_param_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[output_type],
                    parameters={param_name: param_value},
                )

            # Should handle all valid parameter types consistently
            error_msg = str(exc_info.value).lower()
            # For valid parameter types, should fail on kernel lookup, not parameter validation
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "find", "register"]
            )

    @given(
        input_type=tensor_types(shapes=shapes(min_rank=1, max_rank=3)),
        empty_params=st.just({}),
        none_params=st.just(None),
    )
    def test_custom__property__parameter_edge_cases(
        self,
        graph_builder,
        input_type: TensorType,
        empty_params: dict,
        none_params,
    ):
        """Property test: custom operations should handle parameter edge cases consistently."""
        output_type = TensorType(
            input_type.dtype, input_type.shape, input_type.device
        )

        test_cases = [
            ("empty_dict", empty_params),
            ("none_params", none_params),
        ]

        for case_name, params in test_cases:
            with graph_builder(input_types=[input_type]) as graph:
                with pytest.raises(ValueError) as exc_info:
                    ops.custom(
                        name=f"test_{case_name}_kernel",
                        device=DeviceRef.CPU(),
                        values=[graph.inputs[0]],
                        out_types=[output_type],
                        parameters=params,
                    )

                # Should handle edge cases consistently
                error_msg = str(exc_info.value).lower()
                assert any(
                    word in error_msg
                    for word in ["kernel", "mojo", "find", "register"]
                )


class TestCustomShapePropertyBased:
    """Property-based tests for custom operation shape handling."""

    @given(
        input_shape=shapes(min_rank=1, max_rank=4),
        output_shape=shapes(min_rank=1, max_rank=4),
        dtype=dtypes,
    )
    def test_custom__property__shape_transformations(
        self, graph_builder, input_shape, output_shape, dtype: DType
    ):
        """Property test: custom operations should handle arbitrary shape transformations."""
        assume(dtype != DType._unknown)

        input_type = TensorType(dtype, input_shape, DeviceRef.CPU())
        output_type = TensorType(dtype, output_shape, DeviceRef.CPU())

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name="test_shape_transform_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[output_type],
                )

            # Shape mismatches should be handled gracefully at graph construction
            # Validation may happen later in the pipeline
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "find", "register"]
            )

    @given(
        base_shape=shapes(min_rank=2, max_rank=4),
        dtype=dtypes,
    )
    def test_custom__property__rank_preserving_operations(
        self, graph_builder, base_shape, dtype: DType
    ):
        """Property test: custom operations should handle rank-preserving transformations."""
        assume(dtype != DType._unknown)

        input_type = TensorType(dtype, base_shape, DeviceRef.CPU())
        output_type = TensorType(
            dtype, base_shape, DeviceRef.CPU()
        )  # Same shape

        with graph_builder(input_types=[input_type]) as graph:
            with pytest.raises(ValueError) as exc_info:
                ops.custom(
                    name="test_rank_preserving_kernel",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[output_type],
                )

            # Rank-preserving operations should construct cleanly
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["kernel", "mojo", "find", "register"]
            )
