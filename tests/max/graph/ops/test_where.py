# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import re
from functools import reduce
from typing import Optional

import numpy as np
import pytest
from conftest import (
    broadcast_shapes,
    broadcastable_tensor_types,
    tensor_types,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.type import Dim, StaticDim


@given(input_types=broadcastable_tensor_types(3))
def test_where(input_types: list[TensorType]) -> None:
    input_types[0].dtype = DType.bool

    with Graph("where", input_types=input_types) as graph:
        cond, x, y = graph.inputs
        out = ops.where(cond, x, y)

        expected = reduce(broadcast_shapes, (t.shape for t in input_types))
        assert out.shape == expected
        assert out.dtype in (t.dtype for t in input_types)

        graph.output(out)


def not_broadcastable(*shapes: list[Dim]) -> bool:
    def broadcast_dim(d1: Optional[Dim], d2: Optional[Dim]) -> bool:
        if d1 is None or d2 is None:
            return True  # Can broadcast with None
        return d1 == d2 or d1 == StaticDim(1) or d2 == StaticDim(1)

    # Get the maximum rank among all shapes
    max_rank = max(len(shape) for shape in shapes)

    # Check each dimension position
    for i in range(max_rank):
        dims = []
        for shape in shapes:
            # Get dimension at position i (from right), or None if out of bounds
            idx = len(shape) - max_rank + i
            dim = shape[idx] if 0 <= idx < len(shape) else None
            dims.append(dim)

        # Check if all dimensions at this position can broadcast
        for j in range(len(dims)):
            for k in range(j + 1, len(dims)):
                if not broadcast_dim(dims[j], dims[k]):
                    return True  # Found non-broadcastable dimensions

    return False  # All dimensions can broadcast


shared_dtypes = st.shared(st.from_type(DType))


@given(
    condition=tensor_types(dtypes=st.just(DType.bool)),
    x=tensor_types(dtypes=shared_dtypes),
    y=tensor_types(dtypes=shared_dtypes),
)
def test_where_with_non_broadcastable_shapes(
    graph_builder, condition, x, y
) -> None:
    assume(not_broadcastable(condition.shape, x.shape, y.shape))
    with Graph("where", input_types=[condition, x, y]) as graph:
        cond, x, y = graph.inputs
        with pytest.raises(ValueError):
            ops.where(cond, x, y)


def test_where_error_message_with_non_bool_condition() -> None:
    with Graph(
        "where_non_bool",
        input_types=[
            TensorType(shape=[3], dtype=DType.float32, device=DeviceRef.CPU()),
            TensorType(shape=[3], dtype=DType.int32, device=DeviceRef.CPU()),
            TensorType(shape=[3], dtype=DType.int32, device=DeviceRef.CPU()),
        ],
    ) as graph:
        cond, x, y = graph.inputs
        with pytest.raises(
            ValueError,
            match="Expected condition to be a boolean tensor, but got a tensor with dtype DType.float32",
        ):
            ops.where(cond, x, y)


def test_where_error_message_with_mismatched_condition_shape() -> None:
    with Graph(
        "where_mismatched_condition_shape",
        input_types=[
            TensorType(shape=[2, 4], dtype=DType.bool, device=DeviceRef.CPU()),
            TensorType(shape=[2, 6], dtype=DType.int32, device=DeviceRef.CPU()),
            TensorType(shape=[2], dtype=DType.int32, device=DeviceRef.CPU()),
        ],
    ) as graph:
        cond, x, y = graph.inputs
        with pytest.raises(
            ValueError,
            match="are neither equivalent nor broadcastable",
        ):
            ops.where(cond, x, y)


def test_where_error_message_with_mismatched_devices() -> None:
    with Graph(
        "where_mismatched_devices",
        input_types=[
            TensorType(shape=[3], dtype=DType.bool, device=DeviceRef.CPU()),
            TensorType(shape=[3], dtype=DType.int32, device=DeviceRef.CPU()),
            TensorType(shape=[3], dtype=DType.int32, device=DeviceRef.GPU()),
        ],
    ) as graph:
        cond, x, y = graph.inputs
        with pytest.raises(
            ValueError,
            match=re.escape(
                "All tensors must be on the same device, but got devices: cpu:0, cpu:0, gpu:0"
            ),
        ):
            ops.where(cond, x, y)


# The next two tests for dtype promotion rules are pretty non-trivial. Using Cursor AI,
# we've attempted to read the promotion logic in RMO::promoteDtype and validate the
# behavior with these two tests, one for cases where promotion should work for where
# and one for cases where it should not.


@given(
    condition=tensor_types(dtypes=st.just(DType.bool)),
    x_dtype=st.sampled_from(
        [
            # Bool promotes to any type
            DType.bool,
            # UInt promotes to higher bitwidth uint or any float
            DType.uint8,
            DType.uint16,
            DType.uint32,
            DType.uint64,
            # SInt promotes to higher bitwidth sint or any float
            DType.int8,
            DType.int16,
            DType.int32,
            DType.int64,
            # Float promotes to higher bitwidth float
            DType.float16,
            DType.float32,
            DType.float64,
        ]
    ),
    y_dtype=st.sampled_from(
        [
            # Bool promotes to any type
            DType.bool,
            # UInt promotes to higher bitwidth uint or any float
            DType.uint8,
            DType.uint16,
            DType.uint32,
            DType.uint64,
            # SInt promotes to higher bitwidth sint or any float
            DType.int8,
            DType.int16,
            DType.int32,
            DType.int64,
            # Float promotes to higher bitwidth float
            DType.float16,
            DType.float32,
            DType.float64,
        ]
    ),
)
def test_where_with_promotable_dtypes(
    graph_builder, condition, x_dtype, y_dtype
) -> None:
    """Test where with dtypes that should promote according to RMO::promoteDtype rules."""
    # Skip cases where we expect promotion to fail
    assume(
        not (
            # Skip unsafe uint->sint promotions
            (
                x_dtype.is_integral()
                and x_dtype.is_unsigned_integral()
                and y_dtype.is_integral()
                and y_dtype.is_signed_integral()
                and x_dtype.size_in_bytes >= y_dtype.size_in_bytes
            )
            # Skip unsafe uint->sint promotions (reversed)
            or (
                y_dtype.is_integral()
                and y_dtype.is_unsigned_integral()
                and x_dtype.is_integral()
                and x_dtype.is_signed_integral()
                and y_dtype.size_in_bytes >= x_dtype.size_in_bytes
            )
            # Skip unsafe int->float promotions
            or (
                # x is integral, y is float
                (
                    x_dtype.is_integral()
                    and y_dtype.is_float()
                    and (
                        # Skip if integral type is larger than float in bytes
                        x_dtype.size_in_bytes > y_dtype.size_in_bytes
                        # Skip if integral type's precision exceeds float's precision
                        or (
                            y_dtype == DType.float16
                            and (
                                (
                                    x_dtype.is_signed_integral()
                                    and x_dtype.size_in_bytes * 8 - 1 > 10
                                )
                                or (
                                    x_dtype.is_unsigned_integral()
                                    and x_dtype.size_in_bytes * 8 > 10
                                )
                            )
                        )
                        or (
                            y_dtype == DType.float32
                            and (
                                (
                                    x_dtype.is_signed_integral()
                                    and x_dtype.size_in_bytes * 8 - 1 > 23
                                )
                                or (
                                    x_dtype.is_unsigned_integral()
                                    and x_dtype.size_in_bytes * 8 > 23
                                )
                            )
                        )
                        or (
                            y_dtype == DType.float64
                            and (
                                (
                                    x_dtype.is_signed_integral()
                                    and x_dtype.size_in_bytes * 8 - 1 > 52
                                )
                                or (
                                    x_dtype.is_unsigned_integral()
                                    and x_dtype.size_in_bytes * 8 > 52
                                )
                            )
                        )
                    )
                )
                # y is integral, x is float
                or (
                    y_dtype.is_integral()
                    and x_dtype.is_float()
                    and (
                        # Skip if integral type is larger than float in bytes
                        y_dtype.size_in_bytes > x_dtype.size_in_bytes
                        # Skip if integral type's precision exceeds float's precision
                        or (
                            x_dtype == DType.float16
                            and (
                                (
                                    y_dtype.is_signed_integral()
                                    and y_dtype.size_in_bytes * 8 - 1 > 10
                                )
                                or (
                                    y_dtype.is_unsigned_integral()
                                    and y_dtype.size_in_bytes * 8 > 10
                                )
                            )
                        )
                        or (
                            x_dtype == DType.float32
                            and (
                                (
                                    y_dtype.is_signed_integral()
                                    and y_dtype.size_in_bytes * 8 - 1 > 23
                                )
                                or (
                                    y_dtype.is_unsigned_integral()
                                    and y_dtype.size_in_bytes * 8 > 23
                                )
                            )
                        )
                        or (
                            x_dtype == DType.float64
                            and (
                                (
                                    y_dtype.is_signed_integral()
                                    and y_dtype.size_in_bytes * 8 - 1 > 52
                                )
                                or (
                                    y_dtype.is_unsigned_integral()
                                    and y_dtype.size_in_bytes * 8 > 52
                                )
                            )
                        )
                    )
                )
            )
            # Skip unsafe uint->float16/float32 promotions
            or (
                x_dtype.is_integral()
                and x_dtype.is_unsigned_integral()
                and (y_dtype == DType.float16 or y_dtype == DType.float32)
            )
            or (
                y_dtype.is_integral()
                and y_dtype.is_unsigned_integral()
                and (x_dtype == DType.float16 or x_dtype == DType.float32)
            )
            # Skip different float formats with same virtual bitwidth
            or (x_dtype == DType.float16 and y_dtype == DType.bfloat16)
            or (x_dtype == DType.bfloat16 and y_dtype == DType.float16)
        )
    )

    x = TensorType(x_dtype, condition.shape, condition.device)
    y = TensorType(y_dtype, condition.shape, condition.device)

    with graph_builder(input_types=[condition, x, y]) as graph:
        cond, x, y = graph.inputs
        out = ops.where(cond, x, y)

        # Bool always promotes to other type
        if x_dtype == DType.bool:
            assert out.dtype == y_dtype
        elif y_dtype == DType.bool:
            assert out.dtype == x_dtype
        # For numeric types, should promote to max category and bitwidth
        else:
            # Categories align with RMOOps.cpp's DTypeCategory enum:
            # 0 = kBool, 1 = kUInt, 2 = kSInt, 3 = kFloat
            # This hierarchy is used for type promotion: bool < unsigned int < signed int < float
            x_category = (
                0
                if x_dtype == DType.bool
                else 1
                if x_dtype.is_unsigned_integral()
                else 2
                if x_dtype.is_signed_integral()
                else 3
            )
            y_category = (
                0
                if y_dtype == DType.bool
                else 1
                if y_dtype.is_unsigned_integral()
                else 2
                if y_dtype.is_signed_integral()
                else 3
            )
            max_category = max(x_category, y_category)
            max_size_in_bytes = max(
                x_dtype.size_in_bytes, y_dtype.size_in_bytes
            )

            if max_category == 1:  # UInt
                assert (
                    out.dtype.is_unsigned_integral()
                    and out.dtype.size_in_bytes == max_size_in_bytes
                )
            elif max_category == 2:  # SInt
                assert (
                    out.dtype.is_signed_integral()
                    and out.dtype.size_in_bytes == max_size_in_bytes
                )
            else:  # Float
                assert (
                    out.dtype.is_float()
                    and out.dtype.size_in_bytes == max_size_in_bytes
                )


@given(
    condition=tensor_types(dtypes=st.just(DType.bool)),
    x_dtype=st.sampled_from(
        [
            # Unsafe uint->sint with same bitwidth
            DType.uint32,
            DType.uint64,
            # Unsafe int->float when float can't represent all values
            DType.int64,
            DType.uint64,
            # Different float formats with same virtual bitwidth
            DType.float16,
            DType.bfloat16,
        ]
    ),
    y_dtype=st.sampled_from(
        [
            # Unsafe uint->sint with same bitwidth
            DType.int32,
            DType.int64,
            # Unsafe int->float when float can't represent all values
            DType.float32,
            DType.float64,
            # Different float formats with same virtual bitwidth
            DType.bfloat16,
            DType.float16,
        ]
    ),
)
def test_where_with_incompatible_dtypes(
    graph_builder, condition, x_dtype, y_dtype
) -> None:
    """Test where with dtypes that should fail to promote according to RMO::promoteDtype rules."""
    assume(
        # Unsafe uint->sint promotions with same bitwidth
        (
            x_dtype.is_integral()
            and x_dtype.is_unsigned_integral()
            and y_dtype.is_integral()
            and y_dtype.is_signed_integral()
            and x_dtype.size_in_bytes == y_dtype.size_in_bytes
        )
        # Unsafe int->float promotions
        or (
            x_dtype.is_integral()
            and y_dtype.is_float()
            and (
                # Skip if integral type is larger than float in bytes
                x_dtype.size_in_bytes > y_dtype.size_in_bytes
                # Skip if integral type's precision exceeds float's precision
                or (
                    y_dtype == DType.float16
                    and (
                        (
                            x_dtype.is_signed_integral()
                            and x_dtype.size_in_bytes * 8 - 1 > 10
                        )
                        or (
                            x_dtype.is_unsigned_integral()
                            and x_dtype.size_in_bytes * 8 > 10
                        )
                    )
                )
                or (
                    y_dtype == DType.float32
                    and (
                        (
                            x_dtype.is_signed_integral()
                            and x_dtype.size_in_bytes * 8 - 1 > 23
                        )
                        or (
                            x_dtype.is_unsigned_integral()
                            and x_dtype.size_in_bytes * 8 > 23
                        )
                    )
                )
                or (
                    y_dtype == DType.float64
                    and (
                        (
                            x_dtype.is_signed_integral()
                            and x_dtype.size_in_bytes * 8 - 1 > 52
                        )
                        or (
                            x_dtype.is_unsigned_integral()
                            and x_dtype.size_in_bytes * 8 > 52
                        )
                    )
                )
            )
        )
        # Unsafe uint->float16/float32 promotions
        or (
            x_dtype.is_integral()
            and x_dtype.is_unsigned_integral()
            and (y_dtype == DType.float16 or y_dtype == DType.float32)
        )
        # Different float formats with same virtual bitwidth
        or (x_dtype == DType.float16 and y_dtype == DType.bfloat16)
        or (x_dtype == DType.bfloat16 and y_dtype == DType.float16)
    )

    x = TensorType(x_dtype, condition.shape, condition.device)
    y = TensorType(y_dtype, condition.shape, condition.device)

    with graph_builder(input_types=[condition, x, y]) as graph:
        cond, x, y = graph.inputs
        with pytest.raises(
            ValueError, match="Failed to resolve valid dtype: Unsafe cast from"
        ):
            ops.where(cond, x, y)


# Like the Dtype promotion tests above, these two tests validate the behavior of
# where with Python scalar values and NumPy arrays. They use hypothesis to generate
# python and numpy values that should be safe and unsafe for the where operation.
# They use fixed values for the tensor shapes, to isolate the tests only on the python
# and numpy values.


@given(
    # Generate Python integers that are within the range of float16
    safe_int=st.integers(min_value=-(2**15), max_value=2**15 - 1),
    # Generate Python integers that are too large for float16
    unsafe_int_float16=st.integers(min_value=2**16, max_value=2**32 - 1),
    # Generate Python integers that are too large for float32
    unsafe_int_float32=st.integers(min_value=2**32, max_value=2**64 - 1),
    # Generate Python floats that are within the range of float32
    safe_float=st.floats(
        min_value=-3.4e38,
        max_value=3.4e38,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_where_with_python_scalars(
    safe_int, unsafe_int_float16, unsafe_int_float32, safe_float
) -> None:
    """Test where with Python scalar values (int, float) using hypothesis to probe boundaries."""
    with Graph(
        "where_python_scalars",
        input_types=[
            TensorType(DType.bool, [2, 2], DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], DeviceRef.CPU()),
            TensorType(DType.int32, [2, 2], DeviceRef.CPU()),
            TensorType(DType.float16, [2, 2], DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], DeviceRef.CPU()),
        ],
    ) as graph:
        (
            cond,
            float32_tensor1,
            float32_tensor2,
            int32_tensor,
            float16_tensor,
            float32_tensor3,
        ) = graph.inputs

        # Test successful promotions
        # Python int -> float32 (safe case)
        out = ops.where(cond, safe_int, float32_tensor1)
        assert out.dtype == DType.float32

        # Python float -> float32 (safe case)
        out = ops.where(cond, float32_tensor2, safe_float)
        assert out.dtype == DType.float32

        # Python int -> int32 (safe case)
        out = ops.where(cond, safe_int, int32_tensor)
        assert out.dtype == DType.int32

        # Test failed promotions
        # Python int too large for float16
        with pytest.raises(
            ValueError, match="Unsafe cast: Can't promote python int"
        ):
            ops.where(cond, unsafe_int_float16, float16_tensor)

        # Python int too large for float32
        with pytest.raises(
            ValueError, match="Unsafe cast: Can't promote python int"
        ):
            ops.where(cond, unsafe_int_float32, float32_tensor3)


@given(
    # Generate NumPy int32 arrays that are within the range of float32
    safe_int32=st.lists(
        st.integers(min_value=-(2**23), max_value=2**23 - 1),
        min_size=4,
        max_size=4,
    ).map(lambda x: np.array(x, dtype=np.int32).reshape(2, 2)),
    # Generate NumPy float64 arrays that are within the range of float32
    safe_float64=st.lists(
        st.floats(
            min_value=-3.4e38,
            max_value=3.4e38,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=4,
        max_size=4,
    ).map(lambda x: np.array(x, dtype=np.float64).reshape(2, 2)),
    # Generate NumPy int64 arrays that are too large for float32
    unsafe_int64=st.lists(
        st.integers(min_value=2**32, max_value=2**63 - 1),
        min_size=4,
        max_size=4,
    ).map(lambda x: np.array(x, dtype=np.int64).reshape(2, 2)),
    # Generate NumPy uint64 arrays that are too large for float32
    unsafe_uint64=st.lists(
        st.integers(min_value=2**32, max_value=2**64 - 1),
        min_size=4,
        max_size=4,
    ).map(lambda x: np.array(x, dtype=np.uint64).reshape(2, 2)),
)
def test_where_with_numpy_arrays(
    safe_int32, safe_float64, unsafe_int64, unsafe_uint64
) -> None:
    """Test where with NumPy arrays using hypothesis to probe boundaries."""
    with Graph(
        "where_numpy_arrays",
        input_types=[
            TensorType(DType.bool, [2, 2], DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], DeviceRef.CPU()),
            TensorType(DType.int64, [2, 2], DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], DeviceRef.CPU()),
        ],
    ) as graph:
        (
            cond,
            float32_tensor1,
            float32_tensor2,
            int64_tensor,
            float32_tensor3,
            float32_tensor4,
        ) = graph.inputs

        # Test successful promotions
        # NumPy int32 -> float32
        out = ops.where(cond, safe_int32, float32_tensor1)
        assert out.dtype == DType.float32

        # NumPy float64 -> float32
        out = ops.where(cond, safe_float64, float32_tensor2)
        assert out.dtype == DType.float32

        # NumPy int32 -> int64
        out = ops.where(cond, safe_int32, int64_tensor)
        assert out.dtype == DType.int64

        # Test failed promotions
        # NumPy int64 too large for float32
        with pytest.raises(
            ValueError, match="Unsafe cast: Can't promote numpy integer array"
        ):
            ops.where(cond, unsafe_int64, float32_tensor3)

        # NumPy uint64 too large for float32
        with pytest.raises(
            ValueError, match="Unsafe cast: Can't promote numpy integer array"
        ):
            ops.where(cond, unsafe_uint64, float32_tensor4)
