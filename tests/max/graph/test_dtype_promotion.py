# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph dtype promotion."""

import numpy as np
import pytest
from hypothesis import event, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, TensorType, dtype_promotion


@given(graph_type=..., scalar=...)
def test_promote_weak_dtypes__python_float(
    graph_type: TensorType, scalar: float
) -> None:
    with Graph("promote_weak_dtypes", input_types=[graph_type]) as graph:
        if graph_type.dtype in [
            DType.bfloat16,
            DType.float16,
            DType.float32,
            DType.float64,
        ]:
            # float to float will succeed.
            v1, v2 = dtype_promotion._promote_weak_dtypes(
                graph.inputs[0], scalar
            )

            assert v1.dtype == graph_type.dtype
            assert v2.dtype == graph_type.dtype
        else:
            # float to int will fail.
            with pytest.raises(ValueError, match="Unsafe cast"):
                v1, v2 = dtype_promotion._promote_weak_dtypes(
                    graph.inputs[0], scalar
                )


@given(graph_type=..., scalar=...)
def test_promote_weak_dtypes__python_int(
    graph_type: TensorType, scalar: int
) -> None:
    with Graph("promote_weak_dtypes", input_types=[graph_type]) as graph:
        try:
            v1, v2 = dtype_promotion._promote_weak_dtypes(
                graph.inputs[0], scalar
            )

            assert v1.dtype == graph_type.dtype
            assert v2.dtype == graph_type.dtype
            event("types promote")
        except ValueError as e:
            assert "Unsafe cast" in str(e)
            event("types don't promote")


# TODO: This should also sample from bfloat16, but numpy doesn't support bfloat16 arrays.
float_dtype = st.sampled_from([DType.float16, DType.float32, DType.float64])


@given(graph_type=..., np_dtype=float_dtype, value=...)
def test_promote_weak_dtypes__np_float(
    graph_type: TensorType, np_dtype: DType, value: float
) -> None:
    with Graph("promote_weak_dtypes", input_types=[graph_type]) as graph:
        np_const = np.array(value).astype(np_dtype.to_numpy())
        if graph_type.dtype in [
            DType.bfloat16,
            DType.float16,
            DType.float32,
            DType.float64,
        ]:
            # float to float will succeed.
            v1, v2 = dtype_promotion._promote_weak_dtypes(
                graph.inputs[0], np_const
            )

            assert v1.dtype == graph_type.dtype
            assert v2.dtype == graph_type.dtype
        else:
            # float to int will fail.
            with pytest.raises(ValueError, match="Unsafe cast"):
                v1, v2 = dtype_promotion._promote_weak_dtypes(
                    graph.inputs[0], np_const
                )


int_dtype = st.sampled_from(
    [
        DType.bool,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    ]
)


@given(
    graph_type=...,
    np_dtype=int_dtype,
    value=st.integers(-(2**63), 2**63 - 1),
)
def test_promote_weak_dtypes__np_int(
    graph_type: TensorType, np_dtype: DType, value: int
) -> None:
    with Graph("promote_weak_dtypes", input_types=[graph_type]) as graph:
        np_const = np.array(value).astype(np_dtype.to_numpy())
        try:
            v1, v2 = dtype_promotion._promote_weak_dtypes(
                graph.inputs[0], np_const
            )

            assert v1.dtype == graph_type.dtype
            assert v2.dtype == graph_type.dtype
            event("types promote")
        except ValueError as e:
            assert "Unsafe cast" in str(e)
            event("types don't promote")
