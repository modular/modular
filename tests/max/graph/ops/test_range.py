# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.range tests."""

from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, ops

sized_int = st.integers(min_value=-(2**63), max_value=2**63 - 1)


@given(start=sized_int, stop=sized_int, step=sized_int)
def test_range(start: int, stop: int, step: int) -> None:
    assume(step != 0)
    assume((step > 0 and stop >= start) or (step < 0 and stop <= start))

    with Graph("range", input_types=()) as graph:
        dim = (stop - start) // step
        start = ops.constant(start, DType.int64)
        stop = ops.constant(stop, DType.int64)
        step = ops.constant(step, DType.int64)
        out = ops.range(start, stop, step, dim)
        graph.output(out)
