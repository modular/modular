# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import BufferType, BufferValue, Graph


@given(buffer_type=...)
def test_buffer_value(buffer_type: BufferType):
    with Graph(
        "buffer",
        input_types=[
            buffer_type,
        ],
    ) as graph:
        buffer = graph.inputs[0]
        type = buffer.type
        assert isinstance(buffer, BufferValue)
        assert type == buffer_type
