# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import operator
from functools import reduce

import pytest
from conftest import shapes, tensor_types
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from max.graph import Graph, TensorType

input_shapes = st.shared(shapes())
rank = input_shapes.map(len)
within_rank = rank.flatmap(lambda r: st.integers(min_value=0, max_value=r - 1))


@given(
    input_type=tensor_types(shapes=input_shapes),
    normalized_start=within_rank,
    negative_index_start=...,
    normalized_end=within_rank,
    negative_index_end=...,
)
@pytest.mark.skip("MAXPLAT-153")
def test_flatten_success(
    input_type: TensorType,
    normalized_start: int,
    negative_index_start: bool,
    normalized_end: int,
    negative_index_end: bool,
) -> None:
    assume(normalized_start <= normalized_end)

    rank = input_type.rank

    start = normalized_start
    if negative_index_start:
        start -= rank

    end = normalized_end
    if negative_index_end:
        end -= rank

    with Graph("flatten", input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.flatten(start, end)

        assert (
            out.shape[:normalized_start] == input_type.shape[:normalized_start]
        )
        assert (
            out.shape[normalized_start + 1 :]
            == input_type.shape[normalized_end + 1 :]
        )
        assert out.shape[normalized_start] == reduce(
            operator.mul,
            input_type.shape[normalized_start : normalized_end + 1],
            1,
        )

        graph.output(out)


@given(
    input_type=tensor_types(shapes=input_shapes),
    start=...,
    end=...,
)
def test_flatten_failure_indexing(
    input_type: TensorType, start: int, end: int
) -> None:
    rank = input_type.rank
    assume(not ((-rank <= start < rank) and (-rank <= end < rank)))

    with Graph("flatten", input_types=[input_type]) as graph:
        with pytest.raises(IndexError):
            out = graph.inputs[0].tensor.flatten(start, end)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(
    input_type=tensor_types(shapes=input_shapes),
    normalized_start=within_rank,
    negative_index_start=...,
    normalized_end=within_rank,
    negative_index_end=...,
)
def test_flatten_failure_start_greater_end(
    input_type: TensorType,
    normalized_start: int,
    negative_index_start: bool,
    normalized_end: int,
    negative_index_end: bool,
) -> None:
    assume(normalized_start > normalized_end)

    rank = input_type.rank

    start = normalized_start
    if negative_index_start:
        start -= rank

    end = normalized_end
    if negative_index_end:
        end -= rank

    with Graph("flatten", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            out = graph.inputs[0].tensor.flatten(start, end)
