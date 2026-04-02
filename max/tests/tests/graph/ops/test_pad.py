# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""ops.pad tests."""

import pytest
from conftest import GraphBuilder, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, StaticDim, TensorType, ops

input_types = st.shared(tensor_types())


def padded_size(shape: Shape, padding: list[int]) -> int:
    total = 1

    for i, s in enumerate(shape):
        if not isinstance(s, StaticDim):
            continue

        low = padding[2 * i]
        high = padding[2 * i + 1]
        total *= low + s.dim + high

    return total


def paddings_for(input_types, low=0, high=16):  # noqa: ANN001, ANN201
    return input_types.flatmap(
        lambda type: st.lists(
            st.integers(min_value=low, max_value=high),
            min_size=2 * type.rank,
            max_size=2 * type.rank,
        )
    )


@given(input_type=input_types, paddings=paddings_for(input_types, low=-16))
def test_negative_paddings(
    graph_builder: GraphBuilder,
    input_type: TensorType,
    paddings: list[int],
) -> None:
    """Padding by nothing does not change the type."""
    assume(input_type.rank > 0)
    assume(any(x < 0 for x in paddings))

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            _ = ops.pad(graph.inputs[0].tensor, paddings=paddings, value=0)


@given(input_type=input_types)
def test_no_padding(
    graph_builder: GraphBuilder, input_type: TensorType
) -> None:
    """Padding by nothing does not change the type."""
    assume(input_type.rank > 0)
    paddings = [0] * (2 * input_type.rank)

    with graph_builder(input_types=[input_type]) as graph:
        out = ops.pad(graph.inputs[0].tensor, paddings=paddings, value=0)
        assert out.type == input_type
        graph.output(out)


@given(input_type=input_types, paddings=paddings_for(input_types))
def test_positive_paddings(
    graph_builder: GraphBuilder,
    input_type: TensorType,
    paddings: list[int],
) -> None:
    """Test random paddings."""

    assume(0 < input_type.rank)
    with graph_builder(input_types=[input_type]) as graph:
        assume(padded_size(input_type.shape, paddings) < 2**63)

        out = ops.pad(graph.inputs[0].tensor, paddings=paddings, value=0)
        assert out.dtype == input_type.dtype
        graph.output(out)


# ---------------------------------------------------------------------------
# mode='reflect' tests
# ---------------------------------------------------------------------------


def test_reflect_output_shape() -> None:
    """Reflect pad produces the correct output shape."""
    with Graph(
        "pad_reflect",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[3, 4], device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        out = ops.pad(
            graph.inputs[0].tensor, paddings=[1, 2, 0, 1], mode="reflect"
        )
        assert out.shape == [6, 5]
        assert out.dtype == DType.float32
        graph.output(out)


def test_reflect_preserves_dtype() -> None:
    """Reflect pad does not change the element type."""
    for dtype in (DType.float32, DType.float16, DType.int32):
        with Graph(
            "pad_reflect_dtype",
            input_types=[
                TensorType(dtype=dtype, shape=[4, 4], device=DeviceRef.CPU())
            ],
        ) as graph:
            out = ops.pad(
                graph.inputs[0].tensor, paddings=[1, 1, 1, 1], mode="reflect"
            )
            assert out.dtype == dtype
            graph.output(out)


def test_reflect_zero_padding_is_noop() -> None:
    """Zero paddings with reflect mode return the same type."""
    input_type = TensorType(
        dtype=DType.float32, shape=[3, 3], device=DeviceRef.CPU()
    )
    with Graph("pad_reflect_noop", input_types=[input_type]) as graph:
        out = ops.pad(
            graph.inputs[0].tensor, paddings=[0, 0, 0, 0], mode="reflect"
        )
        assert out.type == input_type
        graph.output(out)


@given(input_type=input_types, paddings=paddings_for(input_types))
def test_reflect_positive_paddings(
    graph_builder: GraphBuilder,
    input_type: TensorType,
    paddings: list[int],
) -> None:
    """Reflect pad with arbitrary non-negative paddings builds a valid graph."""
    assume(input_type.rank > 0)
    with graph_builder(input_types=[input_type]) as graph:
        assume(padded_size(input_type.shape, paddings) < 2**63)
        out = ops.pad(graph.inputs[0].tensor, paddings=paddings, mode="reflect")
        assert out.dtype == input_type.dtype
        graph.output(out)


# ---------------------------------------------------------------------------
# mode='edge' tests
# ---------------------------------------------------------------------------


def test_edge_output_shape() -> None:
    """Edge pad produces the correct output shape."""
    with Graph(
        "pad_edge",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[2, 5], device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        out = ops.pad(
            graph.inputs[0].tensor, paddings=[3, 1, 0, 2], mode="edge"
        )
        assert out.shape == [6, 7]
        assert out.dtype == DType.float32
        graph.output(out)


def test_edge_preserves_dtype() -> None:
    """Edge pad does not change the element type."""
    for dtype in (DType.float32, DType.float16, DType.int32):
        with Graph(
            "pad_edge_dtype",
            input_types=[
                TensorType(dtype=dtype, shape=[4, 4], device=DeviceRef.CPU())
            ],
        ) as graph:
            out = ops.pad(
                graph.inputs[0].tensor, paddings=[2, 2, 0, 0], mode="edge"
            )
            assert out.dtype == dtype
            graph.output(out)


def test_edge_zero_padding_is_noop() -> None:
    """Zero paddings with edge mode return the same type."""
    input_type = TensorType(
        dtype=DType.float32, shape=[3, 3], device=DeviceRef.CPU()
    )
    with Graph("pad_edge_noop", input_types=[input_type]) as graph:
        out = ops.pad(
            graph.inputs[0].tensor, paddings=[0, 0, 0, 0], mode="edge"
        )
        assert out.type == input_type
        graph.output(out)


@given(input_type=input_types, paddings=paddings_for(input_types))
def test_edge_positive_paddings(
    graph_builder: GraphBuilder,
    input_type: TensorType,
    paddings: list[int],
) -> None:
    """Edge pad with arbitrary non-negative paddings builds a valid graph."""
    assume(input_type.rank > 0)
    with graph_builder(input_types=[input_type]) as graph:
        assume(padded_size(input_type.shape, paddings) < 2**63)
        out = ops.pad(graph.inputs[0].tensor, paddings=paddings, mode="edge")
        assert out.dtype == input_type.dtype
        graph.output(out)


# ---------------------------------------------------------------------------
# Shared error cases
# ---------------------------------------------------------------------------


def test_unsupported_mode_raises() -> None:
    """An unsupported mode raises ValueError with a helpful message."""
    with Graph(
        "pad_bad_mode",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[3, 3], device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        with pytest.raises(ValueError, match="unsupported padding mode"):
            ops.pad(graph.inputs[0].tensor, paddings=[1, 1, 1, 1], mode="wrap")  # type: ignore[arg-type]


@given(input_type=input_types, paddings=paddings_for(input_types, low=-16))
@pytest.mark.parametrize("mode", ["reflect", "edge"])
def test_negative_paddings_reflect_edge(
    graph_builder: GraphBuilder,
    input_type: TensorType,
    paddings: list[int],
    mode: str,
) -> None:
    """Negative paddings raise ValueError for reflect and edge modes too."""
    assume(input_type.rank > 0)
    assume(any(x < 0 for x in paddings))

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            ops.pad(graph.inputs[0].tensor, paddings=paddings, mode=mode)  # type: ignore[arg-type]
