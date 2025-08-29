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
"""ops.irfft tests."""

import pytest
from conftest import static_axes, tensor_types
from hypothesis import assume, example, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, Shape, TensorType, ops
from max.graph.ops.irfft import Normalization

input_types = st.shared(
    tensor_types(dtypes=st.just(DType.float32), device=DeviceRef.GPU())
)


def expected_output_shape(shape: Shape, n: int | None, axis: int) -> Shape:
    expected_shape = Shape(shape)
    if n is None:
        n = 2 * (int(shape[axis]) - 1)
    expected_shape[axis] = Dim(n)
    return expected_shape


@given(
    input_type=input_types,
    n=st.integers(min_value=1, max_value=1024),
    axis=static_axes(input_types),
    normalization=st.sampled_from(
        [item for item in Normalization]
        + [item.value for item in Normalization]
    ),
)
@example(
    input_type=TensorType(DType.float32, (24,), DeviceRef.GPU()),
    n=3,
    axis=-1,
    normalization="backward",
)
@example(
    input_type=TensorType(DType.float32, (5, 10, 15), DeviceRef.GPU()),
    n=3,
    axis=0,
    normalization="ortho",
)
@example(
    input_type=TensorType(DType.float32, (5, 10, 15), DeviceRef.GPU()),
    n=None,
    axis=1,
    normalization="backward",
)
@example(
    input_type=TensorType(DType.float32, (1, 2, 3), DeviceRef.GPU()),
    n=None,
    axis=2,
    normalization=Normalization.FORWARD,
)
def test_irfft(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    n: int | None,
    axis: int,
    normalization: Normalization | str,
) -> None:
    """Padding by nothing does not change the type."""
    assume(input_type.dtype == DType.float32)
    assume(input_type.rank > 0)

    with graph_builder(input_types=[input_type]) as graph:
        out = ops.irfft(
            graph.inputs[0].tensor, n=n, axis=axis, normalization=normalization
        )
        assert out.type.shape == expected_output_shape(
            input_type.shape, n, axis
        )
        assert out.type.dtype == input_type.dtype


def test_invalid_normalization() -> None:
    input_type = TensorType(DType.float32, (1, 2, 3), DeviceRef.GPU())
    with Graph("irfft", input_types=[input_type]) as graph:
        with pytest.raises(ValueError, match="Invalid normalization: invalid"):
            ops.irfft(
                graph.inputs[0].tensor,
                n=1,
                axis=0,
                normalization="invalid",
            )


def test_invalid_dim() -> None:
    input_type = TensorType(DType.float32, ("batch", 2, 3), DeviceRef.GPU())
    with Graph("irfft", input_types=[input_type]) as graph:
        with pytest.raises(ValueError, match="Axis dimension must be static"):
            ops.irfft(graph.inputs[0].tensor, n=1, axis=0)


def test_invalid_device() -> None:
    input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())
    with Graph("irfft", input_types=[input_type]) as graph:
        with pytest.raises(
            ValueError, match="IRFFT is currently only supported on GPU."
        ):
            ops.irfft(graph.inputs[0].tensor, n=1, axis=0)
