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

from collections.abc import Collection

import pytest
from conftest import shapes, static_dims, symbolic_dims, tensor_types
from hypothesis import assume, example, given
from hypothesis import strategies as st
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, Shape, StaticDim, TensorType


def test_reshape() -> None:
    """Builds a simple graph with a reshape and checks the IR."""
    with Graph(
        "reshape",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[6, 5], device=DeviceRef.CPU()
            ),
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.CPU(),
            ),
        ],
    ) as graph:
        static_reshape = graph.inputs[0].tensor.reshape((3, 10))
        static_reshape_neg_one = graph.inputs[0].tensor.reshape((2, -1))
        assert static_reshape_neg_one.shape == [2, 15]

        symbolic_reshape = graph.inputs[1].tensor.reshape(("channels", "batch"))
        symbolic_reshape_neg_one = graph.inputs[1].tensor.reshape(
            ("channels", -1)
        )
        assert symbolic_reshape_neg_one.shape == ["channels", "batch"]

        graph.output(
            static_reshape,
            static_reshape_neg_one,
            symbolic_reshape,
            symbolic_reshape_neg_one,
        )


def subseqs(c: Collection):
    if not c:
        return st.just(type(c)())
    subseq_indices = st.sets(st.sampled_from(range(len(c))))
    return subseq_indices.map(
        lambda indices: type(c)(v for i, v in enumerate(c) if i in indices)  # type: ignore
    )


def negative_one_reshape(shapes):  # noqa: ANN001
    return (
        shapes.flatmap(subseqs)
        .map(lambda subseq: [*subseq, -1])
        .flatmap(st.permutations)
    )


shared_shapes = st.shared(shapes())
# Use a max rank of 4 to reduce the probability of drawing 1-dims.
shared_static_shapes = st.shared(shapes(dims=static_dims()))


@given(
    input_type=tensor_types(shapes=shared_shapes),
    output_shape=shared_shapes.flatmap(st.permutations),
)
def test_reshape__can_permute_input_shape(
    input_type: TensorType, output_shape: list[Dim]
) -> None:
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.reshape(output_shape)
        assert out.shape == output_shape
        graph.output(out)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    reshape_shape=negative_one_reshape(shared_shapes),
)
@pytest.mark.skip("MAXPLAT-151")
def test_reshapes__can_replace_any_dims_with_negative_one(
    input_type: TensorType, reshape_shape: list[Dim]
) -> None:
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.reshape(reshape_shape)
        assert out.dtype == input_type.dtype
        for dim, expected in zip(out.shape, reshape_shape):
            if expected != -1:
                assert dim == expected
        graph.output(out)


@given(
    input_type=tensor_types(shapes=shapes(include_dims=[0])),
    reshape_shape=shapes(include_dims=[0]),
)
def test_reshapes__zero_dim(
    input_type: TensorType, reshape_shape: list[Dim]
) -> None:
    assume(0 in input_type.shape)
    assume(0 in reshape_shape)
    assume(  # TODO (MSDK-763): remove this assumption
        all(
            d in input_type.shape
            for d in reshape_shape
            if not isinstance(d, StaticDim)
        )
    )
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.reshape(reshape_shape)
        assert out.dtype == input_type.dtype
        assert out.shape == reshape_shape
        graph.output(out)


def shapes_plus_ones(shapes=shapes()):  # noqa: ANN001
    ones = st.lists(st.just(1))
    shapes = shapes.flatmap(lambda shape: ones.map(lambda ones: shape + ones))
    return shapes.flatmap(st.permutations)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    reshape_shape=shapes_plus_ones(shared_shapes),
)
def test_reshapes__unsqueeze(
    input_type: TensorType, reshape_shape: list[Dim]
) -> None:
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.reshape(reshape_shape)
        assert out.dtype == input_type.dtype
        assert out.shape == reshape_shape
        graph.output(out)


@given(
    input_type=tensor_types(shapes=shapes_plus_ones(shared_shapes)),
    reshape_shape=shared_shapes,
)
def test_reshapes__squeeze(
    input_type: TensorType, reshape_shape: list[Dim]
) -> None:
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.reshape(reshape_shape)
        assert out.dtype == input_type.dtype
        assert out.shape == reshape_shape
        graph.output(out)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    output_shape=shared_shapes.flatmap(st.permutations),
    dim=symbolic_dims,
)
@pytest.mark.skip(reason="MAXPLAT-151")
def test_reshape__fails_with_different_symbolic_dim(
    input_type: TensorType,
    output_shape: list[Dim],
    dim: Dim,
) -> None:
    assume(dim not in input_type.shape)
    with Graph("reshape", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].tensor.reshape([*output_shape, dim])


@given(
    input_type=tensor_types(shapes=shared_static_shapes),
    output_shape=shared_static_shapes.flatmap(st.permutations)
    .filter(lambda shape: shape[-1] > 1)
    .map(lambda shape: shape[:-1]),
)
@example(
    # Specifically test an example whose dim product can be represented by an
    # int64, but not by an int32.
    input_type=TensorType(
        DType.int8, Shape([268435456, 17]), device=DeviceRef.CPU()
    ),
    output_shape=Shape([268435456]),
)
@pytest.mark.skip(reason="MAXPLAT-151")
def test_reshape__fails_with_different_number_of_elements(
    input_type: TensorType,
    output_shape: Shape,
) -> None:
    with Graph("reshape", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].tensor.reshape(output_shape)


@given(
    input_type=tensor_types(shapes=st.lists(st.just(1))),
    output_shape=st.lists(st.just(1)),
)
def test_reshape__can_reshape_single_element_tensors(
    input_type: TensorType,
    output_shape: list[Dim],
) -> None:
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.reshape(output_shape)
        assert out.dtype == input_type.dtype
        assert out.shape == output_shape
        graph.output(out)


def test_MAXPLAT_328() -> None:
    input_type = TensorType(
        DType.float32, ["n_patches", 2048], DeviceRef.from_device(CPU())
    )
    with Graph("test_MAXPLAT_328", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        x = x.tensor.rebind([Dim("n_patches_over_4") * 4, 2048])
        n_patches, _ = x.shape
        graph.output(x.reshape([n_patches // 4, 4, 2048]))


def test_MAXPLAT_328_no_new_parameter() -> None:
    input_type = TensorType(
        DType.float32, ["n_patches", 2048], DeviceRef.from_device(CPU())
    )
    with Graph("test_MAXPLAT_328", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        n_patches, _ = x.tensor.shape
        x = x.tensor.rebind([(n_patches // 4) * 4, 2048])
        graph.output(x.reshape([n_patches // 4, 4, 2048]))


@pytest.mark.skip("MAXPLAT-330: This is currently a compile-time error")
def test_reshape_statically_known_impossible_shape() -> None:
    input_type = TensorType(
        DType.float32, [7, 4], device=DeviceRef.from_device(CPU())
    )

    with Graph("reshape", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        x = x.tensor.rebind([Dim("n_patches_over_4") * 4, 4])
        n_patches, _ = x.shape
        with pytest.raises(Exception):
            x.reshape([n_patches // 4, 4, 4])


@pytest.mark.skip(
    "MAXPLAT-329: Point users towards using a rebind before reshape"
)
def test_reshape_needs_rebind_error_message() -> None:
    input_type = TensorType(
        DType.float32, ["n_patches", 2048], DeviceRef.from_device(CPU())
    )
    with Graph("test_MAXPLAT_329", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        n_patches, _ = x.tensor.shape
        graph.output(x.tensor.reshape([n_patches // 4, 4, 2048]))
