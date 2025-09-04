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
"""Mixed layout implementation that unifies compile-time and runtime indices."""

from builtin.variadics import VariadicOf
from sys.intrinsics import _type_is_eq
from os import abort
from .int_tuple import IntTuple
from .layout import LayoutTrait
from ._mixed_tuple import (
    Idx,
    ComptimeInt,
    RuntimeInt,
    MixedTuple,
    MixedTupleLike,
    to_mixed_int_tuple,
    mixed_int_tuple_to_int_tuple,
    crd2idx,
)


struct MixedLayout[
    shape_types: VariadicOf[MixedTupleLike],
    stride_types: VariadicOf[MixedTupleLike],
](ImplicitlyCopyable, Movable):
    """A layout that supports mixed compile-time and runtime dimensions.

    This layout provides a unified interface for layouts where some dimensions
    are known at compile time and others are determined at runtime. It enables
    more ergonomic layout definitions while maintaining performance.

    Parameters:
        shape_types: The types for the shape dimensions.
        stride_types: The types for the stride dimensions.
    """

    alias has_shape = True
    """Indicates whether the layout has a valid shape."""

    var shape: MixedTuple[*shape_types]
    """The shape of the layout as a mixed tuple."""

    var stride: MixedTuple[*stride_types]
    """The stride of the layout as a mixed tuple."""

    fn __init__(
        out self,
        shape: MixedTuple[*shape_types],
        stride: MixedTuple[*stride_types],
    ):
        """Initialize a mixed layout with shape and stride.

        Args:
            shape: The shape as a MixedTuple.
            stride: The stride as a MixedTuple.
        """
        constrained[
            __type_of(shape).__len__() == __type_of(stride).__len__(),
            String(
                (
                    "Shape and stride must have the same length, but got shape"
                    " length: "
                ),
                __type_of(shape).__len__(),
                " stride length: ",
                __type_of(stride).__len__(),
            ),
        ]()
        self.shape = shape
        self.stride = stride

    fn __call__[index_type: MixedTupleLike](self, index: index_type) -> Int:
        """Maps a logical coordinate to a linear memory index.

        Args:
            index: An IntTuple representing the logical coordinates to map.

        Returns:
            The linear memory index corresponding to the given coordinates.
        """
        return Int(crd2idx(index, self.shape, self.stride))

    fn size(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        For a layout with shape (m, n), this returns m * n, representing
        the total number of valid coordinates in the layout.

        Returns:
            The total number of elements in the layout.
        """
        return self.shape.product()

    fn cosize(self) -> Int:
        """Returns the size of the memory region spanned by the layout.

        For a layout with shape `(m, n)` and stride `(r, s)`, this returns
        `(m-1)*r + (n-1)*s + 1`, representing the memory footprint.

        Returns:
            The size of the memory region required by the layout.
        """
        return self(Idx(self.size() - 1)) + 1

    fn to_layout(self) -> Layout:
        return Layout(
            mixed_int_tuple_to_int_tuple(self.shape),
            mixed_int_tuple_to_int_tuple(self.stride),
        )


# TODO(MOCO-2182): These functions don't work for nested layouts right now.
# E.g. ((5, 4), (3, 2)) should have a row-major stride of ((24, 6), (2, 1)).
# A correct solution may need to recurse into the nested layouts in the return
# type.


fn make_row_major[
    First: MixedTupleLike,
    Second: MixedTupleLike, //,
](shape: MixedTuple[First, Second]) -> __type_of(
    MixedLayout(
        shape=MixedTuple(shape[0], shape[1]),
        stride=MixedTuple(shape[1], Idx[1]()),
    )
):
    return MixedLayout(shape=shape, stride=MixedTuple(shape[1], Idx[1]()))


# TODO: A cleaner solution for changing return type based on argument types than
# overloading, since the current overloading algorithm reports ambiguity.
fn make_row_major[
    First: MixedTupleLike,
    Second: Int,
    Third: Int, //,
](
    shape: MixedTuple[First, ComptimeInt[Second], ComptimeInt[Third]]
) -> __type_of(
    MixedLayout(
        shape=MixedTuple(shape[0], shape[1], shape[2]),
        stride=MixedTuple(
            ComptimeInt[Second * Third](),
            ComptimeInt[Third](),
            Idx[1](),
        ),
    )
):
    return MixedLayout(
        shape=shape,
        stride=MixedTuple(
            ComptimeInt[Second * Third](),
            ComptimeInt[Third](),
            Idx[1](),
        ),
    )


fn make_row_major[
    First: MixedTupleLike,
    Second: MixedTupleLike,
    Third: MixedTupleLike, //,
](shape: MixedTuple[First, Second, Third]) -> __type_of(
    MixedLayout(
        shape=MixedTuple(shape[0], shape[1], shape[2]),
        stride=MixedTuple(
            RuntimeInt(shape[1].product() * shape[2].product()),
            shape[2],
            Idx[1](),
        ),
    )
):
    return MixedLayout(
        shape=shape,
        stride=MixedTuple(
            RuntimeInt(shape[1].product() * shape[2].product()),
            shape[2],
            Idx[1](),
        ),
    )
