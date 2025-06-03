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
"""
Provides the `RuntimeLayout` type and functions for working with it. You can use
`RuntimeLayout` to define a layout where the dimensions are not known at compile
time.

You can import these APIs from `layout.runtime_layout`.

```mojo
from layout.runtime_layout import RuntimeLayout, make_layout
```
"""

from sys import bitwidthof

from utils import IndexList

from . import IntTuple, Layout
from .int_tuple import UNKNOWN_VALUE, flatten
from .layout import coalesce as coalesce_layout
from .layout import composition as composition_layout
from .layout import is_tuple
from .layout import make_layout as make_layout_static
from .runtime_tuple import (
    RuntimeTuple,
    crd2idx,
    product,
    idx2crd_int_tuple,
    idx2crd,
)

# A `Layout` like type that uses RuntimeTuple as its storage instead of
# IntTuple.


@register_passable("trivial")
struct RuntimeLayout[
    layout: Layout,
    /,
    *,
    element_type: DType = DType.int64,
    linear_idx_type: DType = DType.int64,
](Stringable, Writable):
    """A runtime-configurable layout that uses `RuntimeTuple` for storage.

    This struct provides a layout implementation that can be modified at runtime,
    unlike the static [`Layout`](/mojo/kernels/layout/layout/Layout) type. It
    uses [`RuntimeTuple`](/mojo/kernels/layout/runtime_tuple/RuntimeTuple) for
    shape and stride storage.

    Parameters:
        layout: The static `Layout` type to base this runtime layout on.
        element_type: The integer type of the each dimension element. Must be signed.
        linear_idx_type: The integer type of the linear index into memory returned by `crd2idx`. Must be signed.

    The layout must have statically known dimensions at compile time, but the
    actual shape and stride values can be modified during execution.
    """

    var shape: RuntimeTuple[layout.shape, element_type=element_type]
    """The shape of the layout as a runtime tuple.

    Stores the size of each dimension. Uses the specified bitwidth and is
    unsigned. Must match the static layout's shape dimensions.
    """

    var stride: RuntimeTuple[layout.stride, element_type=linear_idx_type]
    """The stride of the layout as a runtime tuple.

    Stores the stride (step size) for each dimension. Uses 64-bit unsigned
    integers since strides can be large values. Must match the static layout's
    stride dimensions.
    """

    @always_inline
    fn __init__(out self):
        """Initialize a `RuntimeLayout` with default values.

        Creates a new `RuntimeLayout` instance with default shape and stride
        values. Requires that the static layout has known dimensions at compile
        time.

        Constraints:
            The static layout that this runtime layout is based on must have all
            dimensions known.
        """

        constrained[
            layout.all_dims_known(), "Static layout with known dims is required"
        ]()

        self.shape = __type_of(self.shape)()
        self.stride = __type_of(self.stride)()

    @always_inline
    fn __init__(
        out self,
        shape: RuntimeTuple[layout.shape, element_type=element_type],
        stride: RuntimeTuple[layout.stride, element_type=linear_idx_type],
    ):
        """Initialize a `RuntimeLayout` with specified shape and stride.

        Args:
            shape: A `RuntimeTuple` containing the dimensions of each axis.
            stride: A `RuntimeTuple` containing the stride values for each axis.
        """

        self.shape = shape
        self.stride = stride

    # FIXME: This should probably better done in the RuntimeTuple constructor
    @always_inline
    fn __call__(self, idx: Int) -> Scalar[linear_idx_type]:
        """Convert a single index to a flat linear index.

        Args:
            idx: The one-dimensional index to convert.

        Returns:
            The corresponding flat linear index in the layout.
        """
        return self.__call__(RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx))

    @always_inline
    fn __call__[
        t: IntTuple
    ](self, idx: RuntimeTuple[t, **_]) -> Scalar[linear_idx_type]:
        """Convert a multi-dimensional index to a flat linear index.

        Parameters:
            t: The `IntTuple` type for the index.

        Args:
            idx: A `RuntimeTuple` containing the multi-dimensional coordinates.

        Returns:
            The corresponding flat linear index in the layout.
        """
        return crd2idx[out_type=linear_idx_type](idx, self.shape, self.stride)

    @always_inline("nodebug")
    fn idx2crd[
        t: IntTuple
    ](self, idx: RuntimeTuple[t, **_]) -> RuntimeTuple[
        idx2crd_int_tuple(t, layout.shape, layout.stride),
        element_type=element_type,
    ]:
        """Converts a linear index to logical coordinates.

        This is the inverse operation of the __call__ method, mapping from
        a memory index back to the corresponding logical coordinates.

        Parameters:
            t: The `IntTuple` type for the index.

        Args:
            idx: The linear index to convert.

        Returns:
            The logical coordinates corresponding to the given index.
        """
        return idx2crd(idx, self.shape, self.stride)

    @always_inline
    fn size(self) -> Int:
        """Calculate the total number of elements in the layout.

        Returns:
            The product of all dimensions in the shape, representing the total
            number of elements that can be addressed by this layout.
        """
        return product(self.shape)

    @always_inline
    fn bound_check_required(self) -> Bool:
        """Determine if bounds checking is required for this layout.

        Returns:
            True if any dimension in the shape differs from the static layout's
            shape, False otherwise.
        """

        @parameter
        for i in range(layout.rank()):
            alias dim_i = Int(layout.shape[i])
            if self.shape.value[i] != dim_i:
                return True
        return False

    @always_inline
    fn cast[
        element_type: DType,
        /,
        *,
        linear_idx_type: DType = linear_idx_type,
    ](
        self,
        out result: RuntimeLayout[
            layout, element_type=element_type, linear_idx_type=linear_idx_type
        ],
    ):
        """Cast the layout to use a different element bitwidth.

        Parameters:
            element_type: The target data type.
            linear_idx_type: The target linear idx type.

        Returns:
            A new `RuntimeLayout` with the shape cast to the specified type.
        """
        return __type_of(result)(
            self.shape.cast[element_type](), self.stride.cast[linear_idx_type]()
        )

    @no_inline
    fn __str__(self) -> String:
        """Convert the layout to a string representation.

        Returns:
            A string representation of the layout.
        """
        return String.write(self)

    @staticmethod
    fn row_major[
        rank: Int, //
    ](
        shape: IndexList[rank, **_],
        out result: RuntimeLayout[
            layout,
            element_type=element_type,
            linear_idx_type=linear_idx_type,
        ],
    ):
        """Create a row-major layout from the given shape.

        In row-major layout, elements with adjacent rightmost indices are
        adjacent in memory.

        Parameters:
            rank: The number of dimensions in the layout.

        Args:
            shape: An `IndexList` containing the dimensions of each axis.

        Returns:
            A `RuntimeLayout` with row-major stride ordering.
        """

        constrained[
            shape.element_type == element_type,
            String(
                "Element type mismatch, shape has element type",
                shape.element_type,
                "but layout has element type",
                element_type,
                sep=" ",
            ),
        ]()

        var stride = IndexList[rank, element_type=linear_idx_type]()
        var c_stride = 1
        stride[rank - 1] = c_stride

        @parameter
        for i in reversed(range(rank - 1)):
            var dim = shape[i + 1]
            stride[i] = dim * c_stride
            c_stride *= dim
        return __type_of(result)(shape, stride)

    @staticmethod
    fn col_major[
        rank: Int, //
    ](
        shape: IndexList[rank, **_],
        out result: RuntimeLayout[
            layout,
            element_type=element_type,
            linear_idx_type=linear_idx_type,
        ],
    ):
        """Create a column-major layout from the given shape.

        In column-major layout, elements with adjacent leftmost indices are
        adjacent in memory.

        Parameters:
            rank: The number of dimensions in the layout.

        Args:
            shape: An `IndexList` containing the dimensions of each axis.

        Returns:
            A `RuntimeLayout` with column-major stride ordering.
        """

        constrained[
            shape.element_type == element_type,
            String(
                "Element type mismatch, shape has element type",
                shape.element_type,
                "but layout has element type",
                element_type,
                sep=" ",
            ),
        ]()

        var stride = IndexList[rank, element_type=linear_idx_type]()
        var c_stride = 1
        stride[0] = c_stride

        @parameter
        for i in range(1, rank):
            var dim = shape[i - 1]
            stride[i] = dim * c_stride
            c_stride *= dim
        return __type_of(result)(shape, stride)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Write a string representation of the layout to a writer.

        Parameters:
            W: The `Writer` type.

        Args:
            writer: The `Writer` object to write the layout representation to.
        """

        writer.write("(")
        writer.write(self.shape)
        writer.write(":")
        writer.write(self.stride)
        writer.write(")")

    fn sublayout[
        i: Int
    ](
        self,
        out result: RuntimeLayout[
            layout[i],
            element_type=element_type,
            linear_idx_type=linear_idx_type,
        ],
    ):
        """Extract a nested sublayout at the specified index.

        Parameters:
            i: The index of the nested layout to extract.

        Returns:
            A `RuntimeLayout` representing the nested layout at index i.
        """
        return __type_of(result)(
            rebind[RuntimeTuple[layout[i].shape, element_type=element_type]](
                self.shape[i]
            ),
            rebind[
                RuntimeTuple[layout[i].stride, element_type=linear_idx_type]
            ](self.stride[i]),
        )

    fn dim(self, i: Int) -> Int:
        """Get the size of the dimension at the specified index.

        Args:
            i: The index of the dimension to retrieve.

        Returns:
            The size of the dimension at index `i`.
        """
        return self.shape.value[i]

    @staticmethod
    fn __len__() -> Int:
        """Get the number of dimensions in the layout.

        Returns:
            The number of dimensions (rank) of the layout.
        """
        return len(layout)


fn coalesce[
    l: Layout,
    keep_rank: Bool = False,
](
    layout: RuntimeLayout[l, **_],
    out result: RuntimeLayout[
        coalesce_layout(l, keep_rank),
        element_type = layout.element_type,
        linear_idx_type = layout.linear_idx_type,
    ],
):
    """Coalesce adjacent dimensions in a runtime layout when possible.

    This optimizes the layout by merging adjacent dimensions when their
    relationship allows it, potentially reducing the number of dimensions.

    Parameters:
        l: The static layout type to coalesce.
        keep_rank: Whether to maintain the original rank (currently unsupported).

    Args:
        layout: The input `RuntimeLayout` to coalesce.

    Returns:
        A new `RuntimeLayout` with coalesced dimensions.
    """

    constrained[not keep_rank, "Unsupported coalesce mode"]()

    var res_shape = RuntimeTuple[
        coalesce_layout(l, keep_rank).shape, element_type = layout.element_type
    ]()
    var res_stride = RuntimeTuple[
        coalesce_layout(l, keep_rank).stride,
        element_type = layout.linear_idx_type,
    ]()

    res_shape.value[0] = 1
    res_stride.value[0] = 0

    var idx = 0

    @parameter
    for i in range(len(flatten(l.shape))):
        alias shape = Int(l.shape[i])
        alias stride = Int(l.stride[i])

        # If dynamic, append new mode
        if UNKNOWN_VALUE in (shape, stride):
            res_shape.value[idx] = layout.shape.value[i]
            res_stride.value[idx] = layout.stride.value[i]
            idx += 1
            continue

        # skip their shape-1s
        if shape == 1:
            continue

        # replace our shape-1 with anything
        if res_shape.value[idx] == 1:
            res_shape.value[idx] = layout.shape.value[i]
            res_stride.value[idx] = layout.stride.value[i]

        # merge modes if the shape*stride match
        elif res_shape.value[idx] * res_shape.value[idx] == stride:
            res_shape.value[idx] = res_shape.value[idx] * shape
        # append a new mode
        else:
            res_shape.value[idx] = layout.shape.value[i]
            res_stride.value[idx] = layout.stride.value[i]
            idx += 1

    return __type_of(result)(res_shape, res_stride)


fn make_layout[
    l1: Layout, l2: Layout, /, *, linear_idx_type: DType = DType.uint64
](
    a: RuntimeLayout[l1, **_],
    b: RuntimeLayout[l2, **_],
    out result: RuntimeLayout[
        make_layout_static(l1, l2),
        element_type = b.element_type,
        linear_idx_type=linear_idx_type,
    ],
):
    """Combine two runtime layouts into a single composite layout.

    This creates a new layout by concatenating the dimensions and strides of the
    input layouts.

    Parameters:
        l1: The static layout type of `a`.
        l2: The static layout type of `b`.
        linear_idx_type: The integer type of the all index calculated by the returned
                  runtime layout.

    Args:
        a: The first `RuntimeLayout` to combine.
        b: The second `RuntimeLayout` to combine.

    Returns:
        A new `RuntimeLayout` with dimensions from both input layouts.
    """

    var res_shape = RuntimeTuple[
        make_layout_static(l1, l2).shape,
        element_type = b.element_type,
    ]()
    var res_stride = RuntimeTuple[
        make_layout_static(l1, l2).stride,
        element_type=linear_idx_type,
    ]()

    alias a_length = len(flatten(l1.shape))
    alias b_length = len(flatten(l2.shape))

    @parameter
    for i in range(a_length):
        res_shape.value[i] = a.shape.value[i]
        res_stride.value[i] = a.stride.value[i]

    @parameter
    for i in range(b_length):
        res_shape.value[a_length + i] = b.shape.value[i]
        res_stride.value[a_length + i] = b.stride.value[i]

    return __type_of(result)(res_shape, res_stride)
