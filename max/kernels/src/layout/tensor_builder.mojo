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
Tensor Builder Module

Provides a fluent interface for constructing tensors with various layouts and memory configurations.
It includes utilities for creating both static (compile-time) and dynamic (runtime) tensor dimensions,
supporting row-major, column-major, and custom layouts. The module enables memory placement in different
address spaces (generic, shared, local) and supports features like circular indexing.

Key components:
- `ValueOrUnknown`: Represents static or dynamic tensor dimensions
- `LayoutTensorBuild`: Builder class for tensor construction
- Helper functions for dimension specification and layout creation
"""

from sys import is_gpu

from layout import Layout, LayoutTensor
from layout.layout_tensor import (
    LayoutTensorIter,
    _get_index_type,
    _get_layout_type,
)
from memory import UnsafePointer
from memory.pointer import AddressSpace, _GPUAddressSpace

from utils import Index, IndexList

from .int_tuple import UNKNOWN_VALUE


struct ValueOrUnknown[dim: Int = UNKNOWN_VALUE](Defaultable):
    """
    Represents either a static dimension (known at compile time) or a dynamic dimension (known at runtime).

    Parameters:
        dim: Optional compile-time dimension value. Default is `UNKNOWN_VALUE` for dynamic dimensions.
    """

    var value: Int
    """
    The runtime value of the dimension.

    For static dimensions, this is set to the compile-time value.
    For dynamic dimensions, this is set at runtime.
    """

    fn __init__(out self):
        """
        Initializes a static dimension with compile-time value.

        Note:
            Fails to compile if dim is `UNKNOWN_VALUE`, as dynamic dimensions require a runtime value.
        """
        constrained[
            not dim == UNKNOWN_VALUE,
            "Can't construct a dynamic dim with no runtime value",
        ]()
        self.value = dim

    @implicit
    fn __init__(out self, v: Int):
        """
        Initializes a dynamic dimension with runtime value.

        Args:
            v: Runtime value for the dimension.
        """
        self.value = v


@always_inline
fn static[d: Int]() -> ValueOrUnknown[d]:
    """
    Creates a static dimension with compile-time value.

    Parameters:
        d: The compile-time dimension value to use.

    Returns:
        `ValueOrUnknown[d]` - A static dimension with the given value.
    """
    return ValueOrUnknown[d]()


@always_inline
fn dynamic(d: Int) -> ValueOrUnknown:
    """
    Creates a dynamic dimension with runtime value.

    Args:
        d: Runtime dimension value.

    Returns:
        `ValueOrUnknown` - A dynamic dimension with the given value.
    """
    return ValueOrUnknown(d)


fn _to_int_tuple[n: Int](static_tuple: IndexList[n]) -> IntTuple:
    var int_tuple = IntTuple()

    @parameter
    for i in range(n):
        int_tuple.append(static_tuple[i])
    return int_tuple


fn _to_int_tuple[size: Int](value: Int) -> IntTuple:
    var int_tuple = IntTuple()

    @parameter
    for i in range(size):
        int_tuple.append(value)
    return int_tuple


fn _to_int_tuple[elements: VariadicList[Int]]() -> IntTuple:
    var int_tuple = IntTuple()

    @parameter
    for i in range(len(elements)):
        int_tuple.append(elements[i])
    return int_tuple


@fieldwise_init
@register_passable("trivial")
struct LayoutTensorBuild[
    dtype: DType,
    *,
    __layout: Layout = Layout(1),
    __layout_init: Bool = False,
    __address_space: AddressSpace = _GPUAddressSpace.GENERIC,
    __layout_int_type: DType = _get_layout_type(__layout, __address_space),
    __index_type: DType = _get_index_type(__layout, __address_space),
    __circular: Bool = False,
](Copyable, Defaultable, Movable):
    """
    Tensor layout builder providing a fluent interface for constructing tensors with various layouts.

    Parameters:
        dtype: Data type of tensor elements.
        __layout: The tensor's memory layout.
        __layout_init: Whether the layout has been initialized.
        __address_space: Memory space (generic, shared, local).
        __layout_int_type: Layout index type.
        __index_type: Type used for indexing.
        __circular: Whether tensor has circular indexing semantics.
    """

    var runtime_layout: RuntimeLayout[
        __layout, element_type=__layout_int_type, linear_idx_type=__index_type
    ]
    """
    Runtime representation of the tensor's layout.

    This field stores the layout information that can be manipulated at runtime,
    particularly important for tensors with dynamic dimensions. It encapsulates:
    - The static layout template from `__layout` parameter
    - The bit width for index calculations
    - The appropriate index type based on address space
    """

    fn __init__(out self):
        """
        Initializes a new `LayoutTensorBuild` instance with default values.
        """
        self.runtime_layout = {}

    fn row_major[
        *shapes: Int
    ](
        self,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.row_major(_to_int_tuple[shapes]()),
            __layout_init=True,
        ],
    ):
        """
        Creates a row-major layout using compile-time dimensions.

        Parameters:
            shapes: Variadic parameter specifying the dimensions of the tensor.
                   Each value represents the size of a dimension.

        Returns:
            `LayoutTensorBuild` - A new builder with row-major layout.
        """
        return {}

    fn row_major(
        self,
        shape0: ValueOrUnknown,
        shape1: ValueOrUnknown,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.row_major(shape0.dim, shape1.dim),
            __layout_init=True,
        ],
    ):
        """
        Creates a row-major 2D layout using runtime dimensions.

        Args:
            shape0: First dimension size.
            shape1: Second dimension size.

        Returns:
            `LayoutTensorBuild` - A new builder with row-major layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout).row_major(
                Index[dtype = res.__layout_int_type](shape0.value, shape1.value)
            )
        )

    fn row_major(
        self,
        shape0: ValueOrUnknown,
        shape1: ValueOrUnknown,
        shape2: ValueOrUnknown,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.row_major(shape0.dim, shape1.dim, shape2.dim),
            __layout_init=True,
        ],
    ):
        """
        Creates a row-major 3D layout using runtime dimensions.

        Args:
            shape0: First dimension size.
            shape1: Second dimension size.
            shape2: Third dimension size.

        Returns:
            `LayoutTensorBuild` - A new builder with row-major layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout).row_major(
                Index[dtype = res.__layout_int_type](
                    shape0.value, shape1.value, shape2.value
                )
            )
        )

    fn row_major(
        self,
        shape0: ValueOrUnknown,
        shape1: ValueOrUnknown,
        shape2: ValueOrUnknown,
        shape3: ValueOrUnknown,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.row_major(
                shape0.dim, shape1.dim, shape2.dim, shape3.dim
            ),
            __layout_init=True,
        ],
    ):
        """
        Creates a row-major 4D layout using runtime dimensions.

        Args:
            shape0: First dimension size.
            shape1: Second dimension size.
            shape2: Third dimension size.
            shape3: Fourth dimension size.

        Returns:
            `LayoutTensorBuild` - A new builder with row-major layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout).row_major(
                Index[dtype = res.__layout_int_type](
                    shape0.value, shape1.value, shape2.value, shape3.value
                )
            )
        )

    fn row_major(
        self,
        shape0: ValueOrUnknown,
        shape1: ValueOrUnknown,
        shape2: ValueOrUnknown,
        shape3: ValueOrUnknown,
        shape4: ValueOrUnknown,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.row_major(
                shape0.dim, shape1.dim, shape2.dim, shape3.dim, shape4.dim
            ),
            __layout_init=True,
        ],
    ):
        """
        Creates a row-major 5D layout using runtime dimensions.

        Args:
            shape0: First dimension size.
            shape1: Second dimension size.
            shape2: Third dimension size.
            shape3: Fourth dimension size.
            shape4: Fifth dimension size.

        Returns:
            `LayoutTensorBuild` - A new builder with row-major layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout).row_major(
                Index[dtype = res.__layout_int_type](
                    shape0.value,
                    shape1.value,
                    shape2.value,
                    shape3.value,
                    shape4.value,
                )
            )
        )

    fn col_major[
        *shapes: Int
    ](
        self,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.col_major(_to_int_tuple[shapes]()),
            __layout_init=True,
        ],
    ):
        """
        Creates a column-major layout using compile-time dimensions.

        Parameters:
            shapes: Variadic parameter specifying the dimensions of the tensor.
                   Each value represents the size of a dimension.

        Returns:
            `LayoutTensorBuild` - A new builder with column-major layout.
        """
        return __type_of(res)()

    fn col_major(
        self,
        shape0: ValueOrUnknown,
        shape1: ValueOrUnknown,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.col_major(shape0.dim, shape1.dim),
            __layout_init=True,
        ],
    ):
        """
        Creates a column-major 2D layout using runtime dimensions.

        Args:
            shape0: First dimension size.
            shape1: Second dimension size.

        Returns:
            `LayoutTensorBuild` - A new builder with column-major layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[dtype = res.__layout_int_type](shape0.value, shape1.value)
            )
        )

    fn col_major(
        self,
        shape0: ValueOrUnknown,
        shape1: ValueOrUnknown,
        shape2: ValueOrUnknown,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.col_major(shape0.dim, shape1.dim, shape2.dim),
            __layout_init=True,
        ],
    ):
        """
        Creates a column-major 3D layout using runtime dimensions.

        Args:
            shape0: First dimension size.
            shape1: Second dimension size.
            shape2: Third dimension size.

        Returns:
            `LayoutTensorBuild` - A new builder with column-major layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[dtype = res.__layout_int_type](
                    shape0.value, shape1.value, shape2.value
                )
            )
        )

    fn col_major(
        self,
        shape0: ValueOrUnknown,
        shape1: ValueOrUnknown,
        shape2: ValueOrUnknown,
        shape3: ValueOrUnknown,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.col_major(
                shape0.dim, shape1.dim, shape2.dim, shape3.dim
            ),
            __layout_init=True,
        ],
    ):
        """
        Creates a column-major 4D layout using runtime dimensions.

        Args:
            shape0: First dimension size.
            shape1: Second dimension size.
            shape2: Third dimension size.
            shape3: Fourth dimension size.

        Returns:
            `LayoutTensorBuild` - A new builder with column-major layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[dtype = res.__layout_int_type](
                    shape0.value, shape1.value, shape2.value, shape3.value
                )
            )
        )

    fn col_major(
        self,
        shape0: ValueOrUnknown,
        shape1: ValueOrUnknown,
        shape2: ValueOrUnknown,
        shape3: ValueOrUnknown,
        shape4: ValueOrUnknown,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout.col_major(
                shape0.dim, shape1.dim, shape2.dim, shape3.dim, shape4.dim
            ),
            __layout_init=True,
        ],
    ):
        """
        Creates a column-major 5D layout using runtime dimensions.

        Args:
            shape0: First dimension size.
            shape1: Second dimension size.
            shape2: Third dimension size.
            shape3: Fourth dimension size.
            shape4: Fifth dimension size.

        Returns:
            `LayoutTensorBuild` - A new builder with column-major layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[dtype = res.__layout_int_type](
                    shape0.value,
                    shape1.value,
                    shape2.value,
                    shape3.value,
                    shape4.value,
                )
            )
        )

    fn layout[
        shape0: Int
    ](
        self,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout(shape0),
            __layout_init=True,
        ],
    ):
        """
        Creates a 1D layout with a compile-time dimension.

        Parameters:
            shape0: Size of the single dimension.

        Returns:
            `LayoutTensorBuild` - A new builder with the specified layout.
        """
        return {}

    fn layout[
        rank: Int, shape: IndexList[rank], stride: IndexList[rank]
    ](
        self,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout(_to_int_tuple(shape), _to_int_tuple(stride)),
            __layout_init=True,
        ],
    ):
        """
        Creates a custom layout with compile-time dimensions and strides.

        Parameters:
            rank: Number of dimensions.
            shape: List of dimension sizes.
            stride: List of strides for each dimension.

        Returns:
            `LayoutTensorBuild` - A new builder with the specified custom layout.
        """
        return {}

    fn layout[
        rank: Int
    ](
        self,
        shape: IndexList[rank],
        stride: IndexList[rank],
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout(
                _to_int_tuple[rank](UNKNOWN_VALUE),
                _to_int_tuple[rank](UNKNOWN_VALUE),
            ),
            __layout_init=True,
        ],
    ):
        """
        Creates a custom layout with runtime dimensions and strides.

        Parameters:
            rank: Number of dimensions.

        Args:
            shape: List of dimension sizes.
            stride: List of strides for each dimension.

        Returns:
            `LayoutTensorBuild` - A new builder with the specified custom layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout)(
                shape.cast[__type_of(res.runtime_layout.shape).element_type](),
                stride.cast[
                    __type_of(res.runtime_layout.stride).element_type
                ](),
            )
        )

    fn layout(
        self,
        shape0: ValueOrUnknown,
        out res: LayoutTensorBuild[
            dtype,
            __layout = Layout(
                IntTuple(shape0.dim),
            ),
            __layout_init=True,
        ],
    ):
        """
        Creates a 1D layout with a runtime dimension.

        Args:
            shape0: Size of the single dimension.

        Returns:
            `LayoutTensorBuild` - A new builder with the specified layout.
        """
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[dtype = res.__layout_int_type](shape0.value)
            )
        )

    @always_inline
    fn shared(
        self,
        out res: LayoutTensorBuild[
            dtype,
            __layout=__layout,
            __layout_init=__layout_init,
            __address_space = _GPUAddressSpace.SHARED,
        ],
    ):
        """
        Places the tensor in GPU shared memory.

        Returns:
            `LayoutTensorBuild` - A new builder with shared memory address space.
        """
        constrained[
            is_gpu(),
            "shared memory is supported on NVIDIA and AMD GPU devices only.",
        ]()
        return __type_of(res)(
            rebind[__type_of(res.runtime_layout)](
                self.runtime_layout.cast[
                    res.__layout_int_type, linear_idx_type = res.__index_type
                ]()
            )
        )

    @always_inline
    fn local(
        self,
        out res: LayoutTensorBuild[
            dtype,
            __layout=__layout,
            __layout_init=__layout_init,
            __address_space = _GPUAddressSpace.LOCAL,
        ],
    ):
        """
        Places the tensor in GPU local memory.

        Returns:
            `LayoutTensorBuild` - A new builder with local memory address space.
        """
        constrained[
            is_gpu(),
            "local memory is supported on NVIDIA and AMD GPU devices only.",
        ]()
        return __type_of(res)(
            rebind[__type_of(res.runtime_layout)](
                self.runtime_layout.cast[
                    res.__layout_int_type, linear_idx_type = res.__index_type
                ]()
            )
        )

    @always_inline
    fn alloc(
        self,
        out res: LayoutTensor[
            dtype, __layout, MutableAnyOrigin, address_space=__address_space
        ],
    ):
        """
        Allocates a new tensor using the current layout.

        Returns:
            `LayoutTensor` - A newly allocated tensor with the specified layout

        Note:
            Fails to compile if layout is not set, dimensions are not known, or tensor is circular.
        """
        constrained[__layout_init, "Layout is not set."]()
        constrained[
            __layout.all_dims_known(),
            "Cannot create dynamic tensors on stack.",
        ]()
        constrained[not __circular, "circular tensor not supported!"]()
        return __type_of(res).stack_allocation()

    @always_inline
    fn view[
        address_space: AddressSpace
    ](
        self,
        ptr: UnsafePointer[Scalar[dtype], address_space=address_space],
        out res: LayoutTensor[
            dtype,
            __layout,
            ptr.origin,
            address_space=address_space,
            layout_int_type=__layout_int_type,
            linear_idx_type=__index_type,
        ],
    ):
        """
        Creates a tensor view over existing memory.

        Parameters:
            address_space: Memory address space for the tensor (generic, shared, local).

        Args:
            ptr: Pointer to memory region to create the view over.

        Returns:
            `LayoutTensor` - A tensor view over the specified memory region with the current layout.

        Note:
            Fails to compile if layout is not set, address spaces don't match, or tensor is circular.
        """
        constrained[__layout_init == True, "Layout is not set."]()
        constrained[__address_space == address_space, ""]()
        constrained[not __circular, "circular tensor not supported!"]()

        @parameter
        if __layout.all_dims_known():
            return __type_of(res)(ptr)
        else:
            return __type_of(res)(
                ptr,
                self.runtime_layout.cast[
                    res.layout_int_type, linear_idx_type = res.linear_idx_type
                ](),
            )

    @always_inline
    fn circular(
        self,
        out res: LayoutTensorBuild[
            dtype,
            __layout=__layout,
            __layout_init=__layout_init,
            __address_space=__address_space,
            __circular=True,
        ],
    ):
        """
        Enables circular indexing for the tensor.

        Returns:
            `LayoutTensorBuild` - A new builder with circular indexing enabled.
        """
        return __type_of(res)(
            self.runtime_layout.cast[
                res.__layout_int_type, linear_idx_type = res.__index_type
            ]()
        )

    @always_inline
    fn iter(
        self,
        ptr: UnsafePointer[Scalar[dtype], address_space=__address_space],
        bound: Int,
        out res: LayoutTensorIter[
            dtype,
            __layout,
            ptr.origin,
            address_space=__address_space,
            layout_int_type=__layout_int_type,
            linear_idx_type=__index_type,
            circular=__circular,
        ],
    ):
        """
        Creates an iterator over tensor elements.

        Args:
            ptr: Pointer to memory region.
            bound: Upper bound for iteration.

        Returns:
            `LayoutTensorIter` - An iterator over tensor elements.

        Note:
            Fails to compile if layout is not set or dimensions are not known.
        """
        constrained[__layout_init, "Layout is not set."]()
        constrained[
            __layout.all_dims_known(),
            "Cannot create dynamic iterator",
        ]()
        return __type_of(res)(ptr, bound, self.runtime_layout)
