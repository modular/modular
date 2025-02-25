# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from sys import bitwidthof, is_gpu

from layout import Layout, LayoutTensor
from layout.layout_tensor import LayoutTensorIter, _get_index_type
from memory import UnsafePointer
from memory.pointer import AddressSpace, _GPUAddressSpace

from utils import Index, IndexList, StaticTuple

from .int_tuple import UNKNOWN_VALUE


struct ValueOrUnknown[dim: Int = UNKNOWN_VALUE]:
    var value: Int

    fn __init__(out self):
        constrained[
            not dim == UNKNOWN_VALUE,
            "Can't construct a dynamic dim with no runtime value",
        ]()
        self.value = dim

    @implicit
    fn __init__(out self, v: Int):
        self.value = v


@always_inline
fn static[d: Int]() -> ValueOrUnknown[d]:
    return ValueOrUnknown[d]()


@always_inline
fn dynamic(d: Int) -> ValueOrUnknown:
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


@value
@register_passable("trivial")
struct LayoutTensorBuild[
    dtype: DType,
    *,
    __layout: Layout = Layout(1),
    __layout_init: Bool = False,
    __address_space: AddressSpace = _GPUAddressSpace.GENERIC,
    __layout_bitwidth: Int = bitwidthof[_get_index_type(__address_space)](),
    __circular: Bool = False,
]:
    var runtime_layout: RuntimeLayout[__layout, bitwidth=__layout_bitwidth]

    fn __init__(out self):
        self.runtime_layout = __type_of(self.runtime_layout)()

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
        return __type_of(res)()

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
        return __type_of(res)(
            __type_of(res.runtime_layout).row_major(
                Index[unsigned=True, element_bitwidth = res.__layout_bitwidth](
                    shape0.value, shape1.value
                )
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
        return __type_of(res)(
            __type_of(res.runtime_layout).row_major(
                Index[unsigned=True, element_bitwidth = res.__layout_bitwidth](
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
        return __type_of(res)(
            __type_of(res.runtime_layout).row_major(
                Index[unsigned=True, element_bitwidth = res.__layout_bitwidth](
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
        return __type_of(res)(
            __type_of(res.runtime_layout).row_major(
                Index[unsigned=True, element_bitwidth = res.__layout_bitwidth](
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
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[unsigned=True, element_bitwidth = res.__layout_bitwidth](
                    shape0.value, shape1.value
                )
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
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[unsigned=True, element_bitwidth = res.__layout_bitwidth](
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
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[unsigned=True, element_bitwidth = res.__layout_bitwidth](
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
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[unsigned=True, element_bitwidth = res.__layout_bitwidth](
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
        return __type_of(res)()

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
        return __type_of(res)()

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
        return __type_of(res)(__type_of(res.runtime_layout)(shape, stride))

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
        return __type_of(res)(
            __type_of(res.runtime_layout).col_major(
                Index[unsigned=True, element_bitwidth = res.__layout_bitwidth](
                    shape0.value
                )
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
        constrained[
            is_gpu(),
            "shared memory is supported on NVIDIA and AMD GPU devices only.",
        ]()
        return __type_of(res)(
            rebind[__type_of(res.runtime_layout)](
                self.runtime_layout.cast[
                    _get_index_type(_GPUAddressSpace.SHARED)
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
        constrained[
            is_gpu(),
            "local memory is supported on NVIDIA and AMD GPU devices only.",
        ]()
        return __type_of(res)(
            rebind[__type_of(res.runtime_layout)](
                self.runtime_layout.cast[
                    _get_index_type(_GPUAddressSpace.LOCAL)
                ]()
            )
        )

    @always_inline
    fn alloc(
        self,
        out res: LayoutTensor[dtype, __layout, address_space=__address_space],
    ):
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
            mut = ptr.mut,
            origin = ptr.origin,
            address_space=address_space,
        ],
    ):
        constrained[__layout_init == True, "Layout is not set."]()
        constrained[__address_space == address_space, ""]()
        constrained[not __circular, "circular tensor not supported!"]()

        @parameter
        if __layout.all_dims_known():
            return __type_of(res)(ptr)
        else:
            return __type_of(res)(
                ptr, rebind[__type_of(res.runtime_layout)](self.runtime_layout)
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
        return __type_of(res)(
            rebind[__type_of(res.runtime_layout)](self.runtime_layout)
        )

    @always_inline
    fn iter(
        self,
        ptr: UnsafePointer[Scalar[dtype], address_space=__address_space],
        bound: Int,
        out res: LayoutTensorIter[
            dtype,
            __layout,
            mut = ptr.mut,
            origin = ptr.origin,
            address_space=__address_space,
            circular=__circular,
        ],
    ):
        constrained[__layout_init, "Layout is not set."]()
        constrained[
            __layout.all_dims_known(),
            "Cannot create dynamic iterator",
        ]()
        return __type_of(res)(ptr, bound, self.runtime_layout)
