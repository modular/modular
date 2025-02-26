# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import alignof, bitwidthof

from memory import AddressSpace, UnsafePointer

from utils import IndexList

from . import Layout, RuntimeLayout
from .int_tuple import UNKNOWN_VALUE


@always_inline
fn _get_offset[i: Int](runtime_layout: RuntimeLayout) -> Int:
    """Returns the offset for a single index into the runtime layout.

    Parameters:
        i: The index to get the offset for.

    Args:
        runtime_layout: The runtime layout to get the offset from.

    Returns:
        The offset value for the given index.
    """

    @parameter
    if runtime_layout.layout.all_dims_known():
        alias offset = runtime_layout.layout(i)
        return offset
    else:
        return runtime_layout(i)


@always_inline
fn _get_offset[i: Int, j: Int](runtime_layout: RuntimeLayout) -> Int:
    """Returns the offset for a 2D index into the runtime layout.

    Parameters:
        i: The first index to get the offset for.
        j: The second index to get the offset for.

    Args:
        runtime_layout: The runtime layout to get the offset from.

    Returns:
        The offset value for the given indices.
    """

    @parameter
    if runtime_layout.layout.all_dims_known():
        alias offset = runtime_layout.layout(IntTuple(i, j))
        return offset
    else:
        return runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)](i, j)
        )


# Element is a wrapper around SIMD type, it extends the SIMD type to define
# a vectorized load / store that is driven by the layout of the element.
#
struct Element[
    dtype: DType, layout: Layout, /, *, bitwidth: Int = bitwidthof[Int]()
](Stringable, Writable):
    alias element_data_type = SIMD[dtype, size = layout.size()]

    var element_data: Self.element_data_type

    var runtime_layout: RuntimeLayout[layout, bitwidth=bitwidth]

    @implicit
    fn __init__(out self, element_data: Self.element_data_type):
        self.element_data = element_data
        self.runtime_layout = RuntimeLayout[layout, bitwidth=bitwidth]()

    fn __init__(
        mut self,
        element_data: Self.element_data_type,
        runtime_layout: RuntimeLayout[layout, bitwidth=bitwidth],
    ):
        self.element_data = element_data
        self.runtime_layout = runtime_layout

    @always_inline("nodebug")
    @staticmethod
    fn load(
        ptr: UnsafePointer[Scalar[dtype], **_],
        runtime_layout: RuntimeLayout[
            layout, bitwidth=bitwidth
        ] = RuntimeLayout[layout, bitwidth=bitwidth](),
    ) -> Self:
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()

        var element_data = Self.element_data_type()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                alias alignment = alignof[Self.element_data_type]()
                return ptr.load[
                    width = Self.element_data_type.size, alignment=alignment
                ]()

            @parameter
            for i in range(size):
                element_data[i] = ptr[_get_offset[i](runtime_layout)]
            return Element(element_data, runtime_layout)

        @parameter
        if layout.stride[0] == 1:
            alias size = Int(layout.shape[0])
            alias elements = Int(layout.shape[1])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()

            @parameter
            for i in range(elements):
                var vec_i = ptr.load[width=size, alignment=alignment](
                    _get_offset[0, i](runtime_layout)
                )
                element_data = element_data.insert[offset = i * size](vec_i)
            return Element(element_data, runtime_layout)

        elif layout.stride[1] == 1:
            alias size = Int(layout.shape[1])
            alias elements = Int(layout.shape[0])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()

            @parameter
            for i in range(elements):
                var vec_i = ptr.load[width=size, alignment=alignment](
                    _get_offset[i, 0](runtime_layout)
                )
                element_data = element_data.insert[offset = i * size](vec_i)
            return Element(element_data, runtime_layout)

        alias dim_0 = Int(layout.shape[0])
        alias dim_1 = Int(layout.shape[1])

        @parameter
        for i in range(dim_0):

            @parameter
            for j in range(dim_1):
                element_data[i + j * dim_0] = ptr[
                    _get_offset[i, j](runtime_layout)
                ]
        return Element(element_data, runtime_layout)

    @always_inline("nodebug")
    @staticmethod
    fn masked_load(
        ptr: UnsafePointer[Scalar[dtype], **_],
        runtime_layout: RuntimeLayout[
            layout, bitwidth=bitwidth
        ] = RuntimeLayout[layout, bitwidth=bitwidth](),
    ) -> Self:
        # TODO: Use partial_simd_load after closing KERN-729.
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()
        var element_data = Self.element_data_type()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                alias alignment = alignof[Self.element_data_type]()
                if runtime_layout.dim(0) < size:

                    @parameter
                    for i in range(size):
                        if i >= runtime_layout.dim(0):
                            break
                        element_data[i] = ptr[_get_offset[i](runtime_layout)]
                    return Element(element_data, runtime_layout)

                return ptr.load[
                    width = Self.element_data_type.size, alignment=alignment
                ](0)

            @parameter
            for i in range(size):
                if i >= runtime_layout.dim(0):
                    break
                element_data[i] = ptr[_get_offset[i](runtime_layout)]
            return Element(element_data, runtime_layout)

        # rank-2 element.
        @parameter
        if layout.stride[0] == 1:
            alias size = Int(layout.shape[0])
            alias elements = Int(layout.shape[1])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]
            var element_data = Self.element_data_type()
            if runtime_layout.dim(0) < size:
                alias dim_0 = Int(layout.shape[0])
                alias dim_1 = Int(layout.shape[1])

                @parameter
                for i in range(dim_0):
                    if i >= runtime_layout.dim(0):
                        break

                    @parameter
                    for j in range(dim_1):
                        if j >= runtime_layout.dim(1):
                            break
                        element_data[i + j * dim_0] = ptr[
                            _get_offset[i, j](runtime_layout)
                        ]
                return Element(element_data, runtime_layout)

            @parameter
            for i in range(elements):
                if i >= runtime_layout.dim(0):
                    break
                var vec_i = ptr.load[width=size](
                    _get_offset[0, i](runtime_layout)
                )
                element_data = element_data.insert[offset = i * size](vec_i)
            return Element(element_data, runtime_layout)

        elif layout.stride[1] == 1:
            alias size = Int(layout.shape[1])
            alias elements = Int(layout.shape[0])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]
            var element_data = Self.element_data_type()
            if runtime_layout.dim(1) < size:
                alias dim_0 = Int(layout.shape[0])
                alias dim_1 = Int(layout.shape[1])

                @parameter
                for i in range(dim_0):
                    if i >= runtime_layout.dim(0):
                        break

                    @parameter
                    for j in range(dim_1):
                        if j >= runtime_layout.dim(1):
                            break
                        element_data[i + j * dim_0] = ptr[
                            _get_offset[i, j](runtime_layout)
                        ]
                return Element(element_data, runtime_layout)

            @parameter
            for i in range(elements):
                if i >= runtime_layout.dim(0):
                    break
                var vec_i = ptr.load[width=size](
                    _get_offset[i, 0](runtime_layout)
                )
                element_data = element_data.insert[offset = i * size](vec_i)
            return Element(element_data, runtime_layout)

        alias dim_0 = Int(layout.shape[0])
        alias dim_1 = Int(layout.shape[1])

        @parameter
        for i in range(dim_0):
            if i >= runtime_layout.dim(0):
                break

            @parameter
            for j in range(dim_1):
                if j >= runtime_layout.dim(1):
                    break
                element_data[i + j * dim_0] = ptr[
                    _get_offset[i, j](runtime_layout)
                ]
        return Element(element_data, runtime_layout)

    @always_inline("nodebug")
    fn store(self, ptr: UnsafePointer[Scalar[dtype], mut=True, **_]):
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                alias alignment = alignof[Self.element_data_type]()
                ptr.store[alignment=alignment](self.element_data)
                return

            @parameter
            for i in range(size):
                ptr[_get_offset[i](self.runtime_layout)] = self.element_data[i]
            return

        @parameter
        if layout.stride[0] == 1:
            alias size = Int(layout.shape[0])
            alias elements = Int(layout.shape[1])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()

            @parameter
            for i in range(elements):
                ptr.store[alignment=alignment](
                    _get_offset[0, i](self.runtime_layout),
                    self.element_data.slice[size, offset = i * size](),
                )
            return

        elif layout.stride[1] == 1:
            alias size = Int(layout.shape[1])
            alias elements = Int(layout.shape[0])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()

            @parameter
            for i in range(elements):
                ptr.store[alignment=alignment](
                    _get_offset[i, 0](self.runtime_layout),
                    self.element_data.slice[size, offset = i * size](),
                )
            return

        alias dim_0 = Int(layout.shape[0])
        alias dim_1 = Int(layout.shape[1])

        @parameter
        for i in range(dim_0):

            @parameter
            for j in range(dim_1):
                (ptr + _get_offset[i, j](self.runtime_layout)).store(
                    self.element_data[i + j * dim_0]
                )

    @always_inline("nodebug")
    fn masked_store(self, ptr: UnsafePointer[Scalar[dtype], mut=True, **_]):
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                if self.runtime_layout.dim(0) < size:

                    @parameter
                    for i in range(size):
                        if i >= self.runtime_layout.dim(0):
                            break
                        ptr[
                            _get_offset[i](self.runtime_layout)
                        ] = self.element_data[i]
                    return

                alias alignment = alignof[Self.element_data_type]()
                ptr.store(self.element_data)
                return

            @parameter
            for i in range(size):
                if i >= self.runtime_layout.dim(0):
                    break
                ptr[_get_offset[i](self.runtime_layout)] = self.element_data[i]
            return

        @parameter
        if layout.stride[0] == 1:
            alias size = Int(layout.shape[0])
            alias elements = Int(layout.shape[1])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()
            if self.runtime_layout.dim(1) < size:
                alias dim_0 = Int(layout.shape[0])
                alias dim_1 = Int(layout.shape[1])

                @parameter
                for i in range(dim_0):
                    if i >= self.runtime_layout.dim(0):
                        break

                    @parameter
                    for j in range(dim_1):
                        if j >= self.runtime_layout.dim(1):
                            break
                        ptr[] = _get_offset[i, j](self.runtime_layout)
                return

            @parameter
            for i in range(elements):
                if i >= self.runtime_layout.dim(0):
                    break
                (ptr + _get_offset[i, 0](self.runtime_layout)).store[
                    alignment=alignment
                ](
                    self.element_data.slice[size, offset = i * size](),
                )
            return

        elif layout.stride[1] == 1:
            alias size = Int(layout.shape[1])
            alias elements = Int(layout.shape[0])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()
            if self.runtime_layout.dim(1) < size:
                alias dim_0 = Int(layout.shape[0])
                alias dim_1 = Int(layout.shape[1])

                @parameter
                for i in range(dim_0):
                    if i >= self.runtime_layout.dim(0):
                        break

                    @parameter
                    for j in range(dim_1):
                        if j >= self.runtime_layout.dim(1):
                            break
                        (ptr + _get_offset[i, j](self.runtime_layout)).store(
                            self.element_data[i + j * dim_0],
                        )
                return

            @parameter
            for i in range(elements):
                if i >= self.runtime_layout.dim(0):
                    break
                (ptr + _get_offset[i, 0](self.runtime_layout)).store[
                    alignment=alignment
                ](
                    self.element_data.slice[size, offset = i * size](),
                )
            return

        alias dim_0 = Int(layout.shape[0])
        alias dim_1 = Int(layout.shape[1])

        @parameter
        for i in range(dim_0):
            if i >= self.runtime_layout.dim(0):
                break

            @parameter
            for j in range(dim_1):
                if j >= self.runtime_layout.dim(1):
                    break
                (ptr + _get_offset[i, j](self.runtime_layout)).store(
                    self.element_data[i + j * dim_0]
                )

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.element_data)


# Represents a element in memory, organized according
# to a specific layout structure.
#
struct MemoryElement[
    dtype: DType,
    layout: Layout,
    address_space: AddressSpace,
    alignment: Int,
    /,
    *,
    bitwidth: Int = bitwidthof[Int](),
]:
    var ptr: UnsafePointer[
        Scalar[dtype], address_space=address_space, alignment=alignment
    ]

    var runtime_layout: RuntimeLayout[layout, bitwidth=bitwidth]

    fn __init__(
        mut self,
        ptr: UnsafePointer[
            Scalar[dtype], address_space=address_space, alignment=alignment
        ],
        runtime_layout: RuntimeLayout[layout, bitwidth=bitwidth],
    ):
        self.ptr = ptr
        self.runtime_layout = runtime_layout

    @always_inline("nodebug")
    fn load(self) -> Element[dtype, layout, bitwidth=bitwidth]:
        return Element.load(self.ptr, self.runtime_layout)

    @always_inline("nodebug")
    fn store(self, src: Element[dtype, layout, bitwidth=bitwidth]):
        return src.store(self.ptr)

    @always_inline("nodebug")
    fn transfer(self, src: MemoryElement):
        # Load source element and convert to destination dtype if needed
        var src_element = src.load()
        var converted_element = Element[
            dtype, src.layout, bitwidth = src.bitwidth
        ](src_element.element_data.cast[dtype](), src_element.runtime_layout)
        self.store(
            rebind[Element[dtype, layout, bitwidth=bitwidth]](converted_element)
        )
