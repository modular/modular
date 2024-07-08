# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from . import Layout
from .layout import coalesce, to_int


# Element is a wrapper around SIMD type, it extends the SIMD type to define
# a vectorized load / store that is driven by the layout of the element.
#
struct Element[dtype: DType, layout: Layout](Stringable, Formattable):
    alias element_data_type = SIMD[dtype, size = layout.size()]

    var element_data: Self.element_data_type

    fn __init__(inout self, element_data: Self.element_data_type):
        self.element_data = element_data

    @staticmethod
    fn load[
        address_space: AddressSpace
    ](ptr: DTypePointer[dtype, address_space]) -> Self:
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                alias alignment = alignof[Self.element_data_type]()
                return Self.element_data_type.load[alignment=alignment](ptr, 0)
            else:
                var element_data = Self.element_data_type()

                @parameter
                for i in range(size):
                    alias offset = layout(i)

                    element_data[i] = ptr[offset]
                return element_data
        else:

            @parameter
            if layout.stride[0] == 1:
                alias size = to_int(layout.shape[0])
                alias elements = to_int(layout.shape[1])

                alias vec_type = SIMD[dtype, size]
                alias alignment = alignof[vec_type]

                var element_data = Self.element_data_type()

                @parameter
                for i in range(elements):
                    alias offset = layout(IntTuple(0, i))
                    var vec_i = vec_type.load(ptr, offset)
                    element_data = element_data.insert[offset = i * size](vec_i)

                return element_data
            elif layout.stride[1] == 1:
                alias size = to_int(layout.shape[1])
                alias elements = to_int(layout.shape[0])

                alias vec_type = SIMD[dtype, size]
                alias alignment = alignof[vec_type]

                var element_data = Self.element_data_type()

                @parameter
                for i in range(elements):
                    alias offset = layout(IntTuple(i, 0))
                    var vec_i = vec_type.load(ptr, offset)
                    element_data = element_data.insert[offset = i * size](vec_i)

                return element_data
            else:
                var element_data = Self.element_data_type()

                alias dim_0 = to_int(layout.shape[0])
                alias dim_1 = to_int(layout.shape[1])

                @parameter
                for i in range(dim_0):

                    @parameter
                    for j in range(dim_1):
                        alias offset = layout(IntTuple(i, j))

                        element_data[i + j * dim_1] = ptr[offset]

                return element_data

    fn store[
        address_space: AddressSpace
    ](self, ptr: DTypePointer[dtype, address_space]):
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                alias alignment = alignof[Self.element_data_type]()
                Self.element_data_type.store[alignment=alignment](
                    ptr, self.element_data
                )
            else:

                @parameter
                for i in range(size):
                    alias offset = layout(i)

                    ptr[offset] = self.element_data[i]
        else:

            @parameter
            if layout.stride[0] == 1:
                alias size = to_int(layout.shape[0])
                alias elements = to_int(layout.shape[1])

                alias vec_type = SIMD[dtype, size]
                alias alignment = alignof[vec_type]()

                @parameter
                for i in range(elements):
                    alias offset = layout(IntTuple(0, i))
                    vec_type.store[alignment=alignment](
                        ptr + offset,
                        self.element_data.slice[size, offset = i * size](),
                    )

            elif layout.stride[1] == 1:
                alias size = to_int(layout.shape[1])
                alias elements = to_int(layout.shape[0])

                alias vec_type = SIMD[dtype, size]
                alias alignment = alignof[vec_type]()

                @parameter
                for i in range(elements):
                    alias offset = layout(IntTuple(i, 0))
                    vec_type.store[alignment=alignment](
                        ptr + offset,
                        self.element_data.slice[size, offset = i * size](),
                    )
            else:
                alias dim_0 = to_int(layout.shape[0])
                alias dim_1 = to_int(layout.shape[1])

                @parameter
                for i in range(dim_0):

                    @parameter
                    for j in range(dim_1):
                        alias offset = layout(IntTuple(i, j))

                        self.element_data[i + j * dim_1].store(ptr, offset)

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        # TODO: Avoid intermediate string allocation.
        writer.write(self.element_data.__str__())
