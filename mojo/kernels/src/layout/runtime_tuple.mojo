# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from layout.int_tuple import IntTuple, flatten
from utils.static_tuple import InlineArray


fn concat(owned lhs: IntTuple, rhs: IntTuple) -> IntTuple:
    for i in range(len(rhs)):
        lhs.append(rhs[i])
    return lhs


struct RuntimeTuple[S: IntTuple]:
    alias sentinel = -1
    alias scalar_length = len(flatten(S))
    var value: StaticIntTuple[Self.scalar_length]

    @always_inline
    fn __init__(inout self, *, uninit: ()):
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.value)
        )

    @always_inline
    fn __init__(inout self):
        self.__init__(uninit=())

        alias f = flatten(S)

        @parameter
        for i in range(Self.scalar_length):
            alias v = f[i].value()

            @parameter
            if v != Self.sentinel:
                self.value[i] = v

    @always_inline
    fn __init__(inout self, *values: Int):
        self.value = values

    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.value = other.value

    @staticmethod
    @always_inline
    fn offset_until[i: Int]() -> Int:
        var result = 0

        @parameter
        for j in range(i):
            result += len(flatten(S[j]))
        return result

    @always_inline
    fn get_int(self) -> Int:
        alias comptime_value: Int = S.value()

        @parameter
        if comptime_value != Self.sentinel:
            return comptime_value
        else:
            return self.value[0]

    @always_inline
    fn __getitem__[
        i: Int
    ](ref [_]self) -> ref [__lifetime_of(self)] RuntimeTuple[S[i]]:
        alias offset = Self.offset_until[i]()
        var int_ptr = UnsafePointer.address_of(self).bitcast[Int]() + offset
        return int_ptr.bitcast[RuntimeTuple[S[i]]]()[]

    @always_inline
    @__named_result(out)
    fn concat[
        R: IntTuple
    ](self, rhs: RuntimeTuple[R]) -> RuntimeTuple[concat(S, R)]:
        out.__init__(uninit=())

        alias S_flat = flatten(S)

        @parameter
        for i in range(Self.scalar_length):

            @parameter
            if S_flat[i] == Self.sentinel:
                out.value[i] = self.value[i]

        alias R_flat = flatten(R)

        @parameter
        for i in range(rhs.scalar_length):

            @parameter
            if R_flat[i] == Self.sentinel:
                out.value[Self.scalar_length + i] = rhs.value[i]

    @always_inline
    fn flatten(self) -> ref [__lifetime_of(self)] RuntimeTuple[flatten(S)]:
        return UnsafePointer.address_of(self).bitcast[
            RuntimeTuple[flatten(S)]
        ]()[]

    fn format_to(self, inout f: Formatter):
        @parameter
        if S.is_value():
            self.get_int().format_to(f)
        else:
            f.write_str["("]()

            alias size = len(S)

            @parameter
            for i in range(size):
                self[i].format_to(f)

                @parameter
                if i != size - 1:
                    f.write_str[", "]()
            f.write_str[")"]()
