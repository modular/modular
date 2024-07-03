# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from layout.int_tuple import IntTuple, flatten

from layout.int_tuple import prefix_product as prefix_product_int_tuple

from utils.static_tuple import InlineArray


fn concat(owned lhs: IntTuple, rhs: IntTuple) -> IntTuple:
    for i in range(len(rhs)):
        lhs.append(rhs[i])
    return lhs


@register_passable("trivial")
struct RuntimeTuple[S: IntTuple](Stringable, Sized):
    alias sentinel = -1
    alias scalar_length = len(flatten(S))
    var value: StaticIntTuple[Self.scalar_length]

    @always_inline
    fn __init__(inout self):
        self.value = StaticIntTuple[Self.scalar_length]()

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
    fn __init__[l: Int](inout self, values: StaticIntTuple[l]):
        constrained[Self.scalar_length == l, "Must use same tuple length"]()
        self.value = rebind[StaticIntTuple[Self.scalar_length]](values)

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
    fn __getitem__[i: Int](self) -> RuntimeTuple[S[i]]:
        var res = RuntimeTuple[S[i]]()
        alias offset = Self.offset_until[i]()

        @parameter
        for i in range(Self.scalar_length - offset):
            res.value[i] = self.value[i + offset]
        return res

    @always_inline
    fn __setitem__[i: Int](inout self, val: Int):
        alias offset = Self.offset_until[i]()
        self.value[offset] = val

    @always_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @always_inline
    fn concat[
        R: IntTuple
    ](self, rhs: RuntimeTuple[R]) -> RuntimeTuple[concat(S, R)]:
        var out = RuntimeTuple[concat(S, R)]()

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

        return out

    @always_inline
    fn flatten(self) -> RuntimeTuple[flatten(S)]:
        return RuntimeTuple[flatten(S)](self.value)

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

    @always_inline
    fn __len__(self) -> Int:
        return self.scalar_length


fn is_tuple[t: IntTuple](tuple: RuntimeTuple[t]) -> Bool:
    return t.is_tuple()


fn is_int[t: IntTuple](tuple: RuntimeTuple[t]) -> Bool:
    return t.is_value()


@always_inline
fn prefix_product[
    t: IntTuple
](tuple: RuntimeTuple[t]) -> RuntimeTuple[prefix_product_int_tuple(t)]:
    var res = RuntimeTuple[prefix_product_int_tuple(t)]()
    var prefix_res = 1
    for i in range(len(tuple)):
        res.value[i] = prefix_res
        prefix_res *= tuple.value[i]
    return res
