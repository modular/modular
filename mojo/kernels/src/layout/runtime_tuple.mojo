# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray
from os import abort
from sys import bitwidthof
from builtin.dtype import _int_type_of_width, _uint_type_of_width

from layout.int_tuple import UNKNOWN_VALUE, IntTuple, flatten
from layout.int_tuple import idx2crd as idx2crd_int_tuple
from layout.int_tuple import prefix_product as prefix_product_int_tuple
from layout.int_tuple import shape_div as shape_div_int_tuple

from utils import IndexList


fn concat(owned lhs: IntTuple, rhs: IntTuple) -> IntTuple:
    for i in range(len(rhs)):
        lhs.append(rhs[i])
    return lhs


fn _get_returned_type[bitwidth: Int, unsigned: Bool]() -> DType:
    @parameter
    if unsigned:
        return _uint_type_of_width[bitwidth]()

    return _int_type_of_width[bitwidth]()


@register_passable("trivial")
struct RuntimeTuple[
    S: IntTuple = UNKNOWN_VALUE,
    /,
    *,
    element_bitwidth: Int = bitwidthof[Int](),
    unsigned: Bool = False,
](Stringable, Sized):
    alias int_type: DType = _get_returned_type[element_bitwidth, unsigned]()
    alias scalar_length = len(flatten(S))
    var value: IndexList[
        Self.scalar_length, element_bitwidth=element_bitwidth, unsigned=unsigned
    ]

    @always_inline
    fn __init__(inout self):
        self.value = __type_of(self.value)()

        alias f = flatten(S)

        @parameter
        for i in range(Self.scalar_length):
            alias v = f[i].value()

            @parameter
            if v != UNKNOWN_VALUE:
                self.value[i] = v
            else:
                self.value[i] = UNKNOWN_VALUE

    @always_inline
    fn __init__(inout self, *values: Int):
        self.value = values

    @always_inline
    fn __init__[l: Int](inout self, values: IndexList[l, **_]):
        constrained[Self.scalar_length == l, "Must use same tuple length"]()
        self.value = rebind[__type_of(self.value)](
            values.cast[
                element_bitwidth = __type_of(self.value).element_bitwidth,
                unsigned = __type_of(self.value).unsigned,
            ]()
        )

    @staticmethod
    @always_inline
    fn offset_until[i: Int]() -> Int:
        var result = 0

        @parameter
        for j in range(i):
            result += len(flatten(S[j]))
        return result

    @always_inline
    fn get_int(self) -> Scalar[Self.int_type]:
        alias comptime_value: Scalar[Self.int_type] = S.value()

        @parameter
        if comptime_value != UNKNOWN_VALUE:
            return comptime_value
        else:
            return self.value[0]

    @always_inline
    fn __getitem__[
        i: Int
    ](self) -> RuntimeTuple[S[i], element_bitwidth=element_bitwidth]:
        var res = RuntimeTuple[S[i], element_bitwidth=element_bitwidth]()
        alias offset = Self.offset_until[i]()

        @parameter
        for i in range(res.scalar_length):
            res.value[i] = self.value[i + offset]
        return res

    @always_inline
    fn __setitem__[i: Int](inout self, val: Scalar[Self.int_type]):
        alias offset = Self.offset_until[i]()
        self.value[offset] = int(val)

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @always_inline
    fn concat[
        R: IntTuple
    ](
        self, rhs: RuntimeTuple[R, element_bitwidth=element_bitwidth]
    ) -> RuntimeTuple[concat(S, R), element_bitwidth=element_bitwidth]:
        var out = RuntimeTuple[
            concat(S, R), element_bitwidth=element_bitwidth
        ]()

        alias S_flat = flatten(S)

        @parameter
        for i in range(Self.scalar_length):

            @parameter
            if S_flat[i] == UNKNOWN_VALUE:
                out.value[i] = self.value[i]

        alias R_flat = flatten(R)

        @parameter
        for i in range(rhs.scalar_length):

            @parameter
            if R_flat[i] == UNKNOWN_VALUE:
                out.value[Self.scalar_length + i] = rhs.value[i]

        return out

    @always_inline
    fn flatten(
        self,
    ) -> RuntimeTuple[flatten(S), element_bitwidth=element_bitwidth] as result:
        return __type_of(result)(self.value)

    fn write_to[W: Writer](self, inout writer: W):
        @parameter
        if S.is_value():
            writer.write(self.get_int())
        else:
            writer.write("(")

            alias size = len(S)

            @parameter
            for i in range(size):
                self[i].write_to(writer)

                @parameter
                if i != size - 1:
                    writer.write(", ")
            writer.write(")")

    @always_inline
    fn __len__(self) -> Int:
        return len(S)


fn is_tuple[t: IntTuple](tuple: RuntimeTuple[t, **_]) -> Bool:
    return t.is_tuple()


fn is_int[t: IntTuple](tuple: RuntimeTuple[t, **_]) -> Bool:
    return t.is_value()


@always_inline
fn to_int[
    t: IntTuple,
    element_bitwidth: Int,
    unsigned: Bool,
    /,
    *,
    int_type: DType = _get_returned_type[element_bitwidth, unsigned](),
](
    tuple: RuntimeTuple[t, element_bitwidth=element_bitwidth, unsigned=unsigned]
) -> Scalar[int_type]:
    constrained[t.is_value(), "tuple must be a single int value"]()
    return tuple.value[0]


@always_inline
fn prefix_product[
    t: IntTuple
](tuple: RuntimeTuple[t]) -> RuntimeTuple[prefix_product_int_tuple(t)]:
    var res = RuntimeTuple[prefix_product_int_tuple(t)]()
    var prefix_res = 1
    for i in range(tuple.scalar_length):
        res.value[i] = prefix_res
        prefix_res *= tuple.value[i]
    return res


@always_inline
fn product[t: IntTuple](tuple: RuntimeTuple[t, **_]) -> Int:
    var res: Int = 1

    @parameter
    for i in range(tuple.scalar_length):
        res *= tuple.value[i]
    return res


@always_inline
fn idx2crd[
    idx_t: IntTuple, shape_t: IntTuple, stride_t: IntTuple
](
    idx: RuntimeTuple[idx_t, **_],
    shape: RuntimeTuple[shape_t, **_],
    stride: RuntimeTuple[stride_t, **_],
) -> RuntimeTuple[idx2crd_int_tuple(idx_t, shape_t, stride_t)]:
    var res = RuntimeTuple[idx2crd_int_tuple(idx_t, shape_t, stride_t)]()
    constrained[idx_t.is_value(), "Only scalar index is supported"]()
    for i in range(res.scalar_length):
        res.value[i] = (int(to_int(idx)) // stride.value[i]) % shape.value[i]
    return res


@always_inline
fn idx2crd[
    idx_t: IntTuple, shape_t: IntTuple
](idx: RuntimeTuple[idx_t], shape: RuntimeTuple[shape_t]) -> RuntimeTuple[
    idx2crd_int_tuple(idx_t, shape_t, prefix_product_int_tuple(shape_t))
]:
    return idx2crd(idx, shape, prefix_product(shape))


fn crd2idx[
    crd_t: IntTuple, shape_t: IntTuple, stride_t: IntTuple
](
    crd: RuntimeTuple[crd_t],
    shape: RuntimeTuple[shape_t, **_],
    stride: RuntimeTuple[stride_t, **_],
) -> Int:
    @parameter
    if crd_t.is_tuple():
        constrained[
            shape_t.is_tuple()
            and (len(crd_t) == len(shape_t) == len(stride_t)),
            "Inputs should have same rank",
        ]()
        var r: Int = 0
        alias size = min(min(len(crd_t), len(shape_t)), len(stride_t))

        @parameter
        for i in range(size):
            r += crd2idx(crd[i], shape[i], stride[i])
        return r
    else:
        var int_crd: Int = 0 if len(crd) == 0 else int(to_int(crd))

        @parameter
        if shape_t.is_tuple():  # "int" tuple tuple
            constrained[
                len(shape_t) == len(stride_t),
                "shape and stride should have same rank",
            ]()
            var result: Int = 0

            alias last_elem_idx = len(shape_t) - 1

            @parameter
            for i in range(last_elem_idx):
                divisor, quotient = divmod(int_crd, product(shape[i]))
                result += crd2idx(quotient, shape[i], stride[i])
                int_crd = divisor
            # FIXME(KERN-640): Replace with [-1], currently not giving correct result.
            return result + crd2idx(
                int_crd, shape[last_elem_idx], stride[last_elem_idx]
            )
        else:  # "int" "int" "int"
            return int_crd * int(to_int(stride))


# TODO: This isn't necessarily needed. We need to revisit and simplify
# the implementation. We are keeping it here to be consistent with IntTuple
# shape_div.
@always_inline
fn signum(a: Int) -> Int:
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


# TODO: the returned runtime tuple needs to conform to a type base on a element
# and b element type, for example, if a is int64 and b is int32, return a int64 type
fn shape_div[
    a_t: IntTuple, b_t: IntTuple
](a: RuntimeTuple[a_t, **_], b: RuntimeTuple[b_t, **_]) -> RuntimeTuple[
    shape_div_int_tuple(a_t, b_t)
]:
    @parameter
    if a_t.is_tuple():

        @parameter
        if b_t.is_tuple():
            constrained[
                len(a_t) == len(b_t), "shape and stride length musth match"
            ]()
            var res = RuntimeTuple[shape_div_int_tuple(a_t, b_t)]()

            @parameter
            for i in range(len(a_t)):
                var res_i = shape_div(a[i], b[i])

                @parameter
                for j in range(res_i.scalar_length):
                    res.value[i + j] = res_i.value[j]
            return res
        else:
            var res = RuntimeTuple[shape_div_int_tuple(a_t, b_t)]()
            var vb = int(to_int(b))

            @parameter
            for i in range(len(a_t)):
                var res_i = shape_div(a[i], vb)

                @parameter
                for j in range(res_i.scalar_length):
                    res.value[i + j] = res_i.value[j]

                vb = int(to_int(shape_div(vb, product(a[i]))))
            return res
    else:

        @parameter
        if b_t.is_tuple():
            return shape_div(a, b)
        else:
            var va = int(to_int(a))
            var vb = int(to_int(b))

            if not (va % vb == 0 or vb % va == 0):
                abort("Incompatible shape values: " + str(va) + " " + str(vb))

            return va // vb if va % vb == 0 else signum(va * vb)
