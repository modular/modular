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
    return lhs.owned_copy()


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
    fn __init__(out self):
        """Initialize a RuntimeTuple with default values.

        For dimensions with known compile-time values in S, uses those values.
        For unknown dimensions, initializes them to UNKNOWN_VALUE.
        """
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
    @implicit
    fn __init__(out self, *values: Int):
        """Initialize a RuntimeTuple with the provided values.

        Args:
            values: Variadic number of integer values to initialize the tuple with.
        """
        self.value = values

    @always_inline
    @implicit
    fn __init__[l: Int](mut self, values: IndexList[l, **_]):
        """Initialize a RuntimeTuple from an IndexList.

        Args:
            values: IndexList to initialize from. Must have same length as the RuntimeTuple.
        """
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
    ](self) -> RuntimeTuple[
        S[i], element_bitwidth=element_bitwidth, unsigned=unsigned
    ]:
        var res = RuntimeTuple[
            S[i], element_bitwidth=element_bitwidth, unsigned=unsigned
        ]()
        alias offset = Self.offset_until[i]()

        @parameter
        for i in range(res.scalar_length):
            res.value[i] = self.value[i + offset]
        return res

    @always_inline
    fn __setitem__[i: Int](mut self, val: Scalar[Self.int_type]):
        alias offset = Self.offset_until[i]()
        self.value[offset] = Int(val)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @always_inline
    fn concat[
        R: IntTuple
    ](
        self,
        rhs: RuntimeTuple[
            R, element_bitwidth=element_bitwidth, unsigned=unsigned
        ],
        out result: RuntimeTuple[
            concat(S, R), element_bitwidth=element_bitwidth, unsigned=unsigned
        ],
    ):
        var out = __type_of(result)()

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
        out result: RuntimeTuple[
            flatten(S), element_bitwidth=element_bitwidth, unsigned=unsigned
        ],
    ):
        return __type_of(result)(self.value)

    fn write_to[W: Writer](self, mut writer: W):
        @parameter
        if S.is_value():
            writer.write(self.value[0])
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
        alias l = len(S)
        return l

    @always_inline
    fn cast[
        type: DType
    ](
        self,
        out result: RuntimeTuple[
            S, element_bitwidth = bitwidthof[type](), unsigned=unsigned
        ],
    ):
        return __type_of(result)(self.value.cast[type]())

    @always_inline
    fn __int__(self) -> Int:
        constrained[S.is_value(), "tuple must be a single int value"]()
        return self.value[0]


fn is_tuple[t: IntTuple](tuple: RuntimeTuple[t, **_]) -> Bool:
    return t.is_tuple()


fn is_int[t: IntTuple](tuple: RuntimeTuple[t, **_]) -> Bool:
    return t.is_value()


@always_inline
fn prefix_product[
    t: IntTuple
](tuple: RuntimeTuple[t, **_]) -> RuntimeTuple[prefix_product_int_tuple(t)]:
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
    idx_t: IntTuple,
    shape_t: IntTuple,
    stride_t: IntTuple,
](
    idx: RuntimeTuple[idx_t, **_],
    shape: RuntimeTuple[shape_t, **_],
    stride: RuntimeTuple[stride_t, **_],
    out result: RuntimeTuple[
        idx2crd_int_tuple(idx_t, shape_t, stride_t),
        element_bitwidth = shape.element_bitwidth,
        unsigned = shape.unsigned,
    ],
):
    var res = __type_of(result)()
    constrained[idx_t.is_value(), "Only scalar index is supported"]()
    for i in range(res.scalar_length):
        res.value[i] = (Int(idx) // stride.value[i]) % shape.value[i]
    return res


# take shape as return type
@always_inline
fn idx2crd[
    idx_t: IntTuple,
    shape_t: IntTuple,
](
    idx: RuntimeTuple[idx_t, **_],
    shape: RuntimeTuple[shape_t, **_],
) -> RuntimeTuple[
    idx2crd_int_tuple(idx_t, shape_t, prefix_product_int_tuple(shape_t)),
    element_bitwidth = shape.element_bitwidth,
    unsigned = shape.unsigned,
]:
    return idx2crd(idx, shape, prefix_product(shape))


fn crd2idx[
    crd_t: IntTuple, shape_t: IntTuple, stride_t: IntTuple
](
    crd: RuntimeTuple[crd_t, **_],
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
        var int_crd: Int = 0 if len(crd) == 0 else Int(crd)

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
                result += crd2idx[crd_t](quotient, shape[i], stride[i])
                int_crd = divisor
            # FIXME(KERN-640): Replace with [-1], currently not giving correct result.
            return result + crd2idx[crd_t](
                int_crd, shape[last_elem_idx], stride[last_elem_idx]
            )
        else:  # "int" "int" "int"
            return int_crd * Int(stride)


# TODO: This isn't necessarily needed. We need to revisit and simplify
# the implementation. We are keeping it here to be consistent with IntTuple
# shape_div.
@always_inline
fn signum(a: Int) -> Int:
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


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
            # FIXME: this used to be simpler
            # var vb = Int(to_int(b))
            var vb = RuntimeTuple[IntTuple(UNKNOWN_VALUE)](Int(b))

            @parameter
            for i in range(len(a_t)):
                var res_i = shape_div(a[i], vb)

                @parameter
                for j in range(res_i.scalar_length):
                    res.value[i + j] = res_i.value[j]

                # FIXME: this used to be simpler
                # vb = Int(to_int(shape_div(vb, product(a[i]))))
                vb = Int(
                    shape_div(
                        vb,
                        RuntimeTuple[IntTuple(UNKNOWN_VALUE)](product(a[i])),
                    )
                )
            return res
    else:

        @parameter
        if b_t.is_tuple():
            return shape_div(a, b)
        else:
            var va = Int(a)
            var vb = Int(b)

            if not (va % vb == 0 or vb % va == 0):
                abort("Incompatible shape values: ", va, " ", vb)

            return va // vb if va % vb == 0 else signum(va * vb)
