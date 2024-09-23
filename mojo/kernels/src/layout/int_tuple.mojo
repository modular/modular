# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math
from os import abort

from layout.dynamic_tuple import *

from utils import Formatter, Variant

# IntTuple definition
# TODO: Consider a different approch https://linear.app/modularml/issue/KERN-638
alias UNKNOWN_VALUE = -1


# FIXME: This is a horrible hack around Mojo's lack or proper trait inheritance
struct IntDelegate(ElementDelegate):
    @always_inline
    @staticmethod
    fn is_equal[T: CollectionElement](a: Variant[T], b: Variant[T]) -> Bool:
        if a.isa[Int]() and b.isa[Int]():
            return a[Int] == b[Int]
        else:
            abort("Unexpected data type.")
            return False

    @always_inline
    @staticmethod
    fn format_element_to[
        T: CollectionElement
    ](inout writer: Formatter, a: Variant[T]):
        if a.isa[Int]():
            writer.write(a[Int])
        else:
            abort("Unexpected data type.")


alias IntTuple = DynamicTuple[Int, IntDelegate]


@always_inline
fn signum(a: Int) -> Int:
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


@always_inline
fn to_int(owned v: IntTuple) -> Int:
    return v.value()


@always_inline
fn is_int(t: IntTuple) -> Bool:
    return t.is_value()


@always_inline
fn is_tuple(t: IntTuple) -> Bool:
    return t.is_tuple()


# Python-style reduce functions
# FIXME: Can we unify the two versions?
fn reduce[
    T: AnyTrivialRegType, func: fn (owned a: T, b: IntTuple) capturing [_] -> T
](t: IntTuple, initializer: T) -> T:
    var result: T = initializer
    for e in t:
        result = func(result, e)
    return result


fn reduce[
    T: CollectionElement, func: fn (owned a: T, b: IntTuple) capturing [_] -> T
](t: IntTuple, initializer: T) -> T:
    var result: T = initializer
    for e in t:
        result = func(result, e)
    return result


# IntTuple operations


@always_inline
fn flatten(t: IntTuple) -> IntTuple:
    @always_inline
    @parameter
    fn reducer(owned a: IntTuple, b: IntTuple) -> IntTuple:
        if b.is_value():
            a.append(b)
        else:
            for e in flatten(b):
                a.append(e)
        return a

    return reduce[IntTuple, reducer](t, IntTuple())


# Create a IntTuple with same structure but filled by UNKNOWN_VALUE.
@always_inline
fn to_unknown(t: IntTuple) -> IntTuple:
    if t.is_value():
        return UNKNOWN_VALUE

    @always_inline
    @parameter
    fn reducer(owned a: IntTuple, b: IntTuple) -> IntTuple:
        if b.is_value():
            a.append(UNKNOWN_VALUE)
        else:
            a.append(to_unknown(b))
        return a

    return reduce[IntTuple, reducer](t, IntTuple())


fn _insertion_sort[
    cmp: fn (IntTuple, IntTuple) -> Bool
](inout tuple: IntTuple, start: Int, end: Int):
    for i in range(start + 1, end):
        var value = tuple[i]
        var j = i

        while j > start and not cmp(tuple[j - 1], value):
            tuple[j] = tuple[j - 1]
            j -= 1

        tuple[j] = value


fn lt(a: IntTuple, b: IntTuple) -> Bool:
    for z in zip(a, b):
        if to_int(z[0]) == to_int(z[1]):
            continue
        elif to_int(z[0]) < to_int(z[1]):
            return True
        else:
            return False
    return False


fn sorted[cmp: fn (IntTuple, IntTuple) -> Bool = lt](t: IntTuple) -> IntTuple:
    var _t = t
    _insertion_sort[cmp](_t, 0, len(t))
    return _t


fn sum(t: IntTuple) -> Int:
    @always_inline
    @parameter
    fn reducer(owned a: Int, b: IntTuple) -> Int:
        return -1 if a == UNKNOWN_VALUE else a + (
            to_int(b) if is_int(b) else sum(b)
        )

    return reduce[Int, reducer](t, 0)


@always_inline
fn product(t: IntTuple) -> Int:
    @always_inline
    @parameter
    fn reducer(owned a: Int, b: IntTuple) -> Int:
        return -1 if a == UNKNOWN_VALUE else a * (
            to_int(b) if is_int(b) else product(b)
        )

    return reduce[Int, reducer](t, 1)


# TODO: Can't call this `max` otherwise the compiler incorrectly
# fails to recurse when calling this local function.
fn tuple_max(t: IntTuple) -> Int:
    @always_inline
    @parameter
    fn reducer(owned a: Int, b: IntTuple) -> Int:
        return max(a, to_int(b) if is_int(b) else tuple_max(b))

    # FIXME: limit.min_finite[DType.index]() doesn't seem to work
    alias int_min_val = -2147483648
    return reduce[Int, reducer](t, int_min_val)


fn apply[func: fn (Int) capturing [_] -> Int](t: IntTuple) -> IntTuple:
    if is_int(t):
        return func(to_int(t))
    var res = IntTuple()
    for e in t:
        res.append(apply[func](e))
    return res


fn apply(func: fn (IntTuple) -> IntTuple, t: IntTuple) -> IntTuple:
    var r = IntTuple()
    for v in t:
        r.append(func(v))
    return r


fn apply_zip[
    func: fn (IntTuple, IntTuple) -> IntTuple
](t1: IntTuple, t2: IntTuple) -> IntTuple:
    var r = IntTuple()
    for z in zip(t1, t2):
        r.append(func(z[0], z[1]))
    return r


fn apply_zip[
    func: fn (IntTuple, IntTuple) capturing [_] -> IntTuple
](t1: IntTuple, t2: IntTuple) -> IntTuple:
    var r = IntTuple()
    for z in zip(t1, t2):
        r.append(func(z[0], z[1]))
    return r


# fn apply_zip(
#     func: fn (IntTuple, IntTuple) escaping -> IntTuple,
#     t1: IntTuple,
#     t2: IntTuple,
# ) -> IntTuple:
#     var r = IntTuple()
#     for z in zip(t1, t2):
#         r.append(func(z[0], z[1]))
#     return r


fn apply_zip[
    func: fn (IntTuple, IntTuple, IntTuple) -> IntTuple
](t1: IntTuple, t2: IntTuple, t3: IntTuple,) -> IntTuple:
    var r = IntTuple()
    for z in zip(t1, t2, t3):
        r.append(func(z[0], z[1], z[2]))
    return r


# fn apply_zip(
#     func: fn (IntTuple, IntTuple, IntTuple) escaping -> IntTuple,
#     t1: IntTuple,
#     t2: IntTuple,
#     t3: IntTuple,
# ) -> IntTuple:
#     var r = IntTuple()
#     for z in zip(t1, t2, t3):
#         r.append(func(z[0], z[1], z[2]))
#     return r


fn tuple_min(a: IntTuple, b: IntTuple) -> IntTuple:
    if len(a) != len(b):
        abort("Tuple sizes don't match: " + str(len(a)) + " != " + str(len(b)))
    if is_int(a):
        if UNKNOWN_VALUE in (to_int(a), to_int(b)):
            return UNKNOWN_VALUE
        return min(to_int(a), to_int(b))
    return apply_zip[tuple_min](a, b)


fn inner_product(a: IntTuple, b: IntTuple) -> Int:
    if len(a) != len(b):
        abort("Tuple sizes don't match: " + str(len(a)) + " != " + str(len(b)))
    if is_int(a):
        return to_int(a) * to_int(b)
    var r: Int = 0
    for z in zip(a, b):
        r += inner_product(z[0], z[1])
    return r


fn abs(t: IntTuple) -> IntTuple:
    @parameter
    fn int_abs(x: Int) -> Int:
        return x.__abs__()

    return apply[int_abs](t)


# FIXME: the following crashes the compiler
# fn mul(lhs: IntTuple, rhs: Int) -> IntTuple:
#     @parameter
#     fn my_mul(x: Int) -> Int:
#         return x * rhs

#     return apply[my_mul](lhs)


# Multiply lhs tuple elements by rhs
#
fn mul(lhs: IntTuple, rhs: Int) -> IntTuple:
    if is_int(lhs):
        if UNKNOWN_VALUE in (is_int(lhs), rhs):
            return UNKNOWN_VALUE
        return to_int(lhs) * rhs

    var res = IntTuple()
    for e in lhs:
        res.append(mul(e, rhs))
    return res


# Return the product of elements in a mode
#
fn size(a: IntTuple) -> Int:
    return product(a)


# Test if two IntTuple have the same profile (hierarchical rank division)
#
fn congruent(a: IntTuple, b: IntTuple) -> Bool:
    if is_tuple(a) and is_tuple(b):
        if len(a) != len(b):
            return False
        for z in zip(a, b):
            if not congruent(z[0], z[1]):
                return False
        return True
    if is_int(a) and is_int(b):
        return True
    return False


fn apply_predicate[
    predicate: fn (IntTuple, IntTuple) -> Bool
](a: IntTuple, b: IntTuple) -> Bool:
    if is_tuple(a) and is_tuple(b):
        if len(a) != len(b):
            return False
        for z in zip(a, b):
            if not apply_predicate[predicate](z[0], z[1]):
                return False
        return True
    if is_int(a):
        return predicate(a, b)
    return False


# Test if two IntTuple have the similar profiles up to Shape A (hierarchical rank division)
# weakly_congruent is a partial order on A and B: A <= B
#
fn weakly_congruent(a: IntTuple, b: IntTuple) -> Bool:
    fn predicate(a: IntTuple, b: IntTuple) -> Bool:
        return True

    return apply_predicate[predicate](a, b)


#  Test if Shape A is compatible with Shape B:
#    the size of A and B are the same, and
#    any coordinate into A can also be used as a coordinate into B
# compatible is a partial order on A and B: A <= B
#
fn compatible(a: IntTuple, b: IntTuple) -> Bool:
    fn predicate(a: IntTuple, b: IntTuple) -> Bool:
        return to_int(a) == size(b)

    return apply_predicate[predicate](a, b)


#  Test if Shape A is weakly compatible with Shape B:
#    there exists a Shape C congruent to A such that compatible(elem_scale(A,C), B)
# weakly_compatible is a partial order on A and B: A <= B
#
@always_inline
fn weakly_compatible(a: IntTuple, b: IntTuple) -> Bool:
    fn predicate(a: IntTuple, b: IntTuple) -> Bool:
        return size(b) % to_int(a) == 0

    return apply_predicate[predicate](a, b)


# Exclusive prefix product with output congruent to input a
#
fn prefix_product(a: IntTuple, init: IntTuple = 1) -> IntTuple:
    if is_tuple(a):
        if is_tuple(init):  # tuple tuple
            if len(a) != len(init):
                abort("len(a) != len(init)")
            return apply_zip[prefix_product](a, init)
        else:  # tuple "int"
            var v_init = to_int(init)
            var r = IntTuple()
            for v in a:
                r.append(prefix_product(v, v_init))
                v_init = -1 if v_init == UNKNOWN_VALUE else v_init * product(v)
            return r
    else:
        if is_tuple(init):  # "int" tuple
            abort("'int' tuple not allowed")  # Error
            return IntTuple()
        else:  # "int" "int"
            return init


#  Division for Shapes
# Case Tuple Tuple:
#   Perform shape_div element-wise
# Case Tuple Int:
#   Fold the division of b across each element of a
#   Example: shape_div((4,5,6),40) -> shape_div((1,5,6),10) -> shape_div((1,1,6),2) -> (1,1,3)
# Case Int Tuple:
#   Return shape_div(a, product(b))
# Case Int Int:
#   Enforce the divisibility condition a % b == 0 || b % a == 0 when possible
#   Return a / b with rounding away from 0 (that is, 1 or -1 when a < b)
#
fn shape_div(a: IntTuple, b: IntTuple) -> IntTuple:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            if len(a) != len(b):
                abort(
                    "Tuple sizes don't match: "
                    + str(len(a))
                    + " != "
                    + str(len(b))
                )
            return apply_zip[shape_div](a, b)
        else:  # tuple "int"
            var vb = to_int(b)
            var r = IntTuple()
            for v in a:
                r.append(shape_div(v, vb))
                vb = to_int(shape_div(vb, product(v)))
            return r
    else:
        if is_tuple(b):  # "int" tuple
            return shape_div(a, product(b))
        else:  # "int" "int"
            var va = to_int(a)
            var vb = to_int(b)

            if va == UNKNOWN_VALUE or vb == UNKNOWN_VALUE:
                return UNKNOWN_VALUE

            if not (va % vb == 0 or vb % va == 0):
                abort("Incompatible shape values: " + str(va) + " " + str(vb))

            return va // vb if va % vb == 0 else signum(va * vb)


# idx2crd(i,s) splits an index into a coordinate within Shape
# via a colexicographical enumeration of coordinates in Shape.
# c0 = (idx / 1) % s0
# c1 = (idx / s0) % s1
# c2 = (idx / (s0 * s1)) % s2
# ...
#
@always_inline
fn idx2crd(
    idx: IntTuple, shape: IntTuple, _stride: IntTuple = IntTuple()
) -> IntTuple:
    var stride = _stride
    if len(stride) == 0:
        stride = prefix_product(shape)

    if is_tuple(idx):
        if is_tuple(shape):  # tuple tuple tuple
            if len(idx) != len(shape) or len(idx) != len(stride):
                abort("input shapes mismatch")

            return apply_zip[idx2crd](idx, shape, stride)
        else:  # tuple "int" "int"
            return abort[IntTuple]("Illegal inputs")  # Error
    else:
        if is_tuple(shape):  # "int" tuple tuple
            if len(shape) != len(stride):
                abort("input shapes mismatch")

            @parameter
            fn idx2crd2(shape: IntTuple, stride: IntTuple) -> IntTuple:
                return idx2crd(idx, shape, stride)

            return apply_zip[idx2crd2](shape, stride)
        else:  # "int" "int" "int"
            return UNKNOWN_VALUE if (
                to_int(idx) == UNKNOWN_VALUE
                or to_int(stride) == UNKNOWN_VALUE
                or to_int(shape) == UNKNOWN_VALUE
            ) else (to_int(idx) // to_int(stride)) % to_int(shape)


# Map a logical coordinate to a linear index
#
fn crd2idx(
    crd: IntTuple, shape: IntTuple, _stride: IntTuple = IntTuple()
) -> Int:
    var stride = _stride
    if len(stride) == 0:
        stride = prefix_product(shape)

    if is_tuple(crd):
        if is_tuple(shape):  # tuple tuple tuple
            if len(crd) != len(shape) or len(crd) != len(stride):
                abort("Shape mismatch")
            var r: Int = 0
            for z in zip(crd, shape, stride):
                r += crd2idx(z[0], z[1], z[2])
            return r
        else:  # tuple "int" "int"
            abort("Illegal input types")
            return 0
    else:
        var int_crd: Int = 0 if len(crd) == 0 else to_int(crd)

        if is_tuple(shape):  # "int" tuple tuple
            if len(shape) != len(stride):
                abort("Can't compute idx, shape != stride")
            var result: Int = 0
            for i in range(len(shape) - 1):
                result += crd2idx(
                    int_crd % product(shape[i]), shape[i], stride[i]
                )
                int_crd = int_crd // product(shape[i])
            return result + crd2idx(int_crd, shape[-1], stride[-1])
        else:  # "int" "int" "int"
            return int_crd * to_int(stride)


# Returns an IntTuple with same strcture as src filled with val.
#
fn fill_like(src: IntTuple, val: Int) -> IntTuple:
    if is_tuple(src):
        var res = IntTuple()
        for elem in src:
            res.append(fill_like(elem, val))
        return res
    return val


# Returns an IntTuple that combine all unknown dim from target to src.
#
fn propagate_unknown(src: IntTuple, target: IntTuple) -> IntTuple:
    if is_tuple(target):
        var dim = IntTuple()
        for d in zip(src, target):
            dim.append(propagate_unknown(d[0], d[1]))
        return dim

    if target == UNKNOWN_VALUE:
        return target
    return src


# Returns an IntTuple reversed e.g reverse(1, 2, (3, 4)) returns ((4, 3), 2, 1)
#
fn reverse(src: IntTuple) -> IntTuple:
    var res = IntTuple()
    if is_int(src):
        return src
    for i in range(len(src)):
        res.append(reverse(src[len(src) - i - 1]))
    return res


# Return the depth of an IntTuple, e.g depth((1, 2)) = 1, depth(1) = 0,
# depth((1, (1, 2))) = 2...etc
#
fn depth(src: IntTuple) -> Int:
    if is_int(src):
        return 0
    var res = 1
    for elem in src:
        res += depth(elem)
    return res
