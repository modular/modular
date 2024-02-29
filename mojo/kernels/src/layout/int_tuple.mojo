# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from kernel_utils.dynamic_tuple import *

from utils.variant import Variant

# IntTuple definition


# FIXME: This is a horrible hack around Mojo's lack or proper trait inheritance
struct IntDelegate(ElementDelegate):
    @always_inline
    @staticmethod
    fn is_equal[T: CollectionElement](a: Variant[T], b: Variant[T]) -> Bool:
        if a.isa[Int]() and b.isa[Int]():
            return a.get[Int]()[] == b.get[Int]()[]
        else:
            abort("Unexpected data type.")
            return False

    @always_inline
    @staticmethod
    fn to_string[T: CollectionElement](a: Variant[T]) -> String:
        if a.isa[Int]():
            return a.get[Int]()[]
        else:
            abort("Unexpected data type.")
            return "#"


alias IntTupleBase = DynamicTupleBase[Int, IntDelegate]
alias IntTuple = DynamicTuple[Int, IntDelegate]


@always_inline
fn signum(a: Int) -> Int:
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


@always_inline
fn int(owned v: IntTuple) -> Int:
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
    T: AnyRegType, func: fn (owned a: T, b: IntTuple) capturing -> T
](t: IntTuple, initializer: T) -> T:
    var result: T = initializer
    for e in t:
        result = func(result, e)
    return result


fn reduce[
    T: CollectionElement, func: fn (owned a: T, b: IntTuple) capturing -> T
](t: IntTuple, initializer: T) -> T:
    var result: T = initializer
    for e in t:
        result = func(result, e)
    return result


# IntTuple operations


fn flatten(t: IntTuple) -> IntTuple:
    @always_inline
    @parameter
    fn reducer(owned a: IntTuple, b: IntTuple) -> IntTuple:
        if is_int(b):
            a.append(int(b))
        else:
            for e in flatten(b):
                a.append(e)
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
        if int(z[0]) == int(z[1]):
            continue
        elif int(z[0]) < int(z[1]):
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
        return a + (int(b) if is_int(b) else sum(b))

    return reduce[Int, reducer](t, 0)


fn product(t: IntTuple) -> Int:
    @always_inline
    @parameter
    fn reducer(owned a: Int, b: IntTuple) -> Int:
        return a * (int(b) if is_int(b) else product(b))

    return reduce[Int, reducer](t, 1)


fn max(t: IntTuple) -> Int:
    @always_inline
    @parameter
    fn reducer(owned a: Int, b: IntTuple) -> Int:
        return math.max(a, int(b) if is_int(b) else max(b))

    return reduce[Int, reducer](t, 1)


fn apply[func: fn (Int) capturing -> Int](t: IntTuple) -> IntTuple:
    if is_int(t):
        return func(int(t))
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
    func: fn (IntTuple, IntTuple) capturing -> IntTuple
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


fn apply_zip3[
    func: fn (IntTuple, IntTuple, IntTuple) -> IntTuple
](t1: IntTuple, t2: IntTuple, t3: IntTuple,) -> IntTuple:
    var r = IntTuple()
    for z in zip3(t1, t2, t3):
        r.append(func(z[0], z[1], z[2]))
    return r


# fn apply_zip3(
#     func: fn (IntTuple, IntTuple, IntTuple) escaping -> IntTuple,
#     t1: IntTuple,
#     t2: IntTuple,
#     t3: IntTuple,
# ) -> IntTuple:
#     var r = IntTuple()
#     for z in zip3(t1, t2, t3):
#         r.append(func(z[0], z[1], z[2]))
#     return r


fn min(a: IntTuple, b: IntTuple) -> IntTuple:
    if len(a) != len(b):
        abort("Tuple sizes don't match: " + str(len(a)) + " != " + str(len(b)))
    if is_int(a):
        return math.min(int(a), int(b))
    return apply_zip[min](a, b)


fn inner_product(a: IntTuple, b: IntTuple) -> Int:
    if len(a) != len(b):
        abort("Tuple sizes don't match: " + str(len(a)) + " != " + str(len(b)))
    if is_int(a):
        return int(a) * int(b)
    var r: Int = 0
    for z in zip(a, b):
        r += inner_product(z[0], z[1])
    return r


fn abs(t: IntTuple) -> IntTuple:
    @parameter
    fn int_abs(x: Int) -> Int:
        return math.abs(x)

    return apply[int_abs](t)


# FIXME: the following crashes the compiler
# fn mul(lhs: IntTuple, rhs: Int) -> IntTuple:
#     @parameter
#     fn my_mul(x: Int) -> Int:
#         return x * rhs

#     return apply[my_mul](lhs)


fn mul(lhs: IntTuple, rhs: Int) -> IntTuple:
    if is_int(lhs):
        return int(lhs) * rhs

    var res = IntTuple()
    for e in lhs:
        res.append(mul(e, rhs))
    return res


# Exclusive prefix product with output congruent to input a
fn prefix_product(a: IntTuple, init: IntTuple = 1) -> IntTuple:
    if is_tuple(a):
        if is_tuple(init):  # tuple tuple
            if len(a) != len(init):
                abort("len(a) != len(init)")

            return apply_zip[prefix_product](a, init)
        else:  # tuple "int"
            var v_init = int(init)
            var r = IntTuple()
            for v in a:
                r.append(prefix_product(v, v_init))
                v_init = v_init * product(v)
            return r
    else:
        if is_tuple(init):  # "int" tuple
            abort("'int' tuple not allowed")  # Error
            return IntTuple()
        else:  # "int" "int"
            return init


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
            var vb = int(b)
            var r = IntTuple()
            for v in a:
                r.append(shape_div(v, vb))
                vb = int(shape_div(vb, product(v)))
            return r
    else:
        if is_tuple(b):  # "int" tuple
            return shape_div(a, product(b))
        else:  # "int" "int"
            var va = int(a)
            var vb = int(b)

            if not (va % vb == 0 or vb % va == 0):
                abort("Incompatible shape values: " + str(va) + " " + str(vb))

            return va // vb if va % vb == 0 else signum(va * vb)


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

            return apply_zip3[idx2crd](idx, shape, stride)
        else:  # tuple "int" "int"
            abort("Illegal inputs")  # Error
            return IntTuple()
    else:
        if is_tuple(shape):  # "int" tuple tuple
            if len(shape) != len(stride):
                abort("input shapes mismatch")

            @parameter
            fn idx2crd2(shape: IntTuple, stride: IntTuple) -> IntTuple:
                return idx2crd(idx, shape, stride)

            return apply_zip[idx2crd2](shape, stride)
        else:  # "int" "int" "int"
            return (int(idx) // int(stride)) % int(shape)


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
            for z in zip3(crd, shape, stride):
                r += crd2idx(z[0], z[1], z[2])
            return r
        else:  # tuple "int" "int"
            abort("Illegal input types")
            return 0
    else:
        var int_crd: Int = 0 if len(crd) == 0 else int(crd)

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
            return int_crd * int(stride)
