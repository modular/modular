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
            return a.get[Int]() == b.get[Int]()
        else:
            trap("Unexpected data type.")
            return False

    @always_inline
    @staticmethod
    fn to_string[T: CollectionElement](a: Variant[T]) -> String:
        if a.isa[Int]():
            return a.get[Int]()
        else:
            trap("Unexpected data type.")
            return "#"


alias IntTupleBase = DynamicTupleBase[Int, IntDelegate]
alias IntElement = IntTupleBase.Element
alias IntTuple = DynamicTuple[Int, IntDelegate]


@always_inline
fn signum(a: Int) -> Int:
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


@always_inline
fn int(owned v: IntTuple) -> Int:
    return v.value()


@always_inline
fn tuple(owned tv: IntTuple) -> IntTupleBase:
    return tv.tuple()


@always_inline
fn is_int(t: IntTuple) -> Bool:
    return t.content.isa[Int]()


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


# IntTuple reductions


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


# IntTuple zip iterator


@value
struct _ZipIter:
    var index: Int
    var a: IntTuple
    var b: IntTuple

    @always_inline
    fn __next__(inout self) -> IntTuple:
        self.index += 1
        return IntTuple(self.a[self.index - 1], self.b[self.index - 1])

    @always_inline
    fn __len__(self) -> Int:
        return math.min(len(self.a), len(self.b)) - self.index


@value
struct zip(Sized):
    var a: IntTuple
    var b: IntTuple

    alias IterType = _ZipIter

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self.a, self.b)

    @always_inline
    fn __len__(self) -> Int:
        return math.min(len(self.a), len(self.b))


# Layout operations


fn elementwise_min(a: IntTuple, b: IntTuple) -> IntTuple:
    if len(a) != len(b):
        trap("Tuple sizes don't match: " + str(len(a)) + " != " + str(len(b)))
    if is_int(a):
        return math.min(int(a), int(b))
    var res = IntTuple()
    for z in zip(a, b):
        res.append(elementwise_min(z[0], z[1]))
    return res


fn inner_product(a: IntTuple, b: IntTuple) -> Int:
    if len(a) != len(b):
        trap("Tuple sizes don't match: " + str(len(a)) + " != " + str(len(b)))
    if is_int(a):
        return int(a) * int(b)
    var r: Int = 0
    for z in zip(a, b):
        r += inner_product(z[0], z[1])
    return r


fn mul(lhs: IntTuple, rhs: Int) -> IntTuple:
    var res = IntTuple()
    for elem in lhs:
        if is_int(elem):
            res.append(elem.get[Int]() * rhs)
        else:
            for elem_i in tuple(elem):
                res.append(mul(IntTuple(elem_i), rhs))
    return res


fn shape_div(a: IntTuple, b: IntTuple) -> IntTuple:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            if len(a) != len(b):
                trap(
                    "Tuple sizes don't match: "
                    + str(len(a))
                    + " != "
                    + str(len(b))
                )

            var r = IntTuple()
            for z in zip(a, b):
                r.append(shape_div(z[0], z[1]))
            return r
        else:  # tuple "int"
            var vb = int(b)
            var r = IntTuple()
            for v in tuple(a):
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
                trap("Incompatible shape values: " + str(va) + " " + str(vb))

            return va // vb if va % vb == 0 else signum(va * vb)


fn crd2idx(crd: IntTuple, shape: IntTuple, stride: IntTuple) -> Int:
    if is_tuple(crd):
        if not is_tuple(shape):
            trap("crd and shape should be both IntTuple!")
        var res = 0
        for i in range(len(shape)):
            res += crd2idx(crd[i], shape[i], stride[i])
        return res

    if is_tuple(shape):
        if len(shape) != len(stride):
            trap("Can't compute idx, shape != stride")
        var result = 0
        var curd_int = int(crd)
        for i in range(len(shape) - 1):
            result += crd2idx(
                int(curd_int) % product(shape[i]), shape[i], stride[i]
            )
            curd_int = curd_int // product(shape[i])
        return result + crd2idx(
            curd_int, shape[len(shape) - 1], stride[len(stride) - 1]
        )

    return int(stride) * int(crd)


fn idx2crd(idx: Int, shape: IntTuple, stride: IntTuple) -> IntTuple:
    if is_tuple(shape):
        if len(shape) != len(stride):
            trap("shape and stride should be the same length")
        var res = IntTuple()
        for i in range(len(shape)):
            res.append(idx2crd(idx, shape[i], stride[i]))
        return res

    return (idx // int(stride)) % int(shape)
