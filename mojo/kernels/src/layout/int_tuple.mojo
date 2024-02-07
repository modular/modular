# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from utils.variant import Variant

# IntTuple definition


@value
struct _IntTupleIter:
    var index: Int
    var src: IntTupleBase.Element

    @always_inline
    fn __next__(inout self) -> IntTupleBase.Element:
        self.index += 1
        return self.src.get[Int]() if is_int(self.src) else self.src.get[
            IntTupleBase
        ]()[self.index - 1]

    @always_inline
    fn __len__(self) -> Int:
        return (
            1 if is_int(self.src) else len(self.src.get[IntTupleBase]())
        ) - self.index


struct IntTupleBase(CollectionElement, Sized, Stringable):
    alias Element = Variant[Int, Self]

    alias IterType = _IntTupleIter

    var elts: DynamicVector[Self.Element]

    @always_inline
    fn __init__(inout self: Self, owned *values: Self.Element):
        self.elts = DynamicVector[Self.Element]()
        for v in values:
            self.append(v[])

    @always_inline
    fn __init__(inout self: Self, owned value: IntTuple):
        self.elts = DynamicVector[Self.Element]()
        self.append(value)

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.elts = existing.elts ^

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.elts = existing.elts

    @always_inline
    fn append(inout self, owned *values: IntTuple):
        self.elts.reserve(len(values))
        for v in values:
            self.elts.append(v[].value)

    @always_inline
    fn __getitem__(self, index: Int) -> Self.Element:
        return self.elts[index]

    @always_inline
    fn __setitem__(inout self, index: Int, val: Self.Element):
        self.elts[index] = val

    @always_inline
    fn __len__(self) -> Int:
        return len(self.elts)

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self)

    fn __str__(self) -> String:
        fn reducer(owned a: String, b: IntTuple) -> String:
            if len(a) > 0:
                a += ", "
            return a + (int(b) if is_int(b) else str(b))

        return "(" + reduce(reducer, self, String()) + ")"

    fn __eq__(self, other: IntTupleBase) -> Bool:
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            if is_int(self[i]):
                if self[i].get[Int]() != other[i].get[Int]():
                    return False
            else:
                if self[i].get[IntTupleBase]() != other[i].get[IntTupleBase]():
                    return False
        return True

    @always_inline
    fn __ne__(self, other: IntTupleBase) -> Bool:
        return not self == other


@value
struct IntTuple(CollectionElement, Sized, Intable, Stringable):
    var value: IntTupleBase.Element

    alias IterType = _IntTupleIter

    @always_inline
    fn __init__(inout self: Self, owned value: IntTupleBase.Element):
        self.value = value

    @always_inline
    fn __init__(inout self: Self, owned value: IntTupleBase):
        self.value = value

    @always_inline
    fn __init__(inout self: Self, value: Int):
        self.value = value

    @always_inline
    fn __init__(inout self: Self, owned *values: IntTuple):
        var t = IntTupleBase()
        for v in values:
            t.append(v[])
        self.value = t

    @always_inline
    fn append(inout self, owned *values: IntTuple):
        var new_value: IntTupleBase
        if is_tuple(self):
            new_value = tuple(self)
        else:
            new_value = IntTupleBase(int(self))
        for v in values:
            new_value.append(v[])
        self.value = new_value

    @always_inline
    fn __getitem__(self, index: Int) -> Self:
        if is_int(self):
            return int(self)
        else:
            return tuple(self)[index]

    @always_inline
    fn __setitem__(inout self, index: Int, val: Self):
        if is_int(self) and is_int(val):
            self.value = int(val)
        else:
            var new_value: IntTupleBase = tuple(self)
            if is_int(val):
                new_value[index] = int(val)
            else:
                new_value[index] = tuple(val)
            self.value = new_value

    fn __mul__(self, rhs: Int) -> Self:
        var res = IntTuple()
        for elem in self:
            if is_int(elem):
                res.append(elem.get[Int]() * rhs)
            else:
                for elem_i in tuple(elem):
                    res.append(IntTuple(elem_i) * rhs)
        return res

    @always_inline
    fn __len__(self) -> Int:
        return 1 if is_int(self.value) else len(self.value.get[IntTupleBase]())

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self.value)

    @always_inline
    fn __str__(self) -> String:
        if is_int(self.value):
            return self.value.get[Int]()
        else:
            return self.value.get[IntTupleBase]()

    @always_inline
    fn __eq__(self, other: IntTuple) -> Bool:
        if is_int(self.value) and is_int(other):
            return self.value.get[Int]() == other.value.get[Int]()
        if is_tuple(self) and is_tuple(other):
            return (
                self.value.get[IntTupleBase]()
                == other.value.get[IntTupleBase]()
            )
        return False

    @always_inline
    fn __ne__(self, other: IntTuple) -> Bool:
        return not self == other

    @always_inline
    fn __int__(self) -> Int:
        return self.value.get[Int]()

    @always_inline
    fn tuple(self) -> IntTupleBase:
        return self.value.get[IntTupleBase]()


# IntTuple operations, see: https://github.com/NVIDIA/cutlass/blob/main/python/pycute/int_tuple.py


@always_inline
fn tuple(owned tv: IntTuple) -> IntTupleBase:
    return tv.tuple()


@always_inline
fn signum(a: Int) -> Int:
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


@always_inline
fn is_int(t: IntTuple) -> Bool:
    return t.value.isa[Int]()


@always_inline
fn is_tuple(t: IntTuple) -> Bool:
    return t.value.isa[IntTupleBase]()


# Python-style reduce functions
# FIXME: Can we unify the two versions?


fn reduce[
    T: AnyRegType
](func: fn (owned a: T, b: IntTuple) -> T, t: IntTuple, initializer: T) -> T:
    var result: T = initializer
    for e in t:
        result = func(result, e)
    return result


fn reduce[
    T: CollectionElement
](func: fn (owned a: T, b: IntTuple) -> T, t: IntTuple, initializer: T) -> T:
    var result: T = initializer
    for e in t:
        result = func(result, e)
    return result


# IntTuple reductions


fn flatten(t: IntTuple) -> IntTuple:
    fn reducer(owned a: IntTuple, b: IntTuple) -> IntTuple:
        if is_int(b):
            a.append(int(b))
        else:
            # FIXME: a.append(e) should be enough
            for e in flatten(b):
                a.append(e)
        return a

    return reduce(reducer, t, IntTuple())


fn sum(t: IntTuple) -> Int:
    fn reducer(owned a: Int, b: IntTuple) -> Int:
        return a + (int(b) if is_int(b) else sum(b))

    return reduce(reducer, t, 0)


fn product(t: IntTuple) -> Int:
    fn reducer(owned a: Int, b: IntTuple) -> Int:
        return a * (int(b) if is_int(b) else product(b))

    return reduce(reducer, t, 1)


fn max(t: IntTuple) -> Int:
    fn reducer(owned a: Int, b: IntTuple) -> Int:
        return math.max(a, int(b) if is_int(b) else max(b))

    return reduce(reducer, t, 1)


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


fn elementwise_min(a: IntTuple, b: IntTuple) raises -> IntTuple:
    if len(a) != len(b):
        raise Error(
            "Tuple sizes don't match: " + str(len(a)) + " != " + str(len(b))
        )
    if is_int(a):
        return math.min(int(a), int(b))
    var res = IntTuple()
    for z in zip(a, b):
        res.append(elementwise_min(z[0], z[1]))
    return res


fn inner_product(a: IntTuple, b: IntTuple) raises -> Int:
    if len(a) != len(b):
        raise Error(
            "Tuple sizes don't match: " + str(len(a)) + " != " + str(len(b))
        )
    if is_int(a):
        return int(a) * int(b)
    var r: Int = 0
    for z in zip(a, b):
        r += inner_product(z[0], z[1])
    return r


fn shape_div(a: IntTuple, b: IntTuple) raises -> IntTuple:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            if len(a) != len(b):
                raise Error(
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
            let va = int(a)
            let vb = int(b)

            if not (va % vb == 0 or vb % va == 0):
                raise Error(
                    "Incompatible shape values: " + str(va) + " " + str(vb)
                )

            return va // vb if va % vb == 0 else signum(va * vb)


fn crd2idx(crd: IntTuple, shape: IntTuple, stride: IntTuple) raises -> Int:
    if is_tuple(crd):
        if not is_tuple(shape):
            raise Error("crd and shape should be both IntTuple!")
        var res = 0
        for i in range(len(shape)):
            res += crd2idx(crd[i], shape[i], stride[i])
        return res

    if is_tuple(shape):
        if len(shape) != len(stride):
            raise Error("Can't compute idx, shape != stride")
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


fn idx2crd(idx: Int, shape: IntTuple, stride: IntTuple) raises -> IntTuple:
    if is_tuple(shape):
        if len(shape) != len(stride):
            raise Error("shape and stride should be the same length")
        var res = IntTuple()
        for i in range(len(shape)):
            res.append(idx2crd(idx, shape[i], stride[i]))
        return res

    return (idx // int(stride)) % int(shape)
