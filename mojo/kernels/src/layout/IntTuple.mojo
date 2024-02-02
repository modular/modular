# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from utils.variant import Variant

# Python-style reduce functions
# FIXME: Can we unify the two versions?


fn reduce[
    T: AnyRegType
](
    func: fn (owned a: T, b: IntTupleVal) -> T, t: IntTupleVal, initializer: T
) -> T:
    var result: T = initializer
    for e in t:
        result = func(result, e)
    return result


fn reduce[
    T: CollectionElement
](
    func: fn (owned a: T, b: IntTupleVal) -> T, t: IntTupleVal, initializer: T
) -> T:
    var result: T = initializer
    for e in t:
        result = func(result, e)
    return result


# IntTuple definition


@value
struct _IntTupleIter:
    var index: Int
    var src: IntTuple.Element

    fn __next__(inout self) -> IntTuple.Element:
        self.index += 1
        return self.src.get[Int]() if is_int(self.src) else self.src.get[
            IntTuple
        ]()[self.index - 1]

    fn __len__(self) -> Int:
        return (
            1 if is_int(self.src) else len(self.src.get[IntTuple]())
        ) - self.index


struct IntTuple(CollectionElement, Sized, Stringable):
    alias Element = Variant[Int, Self]

    alias IterType = _IntTupleIter

    var elts: DynamicVector[Self.Element]

    fn __init__(inout self: Self, owned *values: Self.Element):
        self.elts = DynamicVector[Self.Element](len(values))
        for v in values:
            self.append(v[])

    fn __init__(inout self: Self, owned value: IntTupleVal):
        if is_tuple(value):
            self.elts = value.tuple().elts
        else:
            self.elts = DynamicVector[Self.Element](1)
            self.append(int(value))

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.elts = existing.elts ^

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.elts = existing.elts

    @always_inline
    fn append(inout self, owned *values: IntTupleVal):
        self.elts.reserve(len(values))
        for v in values:
            self.elts.append(v[].value)

    @always_inline
    fn __getitem__(self, index: Int) -> Self.Element:
        return self.elts[index]

    @always_inline
    fn __setitem__(inout self, i: Int, val: Self.Element):
        self.elts[i] = val

    @always_inline
    fn __len__(self) -> Int:
        return len(self.elts)

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self)

    fn __str__(self) -> String:
        fn reducer(owned a: String, b: IntTupleVal) -> String:
            if len(a) > 0:
                a += ", "
            return a + (int(b) if is_int(b) else str(b))

        return "(" + reduce(reducer, self, String()) + ")"

    fn __eq__(self, other: IntTuple) -> Bool:
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            if is_int(self[i]):
                if self[i].get[Int]() != other[i].get[Int]():
                    return False
            else:
                if self[i].get[IntTuple]() != other[i].get[IntTuple]():
                    return False
        return True

    fn __ne__(self, other: IntTuple) -> Bool:
        return not self == other


@value
struct IntTupleVal(CollectionElement, Sized, Intable, Stringable):
    var value: IntTuple.Element

    alias IterType = _IntTupleIter

    @always_inline
    fn __init__(inout self: Self, owned value: IntTuple.Element):
        self.value = value

    @always_inline
    fn __init__(inout self: Self, owned value: IntTuple):
        self.value = value

    @always_inline
    fn __init__(inout self: Self, owned value: Int):
        self.value = value

    @always_inline
    fn __len__(self) -> Int:
        return 1 if is_int(self.value) else len(self.value.get[IntTuple]())

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self.value)

    @always_inline
    fn __str__(self) -> String:
        if is_int(self.value):
            return self.value.get[Int]()
        else:
            return self.value.get[IntTuple]()

    @always_inline
    fn __eq__(self, other: IntTupleVal) -> Bool:
        if is_int(self.value) and is_int(other):
            return self.value.get[Int]() == other.value.get[Int]()
        if is_tuple(self) and is_tuple(other):
            return self.value.get[IntTuple]() == other.value.get[IntTuple]()
        return False

    @always_inline
    fn __ne__(self, other: IntTupleVal) -> Bool:
        return not self == other

    @always_inline
    fn __int__(self) -> Int:
        return self.value.get[Int]()

    fn tuple(self) -> IntTuple:
        return self.value.get[IntTuple]()


# IntTuple operations, see: https://github.com/NVIDIA/cutlass/blob/main/python/pycute/int_tuple.py


fn tuple(owned tv: IntTupleVal) -> IntTuple:
    return tv.tuple()


fn signum(a: Int) -> Int:
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


fn is_int(t: IntTupleVal) -> Bool:
    return t.value.isa[Int]()


fn is_tuple(t: IntTupleVal) -> Bool:
    return t.value.isa[IntTuple]()


# IntTuple reductions


fn flatten(t: IntTupleVal) -> IntTuple:
    fn reducer(owned a: IntTuple, b: IntTupleVal) -> IntTuple:
        if is_int(b):
            a.append(int(b))
        else:
            # FIXME: a.append(e) should be enough
            for e in flatten(b):
                a.append(e)
        return a

    return reduce(reducer, t, IntTuple())


fn sum(t: IntTupleVal) -> Int:
    fn reducer(owned a: Int, b: IntTupleVal) -> Int:
        return a + (int(b) if is_int(b) else sum(b))

    return reduce(reducer, t, 0)


fn product(t: IntTupleVal) -> Int:
    fn reducer(owned a: Int, b: IntTupleVal) -> Int:
        return a * (int(b) if is_int(b) else product(b))

    return reduce(reducer, t, 1)


fn max(t: IntTupleVal) -> Int:
    fn reducer(owned a: Int, b: IntTupleVal) -> Int:
        return math.max(a, int(b) if is_int(b) else max(b))

    return reduce(reducer, t, 1)


# Layout operations


fn inner_product(a: IntTupleVal, b: IntTupleVal) raises -> Int:
    if is_tuple(a):  # tuple tuple
        let ta = tuple(a)
        let tb = tuple(b)
        if len(ta) != len(tb):
            raise Error(
                "Tuple sizes don't match: "
                + str(len(ta))
                + " != "
                + str(len(tb))
            )

        var r: Int = 0
        for i in range(len(ta)):
            r += inner_product(ta[i], tb[i])
        return r
    else:  # "int" "int"
        if not is_int(b):
            raise Error("Input types don't match.")
        return int(a) * int(b)


fn shape_div(a: IntTupleVal, b: IntTupleVal) raises -> IntTupleVal:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            let ta = tuple(a)
            let tb = tuple(b)
            if len(ta) != len(tb):
                raise Error(
                    "Tuple sizes don't match: "
                    + str(len(ta))
                    + " != "
                    + str(len(tb))
                )

            var r = IntTuple()
            for i in range(len(ta)):
                let x = ta[i]
                let y = tb[i]
                r.append(shape_div(x, y))
            return r
        else:  # tuple "int"
            var r = IntTuple()
            let ta = tuple(a)
            var vb = int(b)
            for v in ta:
                r.append(shape_div(v, vb))
                vb = int(shape_div(vb, IntTuple(product(v))))
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
