# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from utils.variant import Variant
from collections.dict import EqualityComparable


@value
struct _DynamicTupleIter[T: CollectionElement]:
    alias BaseType = DynamicTupleBase[T]
    alias ElementType = Self.BaseType.Element

    var index: Int
    var src: Self.ElementType

    @always_inline
    fn __next__(inout self) -> Self.ElementType:
        self.index += 1
        return self.src.get[Int]() if self.src.isa[T]() else self.src.get[
            Self.BaseType
        ]()[self.index - 1]

    @always_inline
    fn __len__(self) -> Int:
        return (
            1 if self.src.isa[T]() else len(self.src.get[Self.BaseType]())
        ) - self.index


struct DynamicTupleBase[T: CollectionElement](
    CollectionElement, Sized, Stringable
):
    alias Element = Variant[T, Self]

    alias IterType = _DynamicTupleIter[T]

    var elts: DynamicVector[Self.Element]

    @always_inline
    fn __init__(inout self: Self, *values: Self.Element):
        self.elts = DynamicVector[Self.Element]()
        for v in values:
            self.elts.append(v[])

    @always_inline
    fn __init__(inout self: Self, value: DynamicTuple[T]):
        self.elts = DynamicVector[Self.Element]()
        self.elts.append(value.content)

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.elts = existing.elts ^

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.elts = existing.elts

    @always_inline
    fn append(inout self, owned *values: DynamicTuple[T]):
        self.elts.reserve(len(values))
        for v in values:
            self.elts.append(v[].content)

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

    @staticmethod
    fn to_string(v: Self.Element) -> String:
        if v.isa[Int]():
            return v.get[Int]()
        else:
            var result = String("(")
            if v.isa[Self]():
                var elts = v.get[Self]().elts
                for i in range(len(elts)):
                    let e: Self.Element = elts[i]
                    result += Self.to_string(e)
                    if i < len(elts) - 1:
                        result += ", "
            return result + ")"

    @staticmethod
    fn is_equal(a: Self.Element, b: Self.Element) -> Bool:
        if a.isa[Int]() and b.isa[Int]():
            return is_equal(a.get[Int](), b.get[Int]())
        if a.isa[Self]() and b.isa[Self]():
            let ta = a.get[Self]()
            let tb = b.get[Self]()
            if len(ta) == len(tb):
                for i in range(len(ta)):
                    if not Self.is_equal(ta[i], tb[i]):
                        return False
                return True
        return False

    fn __str__(self) -> String:
        return Self.to_string(self)

    fn __eq__(self, other: Self) -> Bool:
        return Self.is_equal(self, other)

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return not Self.is_equal(self, other)


@value
struct DynamicTuple[T: CollectionElement](CollectionElement, Sized, Stringable):
    alias BaseType = DynamicTupleBase[T]
    alias ElementType = Self.BaseType.Element

    var content: Self.ElementType

    alias IterType = _DynamicTupleIter[T]

    @always_inline
    fn __init__(inout self: Self, value: Self.ElementType):
        self.content = value

    @always_inline
    fn __init__(inout self: Self, value: Self.BaseType):
        self.content = value

    @always_inline
    fn __init__(inout self: Self, value: T):
        self.content = value

    @always_inline
    fn __init__(inout self: Self, *values: Self):
        var t = Self.BaseType()
        for v in values:
            t.append(v[])
        self.content = t

    @always_inline
    fn __len__(self) -> Int:
        if self.is_value():
            return 1
        else:
            return len(self.tuple())

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self.content)

    @always_inline
    fn is_tuple(self) -> Bool:
        return self.content.isa[Self.BaseType]()

    @always_inline
    fn tuple(self) -> Self.BaseType:
        return self.content.get[Self.BaseType]()

    @always_inline
    fn is_value(self) -> Bool:
        return self.content.isa[T]()

    @always_inline
    fn value(self) -> T:
        return self.content.get[T]()

    fn __getitem__(self, index: Int) -> Self:
        if self.is_value():
            return self.value()
        else:
            var r = Self()
            r.content = self.tuple()[index]
            return r

    @always_inline
    fn __setitem__(inout self, index: Int, val: Self):
        if self.is_value() and val.is_value():
            self.content = val.value()
        else:
            var new_content: Self.BaseType = self.tuple()
            if val.is_value():
                new_content[index] = val.value()
            else:
                new_content[index] = val.tuple()
            self.content = new_content

    @always_inline
    fn append(inout self, owned *values: Self):
        var new_content: Self.BaseType
        if self.is_tuple():
            new_content = self.tuple()
        else:
            new_content = Self.BaseType(self.value())
        for v in values:
            new_content.append(v[])
        self.content = new_content

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        if self.is_value() and other.is_value():
            return Self.BaseType.is_equal(self.value(), other.value())
        if self.is_tuple() and other.is_tuple():
            return self.tuple() == other.tuple()
        return False

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return not self == other

    fn __str__(self) -> String:
        return Self.BaseType.to_string(self.content)


# Parameter type support functions


fn to_string[T: Stringable](v: T) -> String:
    return str(v)


fn is_equal[T: EqualityComparable](a: T, b: T) -> Bool:
    return a == b
