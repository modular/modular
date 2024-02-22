# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from utils.variant import Variant


# FIXME: This is a horrible hack around Mojo's lack or proper trait inheritance
trait ElementDelegate:
    @staticmethod
    fn is_equal[T: CollectionElement](a: Variant[T], b: Variant[T]) -> Bool:
        pass

    @staticmethod
    fn to_string[T: CollectionElement](a: Variant[T]) -> String:
        pass


struct DefaultDelegate(ElementDelegate):
    @staticmethod
    fn is_equal[T: CollectionElement](a: Variant[T], b: Variant[T]) -> Bool:
        return False

    @staticmethod
    fn to_string[T: CollectionElement](a: Variant[T]) -> String:
        return "#"


@value
struct _DynamicTupleIter[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
]:
    alias BaseType = DynamicTupleBase[T, D]
    alias ElementType = Self.BaseType.Element

    var index: Int
    var src: Self.ElementType

    @always_inline
    fn __next__(inout self) -> Self.ElementType:
        self.index += 1
        if self.src.isa[T]():
            return self.src.get[T]()[]
        else:
            return self.src.get[Self.BaseType]()[][self.index - 1]

    @always_inline
    fn __len__(self) -> Int:
        return (
            1 if self.src.isa[T]() else len(self.src.get[Self.BaseType]()[])
        ) - self.index


struct DynamicTupleBase[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](CollectionElement, Sized, Stringable, EqualityComparable):
    alias Element = Variant[T, Self]
    alias IterType = _DynamicTupleIter[T, D]

    var elts: DynamicVector[Self.Element]

    @always_inline
    fn __init__(inout self: Self, value: T):
        self.elts = DynamicVector[Self.Element]()
        self.elts.append(value)

    @always_inline
    fn __init__(inout self: Self):
        self.elts = DynamicVector[Self.Element]()

    @always_inline
    fn __init__(inout self: Self, v1: Self.Element):
        self.elts = DynamicVector[Self.Element]()
        self.elts.append(v1)

    @always_inline
    fn __init__(inout self: Self, v1: Self.Element, v2: Self.Element):
        self.elts = DynamicVector[Self.Element]()
        self.elts.append(v1)
        self.elts.append(v2)

    @always_inline
    fn __init__(
        inout self: Self, v1: Self.Element, v2: Self.Element, v3: Self.Element
    ):
        self.elts = DynamicVector[Self.Element]()
        self.elts.append(v1)
        self.elts.append(v2)
        self.elts.append(v3)

    @always_inline
    fn __init__(
        inout self: Self,
        v1: Self.Element,
        v2: Self.Element,
        v3: Self.Element,
        v4: Self.Element,
    ):
        self.elts = DynamicVector[Self.Element]()
        self.elts.append(v1)
        self.elts.append(v2)
        self.elts.append(v3)
        self.elts.append(v4)

    @always_inline
    fn __init__(inout self: Self, value: DynamicTuple[T, D]):
        self.elts = DynamicVector[Self.Element]()
        self.elts.append(value.content)

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.elts = existing.elts ^

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.elts = existing.elts

    @always_inline
    fn append(inout self, owned v1: DynamicTuple[T, D]):
        self.elts.append(v1.content)

    @always_inline
    fn append(
        inout self, owned v1: DynamicTuple[T, D], owned v2: DynamicTuple[T, D]
    ):
        self.elts.append(v1.content)
        self.elts.append(v2.content)

    @always_inline
    fn append(
        inout self,
        owned v1: DynamicTuple[T, D],
        owned v2: DynamicTuple[T, D],
        owned v3: DynamicTuple[T, D],
    ):
        self.elts.append(v1.content)
        self.elts.append(v2.content)
        self.elts.append(v3.content)

    @always_inline
    fn append(
        inout self,
        owned v1: DynamicTuple[T, D],
        owned v2: DynamicTuple[T, D],
        owned v3: DynamicTuple[T, D],
        owned v4: DynamicTuple[T, D],
    ):
        self.elts.append(v1.content)
        self.elts.append(v2.content)
        self.elts.append(v3.content)
        self.elts.append(v4.content)

    @always_inline
    fn __getitem__(self, index: Int) -> Self.Element:
        if index < 0 or index > len(self.elts):
            trap("Index out of bounds.")
        return self.elts[index]

    @always_inline
    fn __setitem__(inout self, index: Int, val: Self.Element):
        if index < 0 or index > len(self.elts):
            trap("Index out of bounds.")
        self.elts[index] = val

    @always_inline
    fn __len__(self) -> Int:
        return len(self.elts)

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self)

    @always_inline
    @staticmethod
    fn rewrap(v: Self.Element) -> Variant[T]:
        return Variant[T](v.get[T]()[])

    @staticmethod
    fn to_string(v: Self.Element) -> String:
        if v.isa[T]():
            return D.to_string[T](Self.rewrap(v))
        else:
            var result = String("(")
            if v.isa[Self]():
                var elts = v.get[Self]()[].elts
                for i in range(len(elts)):
                    var e: Self.Element = elts[i]
                    result += Self.to_string(e)
                    if i < len(elts) - 1:
                        result += ", "
            return result + ")"

    @staticmethod
    fn is_equal(a: Self.Element, b: Self.Element) -> Bool:
        if a.isa[T]() and b.isa[T]():
            return D.is_equal[T](Self.rewrap(a), Self.rewrap(b))
        if a.isa[Self]() and b.isa[Self]():
            var ta = a.get[Self]()[]
            var tb = b.get[Self]()[]
            if len(ta) == len(tb):
                for i in range(len(ta)):
                    if not Self.is_equal(ta[i], tb[i]):
                        return False
                return True
        return False

    @always_inline
    fn __str__(self) -> String:
        return Self.to_string(self)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return Self.is_equal(self, other)

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return not Self.is_equal(self, other)


@value
struct DynamicTuple[T: CollectionElement, D: ElementDelegate = DefaultDelegate](
    CollectionElement, Sized, Stringable, EqualityComparable
):
    alias BaseType = DynamicTupleBase[T, D]
    alias ElementType = Self.BaseType.Element
    alias IterType = _DynamicTupleIter[T, D]

    var content: Self.ElementType

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
    fn __init__(inout self: Self):
        var t = Self.BaseType()
        self.content = t

    @always_inline
    fn __init__(inout self: Self, v1: Self):
        var t = Self.BaseType()
        t.append(v1)
        self.content = t

    #
    @always_inline
    fn __init__(inout self: Self, v1: Self, v2: Self):
        var t = Self.BaseType()
        t.append(v1)
        t.append(v2)
        self.content = t

    #
    @always_inline
    fn __init__(inout self: Self, v1: Self, v2: Self, v3: Self):
        var t = Self.BaseType()
        t.append(v1)
        t.append(v2)
        t.append(v3)
        self.content = t

    #
    @always_inline
    fn __init__(inout self: Self, v1: Self, v2: Self, v3: Self, v4: Self):
        var t = Self.BaseType()
        t.append(v1)
        t.append(v2)
        t.append(v3)
        t.append(v4)
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
        return self.content.get[Self.BaseType]()[]

    @always_inline
    fn is_value(self) -> Bool:
        return self.content.isa[T]()

    @always_inline
    fn value(self) -> T:
        return self.content.get[T]()[]

    @always_inline
    fn __getitem__(self, index: Int) -> Self:
        if self.is_value():
            if index != 0:
                trap("Index should be 0 for value items.")
            return self.value()
        else:
            var r = Self()
            r.content = self.tuple()[index]
            return r

    @always_inline
    fn __setitem__(inout self, index: Int, val: Self):
        if self.is_value() and val.is_value():
            if index != 0:
                trap("Index should be 0 for value items.")
            self.content = val.value()
        else:
            var new_content: Self.BaseType = self.tuple()
            if val.is_value():
                new_content[index] = val.value()
            else:
                new_content[index] = val.tuple()
            self.content = new_content

    @always_inline
    fn append(inout self, owned v1: Self):
        var new_content: Self.BaseType
        if self.is_tuple():
            new_content = self.tuple()
        else:
            new_content = Self.BaseType(self.value())
        new_content.append(v1)
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

    @always_inline
    fn __str__(self) -> String:
        return Self.BaseType.to_string(self.content)
