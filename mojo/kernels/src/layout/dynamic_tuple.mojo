# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math
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


struct DynamicTupleBase[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](CollectionElement, Sized, Stringable, EqualityComparable):
    alias Element = Variant[T, Self]

    var _elements: List[Self.Element]

    @always_inline
    fn __init__(inout self: Self):
        self._elements = List[Self.Element]()

    # FIXME: We should have a single variadic constructor (https://github.com/modularml/modular/issues/32000)
    # @always_inline
    # fn __init__(inout self: Self, *v: Self.Element):
    #     self._elements = List[Self.Element](capacity=len(v))
    #     for e in v:
    #         self._elements.append(e[])

    @always_inline
    fn __init__(inout self: Self, v: Self.Element):
        self._elements = List[Self.Element](capacity=1)
        self._elements.append(v)

    @always_inline
    fn __init__(inout self, owned v1: Self.Element, owned v2: Self.Element):
        self._elements = List[Self.Element](capacity=2)
        self._elements.append(v1)
        self._elements.append(v2)

    @always_inline
    fn __init__(
        inout self,
        owned v1: Self.Element,
        owned v2: Self.Element,
        owned v3: Self.Element,
    ):
        self._elements = List[Self.Element](capacity=3)
        self._elements.append(v1)
        self._elements.append(v2)
        self._elements.append(v3)

    @always_inline
    fn __init__(
        inout self,
        owned v1: Self.Element,
        owned v2: Self.Element,
        owned v3: Self.Element,
        owned v4: Self.Element,
    ):
        self._elements = List[Self.Element](capacity=4)
        self._elements.append(v1)
        self._elements.append(v2)
        self._elements.append(v3)
        self._elements.append(v4)

    @always_inline
    fn __init__(
        inout self,
        owned v1: Self.Element,
        owned v2: Self.Element,
        owned v3: Self.Element,
        owned v4: Self.Element,
        owned v5: Self.Element,
    ):
        self._elements = List[Self.Element](capacity=5)
        self._elements.append(v1)
        self._elements.append(v2)
        self._elements.append(v3)
        self._elements.append(v4)
        self._elements.append(v5)

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self._elements = existing._elements^

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self._elements = existing._elements

    @always_inline
    fn append(inout self: Self, value: Self.Element):
        self._elements.append(value)

    @always_inline
    fn __getitem__(self, _idx: Int) -> Self.Element:
        var idx = len(self) + _idx if _idx < 0 else _idx

        if idx < 0 or idx >= len(self._elements):
            abort("Index out of bounds.")
        return self._elements[idx]

    @always_inline
    fn _adjust_span(self, owned span: Slice) -> Slice:
        if span.start < 0:
            span.start = len(self) + span.start

        if not span._has_end():
            span.end = len(self)
        elif span.end < 0:
            span.end = len(self) + span.end

        return span

    @always_inline
    fn __getitem__(self, owned span: Slice) -> Self:
        span = self._adjust_span(span)
        var result = Self()
        result._elements.reserve(len(span))
        for i in range(span.start, span.end, span.step):
            result._elements.append(self[i])
        return result

    @always_inline
    fn __setitem__(inout self, _idx: Int, val: Self.Element):
        var idx = len(self) + _idx if _idx < 0 else _idx

        if idx < 0 or idx >= len(self._elements):
            abort("Index out of bounds.")
        self._elements[idx] = val

    @always_inline
    fn __len__(self) -> Int:
        return len(self._elements)

    @staticmethod
    fn to_string(v: Self.Element) -> String:
        if v.isa[T]():
            return D.to_string[T](v.get[T]()[])
        else:
            var result = String("(")
            if v.isa[Self]():
                var _elements = v.get[Self]()[]._elements
                for i in range(len(_elements)):
                    var e: Self.Element = _elements[i]
                    result += Self.to_string(e)
                    if i < len(_elements) - 1:
                        result += ", "
            return result + ")"

    @staticmethod
    fn is_equal(a: Self.Element, b: Self.Element) -> Bool:
        if a.isa[T]() and b.isa[T]():
            return D.is_equal[T](a.get[T]()[], b.get[T]()[])
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
struct _DynamicTupleIter[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
]:
    var idx: Int
    var src: DynamicTuple[T, D]

    @always_inline
    fn __next__(inout self) -> DynamicTuple[T, D]:
        self.idx += 1
        return self.src[self.idx - 1]

    @always_inline
    fn __len__(self) -> Int:
        return len(self.src) - self.idx


@value
struct DynamicTuple[T: CollectionElement, D: ElementDelegate = DefaultDelegate](
    CollectionElement, Sized, Stringable, EqualityComparable
):
    alias BaseType = DynamicTupleBase[T, D]
    alias Element = Self.BaseType.Element
    alias IterType = _DynamicTupleIter[T, D]

    var _value: Self.Element

    @always_inline
    fn __init__(inout self: Self):
        self._value = Self.BaseType()

    # FIXME: This constructor shouldn't be necessary
    @always_inline
    fn __init__(inout self: Self, value: T):
        self._value = value

    # FIXME: This constructor is never called
    @always_inline
    fn __init__(inout self: Self, value: Self.Element):
        self._value = value

    @always_inline
    fn __copyinit__(inout self: Self, value: Self):
        self._value = value._value

    @always_inline
    fn __moveinit__(inout self: Self, owned value: Self):
        self._value = value._value^

    # FIXME: We should have a single variadic constructor (https://github.com/modularml/modular/issues/32000)
    # @always_inline
    # fn __init__(inout self: Self, *values: Self):
    #     var value = Self.BaseType()
    #     value._elements.reserve(len(values))
    #     for e in values:
    #         value._elements.append(e[]._value)
    #     self._value = value

    @always_inline
    fn __init__(inout self: Self, v1: Self):
        self._value = Self.BaseType(v1._value)

    @always_inline
    fn __init__(inout self: Self, v1: Self, v2: Self):
        self._value = Self.BaseType(v1._value, v2._value)

    @always_inline
    fn __init__(inout self: Self, v1: Self, v2: Self, v3: Self):
        self._value = Self.BaseType(v1._value, v2._value, v3._value)

    @always_inline
    fn __init__(inout self: Self, v1: Self, v2: Self, v3: Self, v4: Self):
        self._value = Self.BaseType(v1._value, v2._value, v3._value, v4._value)

    @always_inline
    fn __init__(
        inout self: Self, v1: Self, v2: Self, v3: Self, v4: Self, v5: Self
    ):
        self._value = Self.BaseType(
            v1._value, v2._value, v3._value, v4._value, v5._value
        )

    fn __init__(inout self, zipper: _zip2[T, D]):
        self._value = Self.BaseType()
        for z in zipper:
            self.append(z)

    @always_inline
    fn __len__(self) -> Int:
        if self.is_value():
            return 1
        else:
            return len(self.tuple())

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self)

    @always_inline
    fn is_tuple(self) -> Bool:
        return self._value.isa[Self.BaseType]()

    @always_inline
    fn tuple(self) -> Self.BaseType:
        return self._value.get[Self.BaseType]()[]

    @always_inline
    fn is_value(self) -> Bool:
        return self._value.isa[T]()

    @always_inline
    fn value(self) -> T:
        return self._value.get[T]()[]

    @always_inline
    fn __getitem__(self, _idx: Int) -> Self:
        var idx = len(self) + _idx if _idx < 0 else _idx

        if self.is_value():
            if idx:
                return abort[Self]("Index should be 0 for value items.")
            return self.value()

        # FIXME: we should be able to return Self(self.tuple()[idx])
        var r = Self()
        r._value = self.tuple()[idx]
        return r

    @always_inline
    fn __getitem__(self, span: Slice) -> Self:
        if self.is_value():
            return abort[Self]("Can't slice a value.")

        var r = Self()
        r._value = self.tuple()[span]
        return r

    @always_inline
    fn __setitem__(inout self, _idx: Int, val: Self):
        var idx = len(self) + _idx if _idx < 0 else _idx

        if self.is_value() and val.is_value():
            if idx != 0:
                abort("Index should be 0 for value items.")

            self._value = val.value()
        else:
            var new_value: Self.BaseType = self.tuple()
            if val.is_value():
                new_value[idx] = val.value()
            else:
                new_value[idx] = val.tuple()
            self._value = new_value

    @always_inline
    fn append(inout self, owned v1: Self):
        var new_value: Self.BaseType
        if self.is_tuple():
            new_value = self.tuple()
        else:
            new_value = Self.BaseType(self.value())
        new_value.append(v1._value)
        self._value = new_value

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
        return Self.BaseType.to_string(self._value)


# ===----------------------------------------------------------------------===#
# zip
# ===----------------------------------------------------------------------===#


@value
struct _ZipIter2[T: CollectionElement, D: ElementDelegate = DefaultDelegate]:
    var index: Int
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]

    @always_inline
    fn __next__(inout self) -> DynamicTuple[T, D]:
        self.index += 1
        return DynamicTuple[T, D](
            self.a[self.index - 1], self.b[self.index - 1]
        )

    @always_inline
    fn __len__(self) -> Int:
        return math.min(len(self.a), len(self.b)) - self.index


@value
struct _zip2[T: CollectionElement, D: ElementDelegate = DefaultDelegate](Sized):
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]

    alias IterType = _ZipIter2[T, D]

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self.a, self.b)

    @always_inline
    fn __len__(self) -> Int:
        return math.min(len(self.a), len(self.b))


@value
struct _ZipIter3[T: CollectionElement, D: ElementDelegate = DefaultDelegate]:
    var index: Int
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]
    var c: DynamicTuple[T, D]

    @always_inline
    fn __next__(inout self) -> DynamicTuple[T, D]:
        self.index += 1
        return DynamicTuple[T, D](
            self.a[self.index - 1],
            self.b[self.index - 1],
            self.c[self.index - 1],
        )

    @always_inline
    fn __len__(self) -> Int:
        return (
            math.min(len(self.a), math.min(len(self.b), len(self.c)))
            - self.index
        )


@value
struct _zip3[T: CollectionElement, D: ElementDelegate = DefaultDelegate](Sized):
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]
    var c: DynamicTuple[T, D]

    alias IterType = _ZipIter3[T, D]

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self.a, self.b, self.c)

    @always_inline
    fn __len__(self) -> Int:
        return math.min(len(self.a), math.min(len(self.b), len(self.c)))


fn zip[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](a: DynamicTuple[T, D], b: DynamicTuple[T, D]) -> _zip2[T, D]:
    return _zip2(a, b)


fn zip[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](a: DynamicTuple[T, D], b: DynamicTuple[T, D], c: DynamicTuple[T, D]) -> _zip3[
    T, D
]:
    return _zip3(a, b, c)


# ===----------------------------------------------------------------------===#
# product
# ===----------------------------------------------------------------------===#


@value
struct _ProductIter2[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
]:
    var a_index: Int
    var b_index: Int
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]

    @always_inline
    fn __next__(inout self) -> DynamicTuple[T, D]:
        var res = DynamicTuple[T, D](self.a[self.a_index], self.b[self.b_index])
        self.b_index += 1
        if self.b_index == len(self.b):
            self.b_index = 0
            self.a_index += 1
        return res^

    @always_inline
    fn __len__(self) -> Int:
        var shortest = math.min(len(self.a), len(self.b))
        return shortest * shortest - self.a_index * len(self.b) - self.b_index


@value
struct _product2[T: CollectionElement, D: ElementDelegate = DefaultDelegate](
    Sized
):
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]

    alias IterType = _ProductIter2[T, D]

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, 0, self.a, self.b)

    @always_inline
    fn __len__(self) -> Int:
        var shortest = math.min(len(self.a), len(self.b))
        return shortest * shortest


fn product[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](a: DynamicTuple[T, D], b: DynamicTuple[T, D]) -> _product2[T, D]:
    return _product2(a, b)
