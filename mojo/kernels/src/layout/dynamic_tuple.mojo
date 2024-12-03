# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math
from os import abort

from algorithm.functional import _get_start_indices_of_nth_subvolume

from utils import Index, Writer
from utils.variant import Variant


# FIXME: This is a horrible hack around Mojo's lack or proper trait inheritance
trait ElementDelegate:
    @staticmethod
    fn is_equal[T: CollectionElement](a: Variant[T], b: Variant[T]) -> Bool:
        pass

    @staticmethod
    fn format_element_to[
        T: CollectionElement, W: Writer
    ](mut writer: W, a: Variant[T]):
        pass


struct DefaultDelegate(ElementDelegate):
    @staticmethod
    fn is_equal[T: CollectionElement](a: Variant[T], b: Variant[T]) -> Bool:
        return False

    @staticmethod
    fn format_element_to[
        T: CollectionElement, W: Writer
    ](mut writer: W, a: Variant[T]):
        writer.write("#")


struct DynamicTupleBase[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](CollectionElement, Sized, Stringable, Writable, EqualityComparable):
    alias Element = Variant[T, Self]

    var _elements: List[Self.Element]

    @always_inline
    fn __init__(out self):
        self._elements = List[Self.Element]()

    @always_inline
    fn __init__(out self, *v: Self.Element):
        self._elements = List[Self.Element](capacity=len(v))
        for e in v:
            self._elements.append(e[])

    @always_inline
    fn __moveinit__(out self, owned existing: Self):
        self._elements = existing._elements^

    @always_inline
    fn __copyinit__(out self, existing: Self):
        self._elements = existing._elements

    @always_inline
    fn append(mut self, value: Self.Element):
        self._elements.append(value)

    @always_inline
    fn __getitem__(self, _idx: Int) -> Self.Element:
        var idx = len(self) + _idx if _idx < 0 else _idx

        if idx < 0 or idx >= len(self._elements):
            abort("Index out of bounds.")
        return self._elements[idx]

    @always_inline
    fn __getitem__(self, owned span: Slice) -> Self:
        var start: Int
        var end: Int
        var step: Int
        start, end, step = span.indices(len(self))

        var r = range(start, end, step)
        var result = Self()
        result._elements.reserve(len(r))
        for i in r:
            result._elements.append(self[i])
        return result

    @always_inline
    fn __setitem__(mut self, _idx: Int, val: Self.Element):
        var idx = len(self) + _idx if _idx < 0 else _idx

        if idx < 0 or idx >= len(self._elements):
            abort("Index out of bounds.")
        self._elements[idx] = val

    @always_inline
    fn __len__(self) -> Int:
        return len(self._elements)

    @staticmethod
    fn format_element_to[W: Writer, //](mut writer: W, v: Self.Element):
        if v.isa[T]():
            return D.format_element_to[T](writer, v[T])
        else:
            writer.write("(")
            if v.isa[Self]():
                var _elements = v[Self]._elements
                for i in range(len(_elements)):
                    var e: Self.Element = _elements[i]
                    Self.format_element_to(writer, e)
                    if i < len(_elements) - 1:
                        writer.write(", ")
            writer.write(")")

    @staticmethod
    fn is_equal(a: Self.Element, b: Self.Element) -> Bool:
        if a.isa[T]() and b.isa[T]():
            return D.is_equal[T](a[T], b[T])
        if a.isa[Self]() and b.isa[Self]():
            var ta = a[Self]
            var tb = b[Self]
            if len(ta) == len(tb):
                for i in range(len(ta)):
                    if not Self.is_equal(ta[i], tb[i]):
                        return False
                return True
        return False

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer, //](self, mut writer: W):
        Self.format_element_to(writer, self)

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
    fn __next__(mut self) -> DynamicTuple[T, D]:
        self.idx += 1
        return self.src[self.idx - 1]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

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
    fn __init__(out self):
        self._value = Self.BaseType()

    # FIXME: This constructor shouldn't be necessary
    @always_inline
    @implicit
    fn __init__(out self: Self, value: T):
        self._value = value

    # FIXME: This constructor is never called
    @always_inline
    @implicit
    fn __init__(out self: Self, value: Self.Element):
        self._value = value

    @always_inline
    fn __copyinit__(out self, value: Self):
        self._value = value._value

    @always_inline
    fn __moveinit__(out self, owned value: Self):
        self._value = value._value^

    # FIXME: We should have a single variadic constructor (https://github.com/modularml/modular/issues/32000)
    # @always_inline
    # fn __init__(out self, *values: Self):
    #     var value = Self.BaseType()
    #     value._elements.reserve(len(values))
    #     for e in values:
    #         value._elements.append(e[]._value)
    #     self._value = value

    @always_inline
    fn __init__(out self, v1: Self):
        self._value = Self.BaseType(v1._value)

    @always_inline
    fn __init__(out self, v1: Self, v2: Self):
        self._value = Self.BaseType(v1._value, v2._value)

    @always_inline
    fn __init__(out self, v1: Self, v2: Self, v3: Self):
        self._value = Self.BaseType(v1._value, v2._value, v3._value)

    @always_inline
    fn __init__(out self, v1: Self, v2: Self, v3: Self, v4: Self):
        self._value = Self.BaseType(v1._value, v2._value, v3._value, v4._value)

    @always_inline
    fn __init__(mut self, v1: Self, v2: Self, v3: Self, v4: Self, v5: Self):
        self._value = Self.BaseType(
            v1._value, v2._value, v3._value, v4._value, v5._value
        )

    @implicit
    fn __init__(out self, zipper: _zip2[T, D]):
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
        return self._value[Self.BaseType]

    @always_inline
    fn is_value(self) -> Bool:
        return self._value.isa[T]()

    @always_inline
    fn value(self) -> T:
        return self._value[T]

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
    fn __setitem__(mut self, _idx: Int, val: Self):
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
    fn append(mut self, owned v1: Self):
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
        return String.write(self)

    fn write_to[W: Writer, //](self, mut writer: W):
        Self.BaseType.format_element_to(writer, self._value)


# ===-----------------------------------------------------------------------===#
# zip
# ===-----------------------------------------------------------------------===#


@value
struct _ZipIter2[T: CollectionElement, D: ElementDelegate = DefaultDelegate]:
    var index: Int
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]

    @always_inline
    fn __next__(mut self) -> DynamicTuple[T, D]:
        self.index += 1
        return DynamicTuple[T, D](
            self.a[self.index - 1], self.b[self.index - 1]
        )

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        return min(len(self.a), len(self.b)) - self.index


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
        return min(len(self.a), len(self.b))


@value
struct _ZipIter3[T: CollectionElement, D: ElementDelegate = DefaultDelegate]:
    var index: Int
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]
    var c: DynamicTuple[T, D]

    @always_inline
    fn __next__(mut self) -> DynamicTuple[T, D]:
        self.index += 1
        return DynamicTuple[T, D](
            self.a[self.index - 1],
            self.b[self.index - 1],
            self.c[self.index - 1],
        )

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        return min(len(self.a), min(len(self.b), len(self.c))) - self.index


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
        return min(len(self.a), min(len(self.b), len(self.c)))


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


# ===-----------------------------------------------------------------------===#
# product
# ===-----------------------------------------------------------------------===#


@value
struct _ProductIter2[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
]:
    var a_index: Int
    var b_index: Int
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]

    @always_inline
    fn __next__(mut self) -> DynamicTuple[T, D]:
        var res = DynamicTuple[T, D](self.a[self.a_index], self.b[self.b_index])
        self.b_index += 1
        if self.b_index == len(self.b):
            self.b_index = 0
            self.a_index += 1
        return res^

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        var len_a = len(self.a)
        var len_b = len(self.b)
        return len_a * len_b - self.a_index * len_b - self.b_index


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
        var len_a = len(self.a)
        var len_b = len(self.b)
        return len_a * len_b

    @always_inline
    fn __getitem__(self, idx: Int) -> DynamicTuple[T, D]:
        var len_b = len(self.b)

        var idx_b = idx._positive_rem(len_b)
        var idx_a = idx._positive_div(len_b)

        return DynamicTuple[T, D](self.a[idx_a], self.b[idx_b])


@value
struct _ProductIter3[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
]:
    var offset: Int
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]
    var c: DynamicTuple[T, D]

    @always_inline
    fn __next__(mut self) -> DynamicTuple[T, D]:
        self.offset += 1
        var idx = _get_start_indices_of_nth_subvolume[0](
            self.offset - 1, Index(len(self.a), len(self.b), len(self.c))
        )
        return DynamicTuple[T, D](
            self.a[idx[0]], self.b[idx[1]], self.c[idx[2]]
        )

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        return len(self.a) * len(self.b) * len(self.c) - self.offset


@value
struct _product3[T: CollectionElement, D: ElementDelegate = DefaultDelegate](
    Sized
):
    var a: DynamicTuple[T, D]
    var b: DynamicTuple[T, D]
    var c: DynamicTuple[T, D]

    alias IterType = _ProductIter3[T, D]

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self.a, self.b, self.c)

    @always_inline
    fn __len__(self) -> Int:
        return len(self.a) * len(self.b) * len(self.c)

    @always_inline
    fn __getitem__(self, offset: Int) -> DynamicTuple[T, D]:
        var idx = _get_start_indices_of_nth_subvolume[0](
            offset, Index(len(self.a), len(self.b), len(self.c))
        )

        return DynamicTuple[T, D](
            self.a[idx[0]], self.b[idx[1]], self.c[idx[2]]
        )


@always_inline
fn _lift(n: Int, shape: List[Int, *_]) -> List[Int, hint_trivial_type=True]:
    """Lifts the linearized shape to the ND shape. This is the same as
    _get_start_indices_of_nth_subvolume[N, 0] but works in the runtime
    domain."""
    var out = List[Int, hint_trivial_type=True](capacity=len(shape))
    var curr_index = n

    for i in reversed(range(len(shape))):
        out.append(curr_index._positive_rem(shape[i]))
        curr_index = curr_index._positive_div(shape[i])
    out.reverse()
    return out


@always_inline
fn _get_shapes[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](tuples: List[DynamicTuple[T, D]]) -> List[Int, hint_trivial_type=True]:
    var tuples_shapes = List[Int, hint_trivial_type=True](capacity=len(tuples))
    for tup in tuples:
        tuples_shapes.append(len(tup[]))
    return tuples_shapes


struct _ProductIterN[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
]:
    var offset: Int
    var tuples: List[DynamicTuple[T, D]]
    var tuples_shape: List[Int, hint_trivial_type=True]

    @implicit
    fn __init__(out self, tuples: List[DynamicTuple[T, D]]):
        self.offset = 0
        self.tuples = tuples
        self.tuples_shape = _get_shapes(tuples)

    @always_inline
    fn __next__(mut self) -> DynamicTuple[T, D]:
        self.offset += 1
        var idx = _lift(self.offset - 1, self.tuples_shape)
        var res = DynamicTuple[T, D]()
        for i in range(len(self.tuples_shape)):
            res.append(self.tuples[i][idx[i]])
        return res

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        var total = 1
        for i in self.tuples_shape:
            total *= i[]
        return total - self.offset


struct _productN[T: CollectionElement, D: ElementDelegate = DefaultDelegate](
    Sized
):
    var tuples: List[DynamicTuple[T, D]]

    alias IterType = _ProductIterN[T, D]

    @implicit
    fn __init__(out self, *tuples: DynamicTuple[T, D]):
        self.tuples = List[DynamicTuple[T, D]](capacity=len(tuples))
        for tup in tuples:
            self.tuples.append(tup[])

    @implicit
    fn __init__(out self, tuples: List[DynamicTuple[T, D]]):
        self.tuples = tuples

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(self.tuples)

    @always_inline
    fn __len__(self) -> Int:
        var total = 1
        for tup in self.tuples:
            total *= len(tup[])
        return total

    @always_inline
    fn __getitem__(self, offset: Int) -> DynamicTuple[T, D]:
        var tuples_shape = _get_shapes(self.tuples)
        var idx = _lift(offset, tuples_shape)
        var res = DynamicTuple[T, D]()
        for i in range(len(tuples_shape)):
            res.append(self.tuples[i][idx[i]])
        return res


fn product[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](a: DynamicTuple[T, D], b: DynamicTuple[T, D]) -> _product2[T, D]:
    return _product2(a, b)


fn product[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](
    a: DynamicTuple[T, D], b: DynamicTuple[T, D], c: DynamicTuple[T, D]
) -> _product3[T, D]:
    return _product3(a, b, c)


fn product[
    T: CollectionElement, D: ElementDelegate = DefaultDelegate
](*tuples: DynamicTuple[T, D]) -> _productN[T, D]:
    var lst = List[DynamicTuple[T, D]](capacity=len(tuples))
    for tup in tuples:
        lst.append(tup[])
    return _productN(lst^)
