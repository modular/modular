# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, memcpy
from os import abort
from utils.variant import Variant
from testing import assert_equal, assert_false, assert_not_equal, assert_true
from builtin.range import _StridedRange
from collections import InlineArray
from utils import StaticTuple


struct CachedIntArray[cache_size: Int]:
    var _cache: StaticTuple[Int, cache_size]
    var _data: UnsafePointer[Int]
    var _size: Int

    @always_inline
    fn __init__(out self, size: Int = 0):
        self._cache = StaticTuple[Int, cache_size]()
        if size > cache_size:
            self._data = UnsafePointer[Int].alloc(size - cache_size)
        else:
            self._data = UnsafePointer[Int]()
        self._size = size

    @always_inline
    fn __init__(out self, *, non_owned: Self, offset: Int = 0):
        self._cache = StaticTuple[Int, cache_size]()
        var size = non_owned.size() - offset
        for i in range(min(size, cache_size)):
            self._cache[i] = non_owned[i + offset]
        self._data = non_owned._data + offset
        self._size = -(non_owned.size() - offset)

    @always_inline
    fn __copyinit__(out self, existing: Self):
        self._cache = StaticTuple[Int, cache_size]()
        var size = existing.size()
        self._size = existing._size
        if existing.owning():
            if size > cache_size:
                self._data = UnsafePointer[Int].alloc(size - cache_size)
            else:
                self._data = UnsafePointer[Int]()
            self.copy_from(0, existing, size)
        else:
            for i in range(min(size, cache_size)):
                self._cache[i] = existing._cache[i]
            self._data = existing._data

    @always_inline
    fn __moveinit__(out self, owned existing: Self):
        self._cache = existing._cache
        self._size = existing._size
        self._data = existing._data

    @always_inline
    fn __del__(owned self):
        if self.owning() and self._data:
            self._data.free()

    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        if idx < cache_size:
            return self._cache[idx]
        else:
            return self._data[idx - cache_size]

    @always_inline
    fn __setitem__(mut self, idx: Int, value: Int):
        if idx < cache_size:
            self._cache[idx] = value
        else:
            self._data[idx - cache_size] = value

    @always_inline
    fn owning(self) -> Bool:
        return self._size > 0

    @always_inline
    fn size(self) -> Int:
        return math.abs(self._size)

    @always_inline
    fn copy_from(mut self, offset: Int, source: Self, size: Int):
        # FIXME: Optimize by directly accessing _cache and _data
        for i in range(size):
            self[i + offset] = source[i]


alias IntArray = CachedIntArray[2]


alias UNKNOWN_VALUE = -1


@value
struct _IntTupleIter[origin: ImmutableOrigin]:
    var idx: Int
    var src: IntTuple[origin]

    @always_inline
    fn __next__(mut self) -> IntTuple[__origin_of(self.src)]:
        self.idx += 1
        return self.src[self.idx - 1]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        return len(self.src) - self.idx


alias IntOrTuple = Variant[Int, IntTuple]


struct IntTuple[origin: ImmutableOrigin = __origin_of()](
    CollectionElement,
    Sized,
    Stringable,
    Writable,
    EqualityComparable,
):
    var _store: IntArray
    """The underlying storage for the IntTuple. """
    """Int values are represented with positive numbers."""
    """Sub-tuples are represented with a negative offset from the current position."""

    alias IterType = _IntTupleIter

    alias MinimumValue = -0xFFFE

    @staticmethod
    @always_inline
    fn elements_size(elements: VariadicListMem[IntOrTuple]) -> Int:
        var size = 0
        for v in elements:
            if v[].isa[Int]():
                # The size of a single element
                size += 1
            else:
                # the size of the sub tuple plus the element
                size += v[][IntTuple].size() + 1
        return size

    @always_inline
    fn __init__(out self):
        self._store = IntArray(1)
        self._store[0] = 0
        self.validate_structure()

    @implicit
    @always_inline
    fn __init__(out self, *elements: Int):
        var size = len(elements) + 1
        self._store = IntArray(size)
        self._store[0] = len(elements)
        for i in range(len(elements)):
            var value = elements[i]
            if value < Self.MinimumValue:
                abort("Only integers greater than MinimumValue are supported")
            self._store[i + 1] = value

        self.validate_structure()

    @always_inline
    fn __init__(out self, *elements: IntOrTuple):
        var size = Self.elements_size(elements) + 1
        self._store = IntArray(size)
        self._store[0] = len(elements)
        var storage = self._store[0] + 1
        for i in range(len(elements)):
            storage = self.__insert(i, storage, elements[i])

        self.validate_structure()

    @always_inline
    fn __init__(out self, *, non_owned: IntArray):
        self._store = IntArray(non_owned=non_owned)

    @always_inline
    fn __init__(out self, existing: Self, rng: _StridedRange):
        var size = 0
        var len = 0
        for i in rng:
            size += existing[i].size() + 1
            len += 1
        size += 1

        self._store = IntArray(size)
        self._store[0] = len
        var storage = len + 1

        var pos = 0
        for i in rng:
            storage = self.__insert(pos, storage, existing[i])
            pos += 1

        self.validate_structure()

    @implicit
    @always_inline
    fn __init__(out self, zipper: _zip[_, 2]):
        # FIXME: massively inefficient
        self = Self()
        for z in zipper:
            self.append(z)

    # Mojo BUG: __copyinit__ unnecessarily propagates self's origin to the new copy
    @always_inline
    fn __copyinit__(out self, existing: Self):
        var size = existing.size()
        self._store = IntArray(size)
        self._store.copy_from(0, existing._store, size)

    @always_inline
    fn __moveinit__(out self, owned existing: Self):
        self._store = existing._store^

    @always_inline
    fn owned_copy(self) -> IntTuple:
        var copy = IntTuple(non_owned=IntArray())
        var size = self.size()
        copy._store = IntArray(size)
        copy._store.copy_from(0, self._store, size)
        return copy

    # FIXME: this needs a better name and optimization
    @always_inline
    fn replace_entry(self, idx: Int, value: IntTuple) -> IntTuple:
        var result = IntTuple()
        for i in range(len(self)):
            if i != idx:
                result.append(self[i])
            else:
                result.append(value)
        return result

    @always_inline
    fn append(mut self, *elements: IntOrTuple):
        if not self._store.owning():
            abort("Can't modify a sub-tuple.")

        if len(elements) == 0:
            return

        var old_len = len(self)
        var old_size = self.size()
        var old_data = self._store
        var old_tuple = IntTuple[__origin_of(old_data)](non_owned=old_data)
        var new_size = old_size + Self.elements_size(elements)
        self._store = IntArray(new_size)
        var new_len = old_len + len(elements)
        self._store[0] = new_len
        var storage = new_len + 1

        for i in range(old_len):
            if old_tuple.is_value(i):
                self._store[i + 1] = old_tuple.value(i)
            else:
                storage = self.__insert(i, storage, old_tuple[i])

        for i in range(len(elements)):
            if elements[i].isa[Int]():
                self._store[old_len + i + 1] = elements[i][Int]
            else:
                storage = self.__insert(i + old_len, storage, elements[i])

        self.validate_structure()

    @always_inline
    fn extend(mut self, tuple: IntTuple):
        if not self._store.owning():
            abort("Can't modify a sub-tuple.")

        if len(tuple) == 0:
            return

        var old_len = len(self)
        var old_size = self.size()
        var old_data = self._store
        var old_tuple = IntTuple[__origin_of(old_data)](non_owned=old_data)
        var new_size = old_size + tuple.size() - 1  # FIXME: Yuck
        self._store = IntArray(new_size)
        var new_len = old_len + len(tuple)
        self._store[0] = new_len
        var storage = new_len + 1

        for i in range(old_len):
            if old_tuple.is_value(i):
                self._store[i + 1] = old_tuple.value(i)
            else:
                storage = self.__insert(i, storage, old_tuple[i])

        for i in range(len(tuple)):
            if tuple.is_value(i):
                self._store[old_len + i + 1] = tuple.value(i)
            else:
                storage = self.__insert(i + old_len, storage, tuple[i])

        self.validate_structure()

    @always_inline
    fn __insert(mut self, idx: Int, storage: Int, element: IntOrTuple) -> Int:
        if element.isa[Int]():
            var value = element[Int]
            if value < Self.MinimumValue:
                abort("Only integers greater than MinimumValue are supported")
            self._store[idx + 1] = value
            return storage
        else:
            return self.__insert(idx, storage, element[IntTuple])

    @always_inline
    fn __insert(mut self, idx: Int, storage: Int, element: IntTuple) -> Int:
        # Negative offset from current position.
        self._store[idx + 1] = idx + 1 - storage + Self.MinimumValue
        var size = element.size()
        self._store.copy_from(storage, element._store, size)
        return storage + size

    @always_inline
    fn size(self) -> Int:
        if self._store.owning():
            return self._store.size()
        return Self.tuple_size(self._store)

    @staticmethod
    fn tuple_size(data: IntArray) -> Int:
        var len = data[0]
        var size = 1
        for i in range(len):
            var val = data[i + 1]
            if val >= Self.MinimumValue:
                size += 1
            else:
                var sub_data = IntArray(
                    non_owned=data, offset=i + 1 - (val - Self.MinimumValue)
                )
                size += Self.tuple_size(sub_data) + 1
        return size

    @always_inline
    fn validate_structure(self):
        alias DEBUG_STRUCTURE = False

        @parameter
        if DEBUG_STRUCTURE:
            if self._store.owning() > 0:
                var data_size = self._store.size()
                var computed_size = Self.tuple_size(self._store)
                if data_size != computed_size:
                    abort(
                        "size validation failed: "
                        + str(data_size)
                        + " != "
                        + str(computed_size)
                    )

    @always_inline
    fn __len__(self) -> Int:
        return self._store[0]

    @always_inline
    fn __iter__(self) -> Self.IterType[__origin_of(self)]:
        return Self.IterType(
            0, IntTuple[__origin_of(self)](non_owned=self._store)
        )

    @always_inline
    fn __getitem__(self, _idx: Int) -> IntTuple[__origin_of(self)]:
        var idx = len(self) + _idx if _idx < 0 else _idx

        if idx < 0 or idx >= len(self):
            abort("Index out of bounds.")

        # The int value or the (negated) offset to the tuple
        var val = self._store[idx + 1]
        if val >= Self.MinimumValue:
            # Return the Int value
            return IntTuple[__origin_of(self)](val)
        else:
            # Return the sub-tuple
            return IntTuple[__origin_of(self)](
                non_owned=IntArray(
                    non_owned=self._store,
                    offset=idx + 1 - (val - Self.MinimumValue),
                )
            )

    @always_inline
    fn __getitem__(self, span: Slice) -> Self:
        var start: Int
        var end: Int
        var step: Int
        start, end, step = span.indices(len(self))
        return Self(self, range(start, end, step))

    @always_inline
    fn is_value(self) -> Bool:
        return len(self) == 1 and self._store[1] >= Self.MinimumValue

    @always_inline
    fn is_tuple(self) -> Bool:
        return not self.is_value()

    @always_inline
    fn value(self) -> Int:
        return self._store[1]

    @always_inline
    fn is_value(self, i: Int) -> Bool:
        if i < 0 or i >= len(self):
            abort("Index out of bounds.")

        return self._store[i + 1] >= Self.MinimumValue

    @always_inline
    fn is_tuple(self, i: Int) -> Bool:
        return not self.is_value(i)

    @always_inline
    fn value(self, i: Int) -> Int:
        return self._store[i + 1]

    @always_inline
    fn tuple(ref self) -> ref [self] Self:
        # Avoid making gratuitous copies
        return self

    fn __str__(self) -> String:
        if self.is_value():
            return str(self.value())
        var s = String("(")
        var len = len(self)
        for i in range(len):
            if self.is_value(i):
                s += str(self.value(i))
            else:
                s += str(self[i])
            if i < len - 1:
                s += ", "
        return s + ")"

    @staticmethod
    fn is_equal(a: IntTuple, b: IntTuple) -> Bool:
        if len(a) == len(b):
            for i in range(len(a)):
                if a.is_value(i) and b.is_value(i):
                    if a.value(i) != b.value(i):
                        return False
                if a.is_tuple(i) and b.is_tuple(i):
                    if not Self.is_equal(a[i], b[i]):
                        return False
            return True
        return False

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return Self.is_equal(self, other)

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return not Self.is_equal(self, other)

    @always_inline
    fn __repr__(self) -> String:
        return self.__str__()

    @always_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())


@always_inline
fn signum(a: Int) -> Int:
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


@always_inline
fn to_int(v: IntTuple) -> Int:
    return v.value()


@always_inline
fn is_int(t: IntTuple) -> Bool:
    return t.is_value()


@always_inline
fn is_tuple(t: IntTuple) -> Bool:
    return t.is_tuple()


@value
struct _ZipIter[origin: ImmutableOrigin, n: Int]:
    var index: Int
    var ts: InlineArray[IntTuple[origin], n]

    @always_inline
    fn __next__(mut self) -> IntTuple[origin]:
        self.index += 1

        @parameter
        if n == 2:
            return IntTuple[origin](
                self.ts[0][self.index - 1], self.ts[1][self.index - 1]
            )
        elif n == 3:
            return IntTuple[origin](
                self.ts[0][self.index - 1],
                self.ts[1][self.index - 1],
                self.ts[2][self.index - 1],
            )
        else:
            abort("Only zip[2] or zip[3] are supported.")

            var result = IntTuple[origin](self.ts[0][self.index - 1])
            for i in range(1, n):
                result.append(self.ts[i][self.index - 1])
            return result

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        var min_len = len(self.ts[0])

        @parameter
        for i in range(1, n):
            min_len = min(min_len, len(self.ts[i]))
        return min_len - self.index


@value
struct _zip[origin: ImmutableOrigin, n: Int]:
    var ts: InlineArray[IntTuple[origin], n]

    alias IterType = _ZipIter[origin, n]

    @always_inline
    fn __iter__(self) -> Self.IterType:
        return Self.IterType(0, self.ts)

    @always_inline
    fn __len__(self) -> Int:
        var min_len = len(self.ts[0])

        @parameter
        for i in range(1, n):
            min_len = min(min_len, len(self.ts[i]))
        return min_len


@always_inline
fn zip[
    origin: ImmutableOrigin, n: Int
](ts: InlineArray[IntTuple[origin], n]) -> _zip[origin, n]:
    return _zip[origin, n](ts)


@always_inline
fn zip[
    a_origin: ImmutableOrigin, b_origin: ImmutableOrigin
](
    out result: _zip[__origin_of(a_origin, b_origin), 2],
    a: IntTuple[a_origin],
    b: IntTuple[b_origin],
):
    alias common_type = IntTuple[__origin_of(a_origin, b_origin)]
    return __type_of(result)(
        InlineArray[common_type, 2](
            rebind[common_type](IntTuple[a_origin](non_owned=a._store)),
            rebind[common_type](IntTuple[b_origin](non_owned=b._store)),
        )
    )


@always_inline
fn zip[
    a_origin: ImmutableOrigin,
    b_origin: ImmutableOrigin,
    c_origin: ImmutableOrigin,
](
    out result: _zip[__origin_of(a_origin, b_origin, c_origin), 3],
    a: IntTuple[a_origin],
    b: IntTuple[b_origin],
    c: IntTuple[c_origin],
):
    alias common_type = IntTuple[__origin_of(a_origin, b_origin, c_origin)]
    return __type_of(result)(
        InlineArray[common_type, 3](
            rebind[common_type](IntTuple[a_origin](non_owned=a._store)),
            rebind[common_type](IntTuple[b_origin](non_owned=b._store)),
            rebind[common_type](IntTuple[c_origin](non_owned=c._store)),
        )
    )


# Python-style reduce functions


fn reduce[
    T: AnyTrivialRegType, func: fn (a: T, b: IntTuple) capturing [_] -> T
](t: IntTuple, initializer: T) -> T:
    var result: T = initializer
    for e in t:
        result = func(result, e)
    return result


# fn reduce[
#     T: CollectionElement, func: fn (owned a: T, b: IntTuple) capturing [_] -> T
# ](t: IntTuple, initializer: T) -> T:
#     var result: T = initializer
#     for e in t:
#         result = func(result, e)
#     return result


# TODO: This used to be more generic, see above
fn reduce[
    reducer: fn (a: IntTuple, b: IntTuple) capturing [_] -> IntTuple,
](t: IntTuple) -> IntTuple:
    var result = IntTuple()
    for e in t:
        result = reducer(result, e)
    return result


# IntTuple operations


@always_inline
fn flatten(t: IntTuple) -> IntTuple:
    @always_inline
    @parameter
    fn reducer(a: IntTuple, b: IntTuple) -> IntTuple:
        var r = a.owned_copy()  # Avoid propagating a's origin
        if b.is_value():
            r.append(b)
        else:
            for e in flatten(b):
                r.append(e)
        return r

    return reduce[reducer](t)


# Create a IntTuple with same structure but filled by UNKNOWN_VALUE.
@always_inline
fn to_unknown(t: IntTuple) -> IntTuple:
    if t.is_value():
        return UNKNOWN_VALUE

    @always_inline
    @parameter
    fn reducer(a: IntTuple, b: IntTuple) -> IntTuple:
        var r = a.owned_copy()  # Avoid propagating a's origin
        if b.is_value():
            r.append(UNKNOWN_VALUE)
        else:
            r.append(to_unknown(b))
        return r

    return reduce[reducer](t)


@always_inline
fn lt(a: IntTuple, b: IntTuple) -> Bool:
    for z in zip(a, b):
        if to_int(z[0]) == to_int(z[1]):
            continue
        elif to_int(z[0]) < to_int(z[1]):
            return True
        else:
            return False
    return False


@always_inline
fn _merge[
    cmp: fn (IntTuple, IntTuple) -> Bool,
](left: IntTuple, right: IntTuple) -> IntTuple:
    var result = IntTuple()
    var i = 0
    var j = 0

    while i < len(left) and j < len(right):
        if cmp(left[i], right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


fn sorted[
    cmp: fn (IntTuple, IntTuple) -> Bool = lt,
](tuple: IntTuple) -> IntTuple:
    if len(tuple) <= 1:
        return tuple.owned_copy()  # Avoid propagating a's origin
    var mid = len(tuple) // 2
    return _merge[cmp](sorted[cmp](tuple[:mid]), sorted[cmp](tuple[mid:]))


@always_inline
fn sum(t: IntTuple) -> Int:
    @always_inline
    @parameter
    fn reducer(a: Int, b: IntTuple) -> Int:
        return UNKNOWN_VALUE if a == UNKNOWN_VALUE else a + (
            to_int(b) if is_int(b) else sum(b)
        )

    return reduce[Int, reducer](t, 0)


@always_inline
fn product(t: IntTuple) -> Int:
    @always_inline
    @parameter
    fn reducer(a: Int, b: IntTuple) -> Int:
        return UNKNOWN_VALUE if a == UNKNOWN_VALUE else a * (
            to_int(b) if is_int(b) else product(b)
        )

    return reduce[Int, reducer](t, 1)


# TODO: Can't call this `max` otherwise the compiler incorrectly
# fails to recurse when calling this local function.
@always_inline
fn tuple_max(t: IntTuple) -> Int:
    @always_inline
    @parameter
    fn reducer(a: Int, b: IntTuple) -> Int:
        return max(a, to_int(b) if is_int(b) else tuple_max(b))

    alias int_min_val = 0
    return reduce[Int, reducer](t, int_min_val)


fn apply[func: fn (Int) capturing [_] -> Int](t: IntTuple) -> IntTuple:
    if is_int(t):
        return func(to_int(t))
    var res = IntTuple()
    for e in t:
        res.append(apply[func](e))
    return res


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


fn apply_zip[
    func: fn (IntTuple, IntTuple, IntTuple) -> IntTuple
](t1: IntTuple, t2: IntTuple, t3: IntTuple) -> IntTuple:
    var r = IntTuple()
    for z in zip(t1, t2, t3):
        r.append(func(z[0], z[1], z[2]))
    return r


fn apply_zip[
    func: fn (IntTuple, IntTuple, IntTuple) capturing [_] -> IntTuple
](t1: IntTuple, t2: IntTuple, t3: IntTuple) -> IntTuple:
    var r = IntTuple()
    for z in zip(t1, t2, t3):
        r.append(func(z[0], z[1], z[2]))
    return r


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


# Multiply lhs tuple elements by rhs
#
fn mul(lhs: IntTuple, rhs: Int) -> IntTuple:
    if is_int(lhs):
        if UNKNOWN_VALUE in (to_int(lhs), rhs):
            return UNKNOWN_VALUE
        return lhs.value() * rhs

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


fn prefix_product(a: IntTuple) -> IntTuple:
    return prefix_product(a, IntTuple(1))


fn prefix_product(a: IntTuple, init: IntTuple = 1) -> IntTuple:
    return prefix_product2(a, init)


fn prefix_product2(a: IntTuple, init: IntTuple) -> IntTuple:
    if is_tuple(a):
        if is_tuple(init):  # tuple tuple
            if len(a) != len(init):
                abort("len(a) != len(init)")
            return apply_zip[prefix_product2](a, init)
        else:  # tuple "int"
            var v_init = to_int(init)
            var r = IntTuple()
            for v in a:
                r.append(prefix_product2(v, v_init))
                v_init = (
                    UNKNOWN_VALUE if v_init
                    == UNKNOWN_VALUE else v_init * product(v)
                )
            return r
    else:
        if is_tuple(init):  # "int" tuple
            abort("'int' tuple not allowed")  # Error
            return IntTuple()
        else:  # "int" "int"
            return init.owned_copy()


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
fn idx2crd(idx: IntTuple, shape: IntTuple) -> IntTuple:
    return idx2crd2(idx, shape, IntTuple())


@always_inline
fn idx2crd(idx: IntTuple, shape: IntTuple, _stride: IntTuple) -> IntTuple:
    return idx2crd2(idx, shape, _stride)


@always_inline
fn idx2crd2(
    idx: IntTuple,
    shape: IntTuple,
    _stride: IntTuple,  # = IntTuple()
) -> IntTuple:
    var stride: IntTuple
    if len(_stride) == 0:
        stride = prefix_product(shape).owned_copy()
    else:
        stride = _stride.owned_copy()

    if is_tuple(idx):
        if is_tuple(shape):  # tuple tuple tuple
            if len(idx) != len(shape) or len(idx) != len(stride):
                abort("input shapes mismatch")

            return apply_zip[idx2crd2](idx, shape, stride)
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


fn crd2idx(crd: IntTuple, shape: IntTuple) -> Int:
    return crd2idx(crd, shape, IntTuple())


fn crd2idx(
    crd: IntTuple,
    shape: IntTuple,
    _stride: IntTuple,  # = IntTuple()
) -> Int:
    var stride: IntTuple
    if len(_stride) == 0:
        stride = prefix_product(shape).owned_copy()
    else:
        stride = _stride.owned_copy()

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
            if len(shape) == 0:
                return 0
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
        return target.owned_copy()
    return src.owned_copy()


# Returns an IntTuple reversed e.g reverse(1, 2, (3, 4)) returns ((4, 3), 2, 1)
#
fn reverse(src: IntTuple) -> IntTuple:
    var res = IntTuple()
    if is_int(src):
        return src.owned_copy()
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
