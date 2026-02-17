# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Defines `Dict`, a collection that stores key-value pairs.

Dict provides an efficient, O(1) amortized
average-time complexity for insert, lookup, and removal of dictionary elements.
It uses a Swiss Table implementation with SIMD group probing for fast lookups:

- Performance and size are heavily optimized for small dictionaries, but can
  scale to large dictionaries.

- Insertion order is implicitly preserved. Iteration over keys, values, and
  items have a deterministic order based on insertion.

- For more information on the Mojo `Dict` type, see the
  [Mojo `Dict` manual](/mojo/manual/types/#dict). To learn more about using
  Python dictionaries from Mojo, see
  [Python types in Mojo](/mojo/manual/python/types/#python-types-in-mojo).

Key elements must implement the `KeyElement` trait composition, which includes
`Hashable`, `Equatable`, and `Copyable`. The `Copyable`
requirement will eventually be removed.

Value elements must be `Copyable`. As with `KeyElement`, the
`Copyable` requirement for value elements will eventually be removed.

See the `Dict` docs for more details.
"""

from builtin.constrained import _constrained_conforms_to
from compile import get_type_name
from hashlib import Hasher, default_comp_time_hasher, default_hasher
from sys.intrinsics import is_compile_time, likely

from bit import count_trailing_zeros, next_power_of_two
from memory import alloc, bitcast, memcpy, memset, pack_bits

comptime KeyElement = Copyable & Hashable & Equatable
"""A trait composition for types which implement all requirements of
dictionary keys. Dict keys must minimally be `Copyable`, `Hashable`,
and `Equatable`."""

# ===-----------------------------------------------------------------------===#
# Swiss Table constants and helpers
# ===-----------------------------------------------------------------------===#

comptime _CTRL_EMPTY: UInt8 = 0xFF
"""Control byte for an empty slot."""
comptime _CTRL_DELETED: UInt8 = 0x80
"""Control byte for a deleted (tombstone) slot."""
comptime _GROUP_WIDTH: Int = 16
"""Number of control bytes processed in one SIMD operation."""
comptime _INITIAL_CAPACITY: Int = 16
"""Minimum table capacity. Must be >= _GROUP_WIDTH."""


@always_inline
fn _h2(hash: UInt64) -> UInt8:
    """Extract the top 7 bits of the hash as a fingerprint (0x00-0x7F)."""
    return UInt8(hash >> 57)


@always_inline
fn _is_occupied(ctrl: UInt8) -> Bool:
    """Check if a control byte represents an occupied slot.

    Occupied slots have h2 values in range 0x00-0x7F (top bit clear).
    DELETED (0x80) and EMPTY (0xFF) both have top bit set.
    """
    return ctrl < _CTRL_DELETED


# ===-----------------------------------------------------------------------===#
# _Group: SIMD group operations on 16 control bytes
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Group(Copyable, Movable):
    """A group of control bytes for SIMD probing.

    Loads 16 control bytes at once and performs parallel matching using
    SIMD comparison operations, enabling fast hash table lookups.
    """

    var ctrl: SIMD[DType.uint8, _GROUP_WIDTH]

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[UInt8]):
        """Load a group of control bytes from memory.

        Args:
            ptr: Pointer to the start of 16 consecutive control bytes.
        """
        self.ctrl = ptr.load[width=_GROUP_WIDTH]()

    # TODO: Remove `is_compile_time()` branches once `pack_bits` is supported
    # by the compile-time interpreter. Currently `pack_bits` uses `pop.bitcast`
    # which the interpreter can't handle, so we fall back to scalar loops for
    # comptime contexts (e.g., Dict used in `comptime` expressions).

    @always_inline
    fn match_h2(self, h2: UInt8) -> UInt16:
        """Return a bitmask of slots matching the given h2 fingerprint.

        Args:
            h2: The h2 fingerprint to match (0x00-0x7F).

        Returns:
            A bitmask where bit i is set if ctrl[i] == h2.
        """
        if is_compile_time():
            return Self._scalar_match(self.ctrl, h2)
        return pack_bits(self.ctrl.eq(SIMD[DType.uint8, _GROUP_WIDTH](h2)))

    @always_inline
    fn match_empty(self) -> UInt16:
        """Return a bitmask of empty slots.

        Returns:
            A bitmask where bit i is set if ctrl[i] == EMPTY (0xFF).
        """
        if is_compile_time():
            return Self._scalar_match(self.ctrl, _CTRL_EMPTY)
        return pack_bits(
            self.ctrl.eq(SIMD[DType.uint8, _GROUP_WIDTH](_CTRL_EMPTY))
        )

    @always_inline
    fn match_empty_or_deleted(self) -> UInt16:
        """Return a bitmask of empty or deleted slots.

        Both EMPTY (0xFF) and DELETED (0x80) have the top bit set,
        so they are >= 0x80. All occupied h2 values are 0x00-0x7F.

        Returns:
            A bitmask where bit i is set if ctrl[i] is EMPTY or DELETED.
        """
        if is_compile_time():
            var result = UInt16(0)

            @parameter
            for i in range(_GROUP_WIDTH):
                if self.ctrl[i] >= _CTRL_DELETED:
                    result |= UInt16(1) << UInt16(i)
            return result
        return pack_bits(
            self.ctrl.ge(SIMD[DType.uint8, _GROUP_WIDTH](_CTRL_DELETED))
        )

    @staticmethod
    @always_inline
    fn _scalar_match(
        ctrl: SIMD[DType.uint8, _GROUP_WIDTH], target: UInt8
    ) -> UInt16:
        """Scalar fallback for compile-time evaluation.

        Args:
            ctrl: The control byte vector.
            target: The byte value to match.

        Returns:
            A bitmask where bit i is set if ctrl[i] == target.
        """
        var result = UInt16(0)

        @parameter
        for i in range(_GROUP_WIDTH):
            if ctrl[i] == target:
                result |= UInt16(1) << UInt16(i)
        return result


# ===-----------------------------------------------------------------------===#
# Error types
# ===-----------------------------------------------------------------------===#


struct DictKeyError[K: KeyElement](ImplicitlyCopyable, Writable):
    """A custom error type for Dict lookups that fail.

    Parameters:
        K: The key type of the elements in the dictionary.
    """

    @doc_private
    fn __init__(out self):
        pass

    fn write_to(self, mut writer: Some[Writer]):
        """Write the error and the key to the writer.

        Args:
            writer: The writer to write to.
        """
        writer.write("DictKeyError[", get_type_name[Self.K](), "]")


@fieldwise_init
struct EmptyDictError(ImplicitlyCopyable, Writable):
    """A custom error type for when a `Dict` is empty."""

    fn write_to(self, mut writer: Some[Writer]):
        """This always writes "EmptyDictError".

        Args:
            writer: The writer to write to.
        """
        writer.write("EmptyDictError")


# ===-----------------------------------------------------------------------===#
# Iterators
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _DictEntryIter[
    mut: Bool,
    //,
    K: KeyElement,
    V: Copyable & ImplicitlyDestructible,
    H: Hasher,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator over immutable DictEntry references.

    Parameters:
        mut: Whether the reference to the dictionary is mutable.
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
        origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = DictEntry[Self.K, Self.V, Self.H]

    var index: Int
    var seen: Int
    var src: Pointer[Dict[Self.K, Self.V, Self.H], Self.origin]

    fn __init__(
        out self,
        index: Int,
        seen: Int,
        ref[Self.origin] dict: Dict[Self.K, Self.V, Self.H],
    ):
        self.index = index
        self.seen = seen
        self.src = Pointer(to=dict)

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    fn __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        if self.seen >= len(self.src[]):
            raise StopIteration()

        while 0 <= self.index < len(self.src[]._order):
            var idx = self.index

            @parameter
            if Self.forward:
                self.index += 1
            else:
                self.index -= 1

            var slot = Int(self.src[]._order[idx])
            if _is_occupied(self.src[]._ctrl[slot]):
                self.seen += 1
                return (
                    (self.src[]._slots + slot)
                    .unsafe_mut_cast[Self.mut]()
                    .unsafe_origin_cast[Self.origin]()[]
                )

        debug_assert(
            self.seen == len(self.src[]),
            "_order exhausted but not all entries seen: _len out of sync",
        )
        raise StopIteration()

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self.src[]) - self.seen
        return (len, {len})


@fieldwise_init
struct _TakeDictEntryIter[
    K: KeyElement,
    V: Copyable & ImplicitlyDestructible,
    H: Hasher,
    origin: MutOrigin,
](Copyable, Iterable, Iterator):
    """Iterator over mutable DictEntry references that moves entries out of the dictionary.

    Parameters:
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
        origin: The mutable origin of the Dict
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = DictEntry[Self.K, Self.V, Self.H]

    var index: Int
    var src: Pointer[Dict[Self.K, Self.V, Self.H], Self.origin]

    fn __init__(out self, ref[Self.origin] dict: Dict[Self.K, Self.V, Self.H]):
        self.index = 0
        self.src = Pointer(to=dict)

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    fn __next__(
        mut self,
    ) raises StopIteration -> Self.Element:
        if len(self.src[]) <= 0:
            raise StopIteration()

        while self.index < len(self.src[]._order):
            var slot = Int(self.src[]._order[self.index])
            self.index += 1

            if _is_occupied(self.src[]._ctrl[slot]):
                var entry = (self.src[]._slots + slot).take_pointee()
                self.src[]._set_ctrl(slot, _CTRL_DELETED)
                self.src[]._len -= 1
                return entry^

        debug_assert(
            len(self.src[]) == 0,
            "_order exhausted but _len > 0: ctrl bytes and _len out of sync",
        )
        raise StopIteration()


@fieldwise_init
struct _DictKeyIter[
    mut: Bool,
    //,
    K: KeyElement,
    V: Copyable & ImplicitlyDestructible,
    H: Hasher,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator over immutable Dict key references.

    Parameters:
        mut: Whether the reference to the vector is mutable.
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
        origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime dict_entry_iter = _DictEntryIter[
        Self.K, Self.V, Self.H, Self.origin, Self.forward
    ]
    comptime Element = Self.K

    var iter: Self.dict_entry_iter

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    fn __next__(
        mut self,
    ) raises StopIteration -> ref[self.iter.__next__().key] Self.Element:
        return self.iter.__next__().key

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return self.iter.bounds()


@fieldwise_init
struct _DictValueIter[
    mut: Bool,
    //,
    K: KeyElement,
    V: Copyable & ImplicitlyDestructible,
    H: Hasher,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator over Dict value references. These are mutable if the dict
    is mutable.

    Parameters:
        mut: Whether the reference to the vector is mutable.
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
        origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    var iter: _DictEntryIter[Self.K, Self.V, Self.H, Self.origin, Self.forward]
    comptime Element = Self.V

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __reversed__(
        self,
    ) -> _DictValueIter[Self.K, Self.V, Self.H, Self.origin, False]:
        var src = self.iter.src
        return _DictValueIter(
            _DictEntryIter[Self.K, Self.V, Self.H, Self.origin, False](
                len(src[]._order) - 1, 0, src
            )
        )

    fn __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        ref entry_ref = self.iter.__next__()
        # Cast through a pointer to grant additional mutability because
        # _DictEntryIter.next erases it.
        return UnsafePointer(to=entry_ref.value).unsafe_origin_cast[
            Self.origin
        ]()[]

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return self.iter.bounds()


# ===-----------------------------------------------------------------------===#
# DictEntry
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct DictEntry[
    K: KeyElement, V: Copyable & ImplicitlyDestructible, H: Hasher
](Copyable):
    """Store a key-value pair entry inside a dictionary.

    Parameters:
        K: The key type of the dict. Must be Hashable+Equatable.
        V: The value type of the dict.
        H: The type of the hasher used to hash the key.
    """

    var hash: UInt64
    """`key.__hash__()`, stored so hashing isn't re-computed during dict
    lookup."""
    var key: Self.K
    """The unique key for the entry."""
    var value: Self.V
    """The value associated with the key."""

    fn __init__(out self, var key: Self.K, var value: Self.V):
        """Create an entry from a key and value, computing the hash.

        Args:
            key: The key of the entry.
            value: The value of the entry.
        """
        self.hash = hash[HasherType = Self.H](key)
        self.key = key^
        self.value = value^

    fn reap_value(deinit self) -> Self.V:
        """Take the value from an owned entry.

        Returns:
            The value of the entry.
        """
        return self.value^


# ===-----------------------------------------------------------------------===#
# Dict
# ===-----------------------------------------------------------------------===#


struct Dict[
    K: KeyElement,
    V: Copyable & ImplicitlyDestructible,
    H: Hasher = default_hasher,
](
    Boolable,
    Copyable,
    Defaultable,
    Iterable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    """A container that stores key-value pairs.

    The `Dict` type is Mojo's primary associative collection, similar to
    Python's `dict` (dictionary). Unlike a `List`, which stores elements by
    index, a `Dict` stores values associated with unique keys, which enables
    fast lookups, insertions, and deletions.

    You can create a `Dict` in several ways:

    ```mojo
    # Empty dictionary
    var empty_dict = Dict[String, Int]()

    # Dictionary literal syntax
    var scores = {"Alice": 95, "Bob": 87, "Charlie": 92}

    # Pre-allocated capacity
    var large_dict = Dict[String, Int](capacity=64)

    # From separate key and value lists
    var keys = ["red", "green", "blue"]
    var values = [255, 128, 64]
    var colors = Dict[String, Int]()
    for key, value in zip(keys, values):
        colors[String(key)] = value # cast list iterator to key-type
    ```

    Be aware of the following characteristics:

    - **Type safety**: Both keys and values must be homogeneous types,
    determined at compile time. This is more restrictive than Python
    dictionaries but provides better performance:

      ```mojo
      var string_to_int = {"count": 42}     # Dict[String, Int]
      var int_to_string = {1: "one"}        # Dict[Int, String]
      var mixed = {"key": 1, 2: "val"}      # Error! Keys must be same type
      ```

      However, you can get around this by defining your dictionary key and/or
      value type as [`Variant`](/mojo/std/utils/variant/Variant). This is
      a discriminated union type, meaning it can store any number of different
      types that can vary at runtime.

    - **Insertion order**: Iteration over keys, values, and items follows
      insertion order. Updating an existing key's value does not change its
      position. This matches the ordering guarantee of Python's `dict`.

    - **Value semantics**: A `Dict` is value semantic by default. Copying a
      `Dict` creates a deep copy of all key-value pairs. To avoid accidental
      copies, `Dict` is not implicitly copyable—you must explicitly copy it
      using the `.copy()` method.

      ```mojo
      var dict1 = {"a": 1, "b": 2}
      # var dict2 = dict1  # Error: Dict is not implicitly copyable
      var dict2 = dict1.copy()  # Deep copy
      dict2["c"] = 3
      print(dict1.__str__())   # => {"a": 1, "b": 2}
      print(dict2.__str__())   # => {"a": 1, "b": 2, "c": 3}
      ```

      This is different from Python, where assignment creates a reference to
      the same dictionary. For more information, read about [value
      semantics](/mojo/manual/values/value-semantics).

    - **Iteration uses immutable references**: When iterating over keys, values,
      or items, you get immutable references unless you specify `ref` or `var`:

      ```mojo
      var inventory = {"apples": 10, "bananas": 5}

      # Default behavior creates immutable (read-only) references
      for value in inventory.values():
          value += 1  # error: expression must be mutable

      # Using `ref` gets mutable (read-write) references
      for ref value in inventory.values():
          value += 1  # Modify inventory values in-place
      print(inventory.__str__())  # => {"apples": 11, "bananas": 6}

      # Using `var` gets an owned copy of the value
      for var key in inventory.keys():
          inventory[key] += 1  # Modify inventory values in-place
      print(inventory.__str__())  # => {"apples": 12, "bananas": 7}
      ```

      Note that indexing into a `Dict` with a key that's a reference to the
      key owned by the `Dict` produces a confusing error related to
      [argument exclusivity](/mojo/manual/values/ownership#argument-exclusivity).
      Using `var key` in the previous example creates an owned copy of the key,
      avoiding the error.

    - **KeyError handling**: Directly accessing values with the `[]` operator
      will raise `DictKeyError` if the key is not found:

      ```mojo
      var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}
      print(phonebook["Charlie"])  # => DictKeyError
      ```

      For safe access, you should instead use `get()`:

      ```mojo
      var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}
      var phone = phonebook.get("Charlie")
      print(phone.__str__()) if phone else print('phone not found')
      ```


    Examples:

    ```mojo
    var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}

    # Add/update entries
    phonebook["Charlie"] = "555-0103"    # Add new entry
    phonebook["Alice"] = "555-0199"      # Update existing entry

    # Access directly (unsafe and raises DictKeyError if key not found)
    print(phonebook["Alice"])            # => 555-0199

    # Access safely
    var phone = phonebook.get("David")   # Returns Optional type
    print(phone.or_else("phone not found!"))

    # Access safely with default value
    phone = phonebook.get("David", "555-0000")
    print(phone.__str__())               # => '555-0000'

    # Check for keys
    if "Bob" in phonebook:
        print("Found Bob")

    # Remove (pop) entries
    print(phonebook.pop("Charlie"))         # Remove and return: "555-0103"
    print(phonebook.pop("Unknown", "N/A"))  # Pop with default

    # Iterate over a dictionary
    for key in phonebook.keys():
        print("Key:", key)

    for value in phonebook.values():
        print("Value:", value)

    for item in phonebook.items():
        print(item.key, "=>", item.value)

    for var key in phonebook:
        print(key, "=>", phonebook[key])

    # Number of key-value pairs
    print('len:', len(phonebook))        # => len: 2

    # Dictionary operations
    var backup = phonebook.copy()        # Explicit copy
    phonebook.clear()                    # Remove all entries

    # Merge dictionaries
    var more_numbers = {"David": "555-0104", "Eve": "555-0105"}
    backup.update(more_numbers)          # Merge in-place
    var combined = backup | more_numbers # Create new merged dict
    print(combined.__str__())
    ```

    Parameters:
        K: The type of keys stored in the dictionary.
        V: The type of values stored in the dictionary.
        H: The type of hasher used to hash the keys.
    """

    # Implementation:
    #
    # This Dict uses a Swiss Table design with flat layout + insertion-order
    # side array. Key features:
    #
    # - SIMD group probing: 16 control bytes are compared in parallel using
    #   SIMD operations for fast lookups.
    # - Control bytes: Each slot has a 1-byte control that is either EMPTY
    #   (0xFF), DELETED (0x80), or an h2 fingerprint (0x00-0x7F).
    # - Flat slot array: DictEntry values stored directly in a flat array.
    #   Only occupied slots are initialized.
    # - Insertion-order array: A separate List[Int32] tracks the order of
    #   insertion for deterministic iteration.
    # - Load factor 7/8: Higher than CPython's 2/3, enabled by SIMD probing.
    # - Ctrl mirroring: The first GROUP_WIDTH bytes are mirrored after the
    #   main ctrl array to enable SIMD loads that wrap around.

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _DictKeyIter[Self.K, Self.V, Self.H, iterable_origin]
    """The iterator type for this dictionary.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _ctrl: UnsafePointer[UInt8, MutExternalOrigin]
    """Control byte array. Size is _capacity + _GROUP_WIDTH.
    Each byte is EMPTY (0xFF), DELETED (0x80), or h2 fingerprint (0x00-0x7F).
    The last _GROUP_WIDTH bytes mirror the first _GROUP_WIDTH for SIMD wrapping.
    """

    var _slots: UnsafePointer[
        DictEntry[Self.K, Self.V, Self.H], MutExternalOrigin
    ]
    """Flat slot array. Size is _capacity. Only occupied slots are initialized.
    """

    var _order: List[Int32]
    """Insertion-order array of slot indices. Stale entries (from deleted slots)
    are skipped during iteration by checking the ctrl byte.
    """

    var _len: Int
    """The number of live elements currently stored in the dict."""

    var _capacity: Int
    """The number of slots (always a power of 2, >= _INITIAL_CAPACITY)."""

    var _growth_left: Int
    """Number of EMPTY slots that can still be used before a resize is needed.
    Decremented on each new insertion into an EMPTY slot. Reset on resize.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """Initialize an empty dictiontary."""
        self._capacity = _INITIAL_CAPACITY
        self._ctrl = alloc[UInt8](self._capacity + _GROUP_WIDTH)
        memset(self._ctrl, _CTRL_EMPTY, self._capacity + _GROUP_WIDTH)
        self._slots = alloc[DictEntry[Self.K, Self.V, Self.H]](self._capacity)
        self._order = List[Int32](capacity=self._capacity * 7 // 8)
        self._len = 0
        self._growth_left = self._capacity * 7 // 8

    @always_inline
    fn __init__(out self, *, capacity: Int):
        """Initialize an empty dictionary with a pre-reserved capacity.

        The capacity is rounded up to the next power of two (minimum 16)
        to satisfy internal layout requirements. The usable capacity
        before resizing is 7/8 of the rounded value.

        Args:
            capacity: The requested minimum number of slots.

        Examples:

        ```mojo
        var x = Dict[Int, Int](capacity=1000)
        # Actual capacity is 1024; can hold 896 entries without resizing.
        ```
        """
        self._capacity = max(next_power_of_two(capacity), _INITIAL_CAPACITY)
        self._ctrl = alloc[UInt8](self._capacity + _GROUP_WIDTH)
        memset(self._ctrl, _CTRL_EMPTY, self._capacity + _GROUP_WIDTH)
        self._slots = alloc[DictEntry[Self.K, Self.V, Self.H]](self._capacity)
        self._order = List[Int32](capacity=self._capacity * 7 // 8)
        self._len = 0
        self._growth_left = self._capacity * 7 // 8

    @always_inline
    fn __init__(
        out self,
        var keys: List[Self.K],
        var values: List[Self.V],
        __dict_literal__: (),
    ):
        """Constructs a dictionary from the given keys and values.

        Args:
            keys: The list of keys to build the dictionary with.
            values: The corresponding values to pair with the keys.
            __dict_literal__: Tell Mojo to use this method for dict literals.
        """
        # TODO: Use capacity to reserve space.
        self = Self()
        debug_assert(
            len(keys) == len(values),
            "keys and values must have the same length",
        )

        # TODO: Should transfer the key/value's from the list to avoid copying
        # the values.
        for key, value in zip(keys, values):
            self._insert(key.copy(), value.copy())

    # TODO: add @property when Mojo supports it to make
    # it possible to do `self._reserved`.
    @always_inline
    fn _reserved(self) -> Int:
        return self._capacity

    @staticmethod
    fn fromkeys(keys: List[Self.K, ...], value: Self.V) -> Self:
        """Create a new dictionary with keys from list and values set to value.

        Args:
            keys: The keys to set.
            value: The value to set.

        Returns:
            The new dictionary.
        """
        var my_dict = Dict[Self.K, Self.V, Self.H]()
        for key in keys:
            my_dict[key.copy()] = value.copy()
        return my_dict^

    @staticmethod
    fn fromkeys(
        keys: List[Self.K, ...], value: Optional[Self.V] = None
    ) -> Dict[Self.K, Optional[Self.V], Self.H]:
        """Create a new dictionary with keys from list and values set to value.

        Args:
            keys: The keys to set.
            value: The value to set.

        Returns:
            The new dictionary.
        """
        return Dict[Self.K, Optional[Self.V], Self.H].fromkeys(keys, value)

    fn __copyinit__(out self, copy: Self):
        """Copy an existing dictiontary.

        Args:
            copy: The existing dict.
        """
        self._capacity = copy._capacity
        self._len = copy._len
        self._growth_left = copy._growth_left

        # Allocate and copy control bytes
        self._ctrl = alloc[UInt8](self._capacity + _GROUP_WIDTH)
        memcpy(
            dest=self._ctrl,
            src=copy._ctrl,
            count=self._capacity + _GROUP_WIDTH,
        )

        # Allocate slots and deep-copy occupied entries
        self._slots = alloc[DictEntry[Self.K, Self.V, Self.H]](self._capacity)
        for i in range(self._capacity):
            if _is_occupied(self._ctrl[i]):
                (self._slots + i).init_pointee_copy((copy._slots + i)[])

        # Copy the order array
        self._order = copy._order.copy()

    fn __del__(deinit self):
        """Destroy all keys and values in the dictionary and free memory."""
        # Destroy all occupied slot entries
        for i in range(self._capacity):
            if _is_occupied(self._ctrl[i]):
                (self._slots + i).destroy_pointee()

        # Free allocated memory
        self._ctrl.free()
        self._slots.free()
        # _order is cleaned up by List destructor

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __getitem__(
        ref self, ref key: Self.K
    ) raises DictKeyError[Self.K] -> ref[self] Self.V:
        """Retrieve a value out of the dictionary.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, if it's present.

        Raises:
            `DictKeyError` if the key isn't present.
        """
        return self._find_ref(key)

    fn __setitem__(mut self, var key: Self.K, var value: Self.V):
        """Set a value in the dictionary by key.

        Args:
            key: The key to associate with the specified value.
            value: The data to store in the dictionary.
        """
        self._insert(key^, value^)

    fn __contains__(self, key: Self.K) -> Bool:
        """Check if a given key is in the dictionary or not.

        Args:
            key: The key to check.

        Returns:
            True if the key exists in the dictionary, False otherwise.
        """
        var found, _ = self._find_slot(hash[HasherType = Self.H](key), key)
        return found

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return _DictKeyIter(_DictEntryIter(0, 0, self))

    fn __reversed__(
        ref self,
    ) -> _DictKeyIter[Self.K, Self.V, Self.H, origin_of(self), False]:
        """Iterate backwards over the dict keys, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the dict keys.
        """
        return _DictKeyIter(
            _DictEntryIter[forward=False](len(self._order) - 1, 0, self)
        )

    fn __or__(self, other: Self) -> Self:
        """Merge self with other and return the result as a new dict.

        Args:
            other: The dictionary to merge with.

        Returns:
            The result of the merge.
        """
        var result = self.copy()
        result.update(other)
        return result^

    fn __ior__(mut self, other: Self):
        """Merge self with other in place.

        Args:
            other: The dictionary to merge with.
        """
        self.update(other)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __len__(self) -> Int:
        """The number of elements currently stored in the dictionary.

        Returns:
            The number of elements currently stored in the dictionary.
        """
        return self._len

    fn __bool__(self) -> Bool:
        """Check if the dictionary is empty or not.

        Returns:
            `False` if the dictionary is empty, `True` if there is at least one
            element.
        """
        return len(self).__bool__()

    @no_inline
    fn __repr__(self) -> String:
        """Returns a string representation of a `Dict`.

        Returns:
            A string representation of the Dict.
        """
        return self.__str__()

    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of a `Dict`.

        Returns:
            A string representation of the Dict.

        Examples:

        ```mojo
        var my_dict = Dict[Int, Float64]()
        my_dict[1] = 1.1
        my_dict[2] = 2.2
        dict_as_string = String(my_dict)
        print(dict_as_string)
        # prints "{1: 1.1, 2: 2.2}"
        ```
        """
        var minimum_capacity = self._minimum_size_of_string_representation()
        var output = String(capacity=minimum_capacity)
        self.write_to(output)
        return output^

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Write `my_list.__str__()` to a `Writer`.

        Constraints:
            `K` must conform to `Representable`.
            `V` must conform to `Representable`.

        Args:
            writer: The object to write to.
        """
        _constrained_conforms_to[
            conforms_to(Self.K, Representable),
            Parent=Self,
            Element = Self.K,
            ParentConformsTo="Stringable",
            ElementConformsTo="Representable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.V, Representable),
            Parent=Self,
            Element = Self.V,
            ParentConformsTo="Stringable",
            ElementConformsTo="Representable",
        ]()

        writer.write("{")

        var i = 0
        for key_entry in self.items():
            ref key = trait_downcast[Representable](key_entry.key)
            ref val = trait_downcast[Representable](key_entry.value)
            writer.write(repr(key), ": ", repr(val))
            if i < len(self) - 1:
                writer.write(", ")
            i += 1
        writer.write("}")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn _minimum_size_of_string_representation(self) -> Int:
        # we do a rough estimation of the minimum number of chars that we'll see
        # in the string representation, we assume that String(key) and String(value)
        # will be both at least one char.
        return (
            2  # '{' and '}'
            + len(self) * 6  # String(key), String(value) ": " and ", "
            - 2  # remove the last ", "
        )

    fn find(self, key: Self.K) -> Optional[Self.V]:
        """Find a value in the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """

        try:
            return self._find_ref(key).copy()
        except:
            return Optional[Self.V](None)

    fn _find_ref(
        ref self, ref key: Self.K
    ) raises DictKeyError[Self.K] -> ref[self] Self.V:
        """Find a value in the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a reference to the value if it is
            present, otherwise an empty Optional.
        """
        var hash = hash[HasherType = Self.H](key)
        var found, slot_idx = self._find_slot(hash, key)

        if found:
            debug_assert(
                _is_occupied(self._ctrl[slot_idx]),
                "_find_slot returned found=True but ctrl byte is not occupied",
            )
            return (self._slots + slot_idx)[].value

        raise DictKeyError[Self.K]()

    fn get(self, key: Self.K) -> Optional[Self.V]:
        """Get a value from the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        return self.find(key)

    fn get(self, key: Self.K, var default: Self.V) -> Self.V:
        """Get a value from the dictionary by key.

        Args:
            key: The key to search for in the dictionary.
            default: Default value to return.

        Returns:
            A copy of the value if it was present, otherwise default.
        """
        return self.find(key).or_else(default^)

    fn pop(mut self, key: Self.K, var default: Self.V) -> Self.V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.
            default: A default value to return if the key
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the dictionary.
            If it wasn't, return the provided default value instead.
        """
        try:
            return self.pop(key)
        except:
            return default^

    fn pop(mut self, ref key: Self.K) raises DictKeyError[Self.K] -> Self.V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.

        Returns:
            The value associated with the key, if it was in the dictionary.
            Raises otherwise.

        Raises:
            `DictKeyError` if the key was not present in the dictionary.
        """
        var hash = hash[HasherType = Self.H](key)
        var found, slot_idx = self._find_slot(hash, key)
        if found:
            debug_assert(
                _is_occupied(self._ctrl[slot_idx]),
                "_find_slot returned found=True but ctrl byte is not occupied",
            )
            var entry = (self._slots + slot_idx).take_pointee()
            self._set_ctrl(slot_idx, _CTRL_DELETED)
            self._len -= 1
            return entry^.reap_value()
        raise DictKeyError[Self.K]()

    fn popitem(
        mut self,
    ) raises EmptyDictError -> DictEntry[Self.K, Self.V, Self.H]:
        """Remove and return a (key, value) pair from the dictionary.

        Returns:
            Last dictionary item

        Raises:
            `EmptyDictError` if the dictionary is empty.

        Notes:
            Pairs are returned in LIFO order. popitem() is useful to
            destructively iterate over a dictionary, as often used in set
            algorithms. If the dictionary is empty, calling popitem() raises a
            EmptyDictError.
        """

        var i = len(self._order) - 1
        while i >= 0:
            var slot = Int(self._order[i])
            if _is_occupied(self._ctrl[slot]):
                var entry = (self._slots + slot).take_pointee()
                self._set_ctrl(slot, _CTRL_DELETED)
                self._len -= 1
                return entry^
            i -= 1

        raise EmptyDictError()

    fn keys(ref self) -> _DictKeyIter[Self.K, Self.V, Self.H, origin_of(self)]:
        """Iterate over the dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return Self.__iter__(self)

    fn values(
        ref self,
    ) -> _DictValueIter[Self.K, Self.V, Self.H, origin_of(self)]:
        """Iterate over the dict's values as references.

        Returns:
            An iterator of references to the dictionary values.
        """
        return _DictValueIter(_DictEntryIter(0, 0, self))

    fn items(
        ref self,
    ) -> _DictEntryIter[Self.K, Self.V, Self.H, origin_of(self)]:
        """Iterate over the dict's entries as immutable references.

        Returns:
            An iterator of immutable references to the dictionary entries.

        Examples:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2

        for e in my_dict.items():
            print(e.key, e.value)
        ```

        Notes:
            These can't yet be unpacked like Python dict items, but you can
            access the key and value as attributes.
        """
        return _DictEntryIter(0, 0, self)

    fn take_items(
        mut self,
    ) -> _TakeDictEntryIter[Self.K, Self.V, Self.H, origin_of(self)]:
        """Iterate over the dict's entries and move them out of the dictionary
        effectively draining the dictionary.

        Returns:
            An iterator of mutable references to the dictionary entries that
            moves them out of the dictionary.

        Examples:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2

        for entry in my_dict.take_items():
            print(entry.key, entry.value)

        print(len(my_dict))
        # prints 0
        ```
        """
        return _TakeDictEntryIter(self)

    fn update(mut self, other: Self, /):
        """Update the dictionary with the key/value pairs from other,
        overwriting existing keys.

        Args:
            other: The dictionary to update from.

        Notes:
            The argument must be positional only.
        """
        for entry in other.items():
            self[entry.key.copy()] = entry.value.copy()

    fn clear(mut self):
        """Remove all elements from the dictionary."""
        # Destroy all occupied entries
        for i in range(self._capacity):
            if _is_occupied(self._ctrl[i]):
                (self._slots + i).destroy_pointee()

        # Reset ctrl to all EMPTY
        memset(self._ctrl, _CTRL_EMPTY, self._capacity + _GROUP_WIDTH)

        # Clear state
        self._order.clear()
        self._len = 0
        self._growth_left = self._capacity * 7 // 8

    fn setdefault(
        mut self, key: Self.K, var default: Self.V
    ) -> ref[self] Self.V:
        """Get a value from the dictionary by key, or set it to a default if it
        doesn't exist.

        Args:
            key: The key to search for in the dictionary.
            default: The default value to set if the key is not present.

        Returns:
            The value associated with the key, or the default value if it wasn't
            present.
        """
        self._maybe_resize()
        var h = hash[HasherType = Self.H](key)
        var found, slot_idx = self._find_slot(h, key)
        if not found:
            var entry = DictEntry[H = Self.H](key.copy(), default^)
            self._set_ctrl(slot_idx, _h2(h))
            (self._slots + slot_idx).init_pointee_move(entry^)
            self._order.append(Int32(slot_idx))
            self._len += 1
            self._growth_left -= 1
        else:
            debug_assert(
                _is_occupied(self._ctrl[slot_idx]),
                "_find_slot returned found=True but ctrl byte is not occupied",
            )
        return (self._slots + slot_idx)[].value

    # ===-------------------------------------------------------------------===#
    # Internal methods
    # ===-------------------------------------------------------------------===#

    fn _insert(mut self, var key: Self.K, var value: Self.V):
        self._insert(DictEntry[Self.K, Self.V, Self.H](key^, value^))

    fn _insert[
        safe_context: Bool = False
    ](mut self, var entry: DictEntry[Self.K, Self.V, Self.H]):
        @parameter
        if not safe_context:
            self._maybe_resize()
        var found, slot_idx = self._find_slot(entry.hash, entry.key)

        if found:
            # Update existing entry: destroy old, move new in
            (self._slots + slot_idx).destroy_pointee()
            (self._slots + slot_idx).init_pointee_move(entry^)
        else:
            # New entry
            self._set_ctrl(slot_idx, _h2(entry.hash))
            (self._slots + slot_idx).init_pointee_move(entry^)
            self._order.append(Int32(slot_idx))
            self._len += 1
            self._growth_left -= 1
            debug_assert(
                self._growth_left >= 0,
                "_growth_left went negative after insert",
            )

    @always_inline
    fn _set_ctrl(mut self, index: Int, value: UInt8):
        """Set a control byte, maintaining the mirror for wrap-around SIMD loads.

        Args:
            index: The slot index.
            value: The control byte value (h2, EMPTY, or DELETED).
        """
        debug_assert(
            0 <= index < self._capacity,
            "ctrl index out of bounds",
        )
        self._ctrl[index] = value
        # Mirror first GROUP_WIDTH bytes at the end of the ctrl array
        if index < _GROUP_WIDTH:
            self._ctrl[self._capacity + index] = value

    @always_inline
    fn _find_slot(self, hash: UInt64, key: Self.K) -> Tuple[Bool, Int]:
        """Find a slot matching the given key, or an empty slot for insertion.

        This intentionally does NOT return DELETED slots for insertion.
        Reusing a DELETED slot would place the entry at an arbitrary position
        in the slot array, but the caller appends to `_order` expecting the
        slot index to correspond to insertion order. Skipping DELETED slots
        keeps the mapping between `_order` and `_slots` consistent, and
        DELETED slots are reclaimed during resize anyway.

        Args:
            hash: The hash of the key.
            key: The key to search for.

        Returns:
            A tuple of (found, slot_index). If found, slot_index is the
            matching slot. If not found, slot_index is the first EMPTY slot
            suitable for insertion.
        """
        var h2_val = _h2(hash)
        var pos = Int(hash) & (self._capacity - 1)

        while True:
            var group = _Group(self._ctrl + pos)

            # Check for h2 fingerprint matches in this group
            var match_mask = group.match_h2(h2_val)
            while match_mask != 0:
                var bit = count_trailing_zeros(Int(match_mask))
                var slot_idx = (pos + bit) & (self._capacity - 1)
                if (self._slots + slot_idx)[].hash == hash and likely(
                    (self._slots + slot_idx)[].key == key
                ):
                    return (True, slot_idx)
                match_mask &= match_mask - 1  # Clear lowest set bit

            # If any EMPTY slot in this group, key is definitely absent
            var empty_mask = group.match_empty()
            if empty_mask != 0:
                var bit = count_trailing_zeros(Int(empty_mask))
                return (False, (pos + bit) & (self._capacity - 1))

            # No match and no EMPTY in this group, continue probing
            pos = (pos + _GROUP_WIDTH) & (self._capacity - 1)

    @always_inline
    fn _find_empty_slot(self, hash: UInt64) -> Int:
        """Find the first EMPTY or DELETED slot for the given hash.

        Used during resize when we know the key is unique.

        Args:
            hash: The hash to determine the starting probe position.

        Returns:
            The index of the first available slot.
        """
        var pos = Int(hash) & (self._capacity - 1)

        while True:
            var group = _Group(self._ctrl + pos)
            var mask = group.match_empty_or_deleted()
            if mask != 0:
                var bit = count_trailing_zeros(Int(mask))
                return (pos + bit) & (self._capacity - 1)
            pos = (pos + _GROUP_WIDTH) & (self._capacity - 1)

    fn _maybe_resize(mut self):
        """Resize the table if growth_left has been exhausted."""
        if self._growth_left > 0:
            self._maybe_compact_order()
            return

        # TODO: When most non-EMPTY slots are DELETED (tombstones) rather than
        # occupied, we could rehash in-place at the same capacity instead of
        # doubling. This avoids unnecessary memory growth for workloads with
        # heavy insert/delete churn. The abseil Swiss Table implements this as:
        # if _len <= capacity * 7 / 16, rehash at same size; else double.
        # For now we always double, which is correct but uses more memory.

        # Double capacity and rehash
        var new_capacity = self._capacity * 2
        var old_ctrl = self._ctrl
        var old_slots = self._slots
        var old_order = self._order^

        # Allocate new storage
        self._ctrl = alloc[UInt8](new_capacity + _GROUP_WIDTH)
        memset(self._ctrl, _CTRL_EMPTY, new_capacity + _GROUP_WIDTH)
        self._slots = alloc[DictEntry[Self.K, Self.V, Self.H]](new_capacity)
        self._capacity = new_capacity
        self._growth_left = new_capacity * 7 // 8 - self._len

        # Rebuild order (compacted) by walking old order
        self._order = List[Int32](capacity=self._len)

        for i in range(len(old_order)):
            var old_slot = Int(old_order[i])
            if _is_occupied(old_ctrl[old_slot]):
                # Move entry from old table to new table
                var entry = (old_slots + old_slot).take_pointee()
                var h2_val = _h2(entry.hash)
                var new_slot = self._find_empty_slot(entry.hash)
                self._set_ctrl(new_slot, h2_val)
                (self._slots + new_slot).init_pointee_move(entry^)
                self._order.append(Int32(new_slot))

        debug_assert(
            len(self._order) == self._len,
            "order length doesn't match _len after resize",
        )

        # Free old storage
        old_ctrl.free()
        old_slots.free()

    fn _maybe_compact_order(mut self):
        """Compact the order array if it has too many stale entries."""
        if len(self._order) <= 2 * self._len:
            return
        var new_order = List[Int32](capacity=self._len)
        for i in range(len(self._order)):
            var slot = Int(self._order[i])
            if _is_occupied(self._ctrl[slot]):
                new_order.append(self._order[i])
        self._order = new_order^


struct OwnedKwargsDict[V: Copyable & ImplicitlyDestructible](
    Copyable, Defaultable, Iterable, Sized
):
    """Container used to pass owned variadic keyword arguments to functions.

    Parameters:
        V: The value type of the dictionary. Currently must be Copyable.

    This type mimics the interface of a dictionary with `String` keys, and
    should be usable more-or-less like a dictionary. Notably, however, this type
    should not be instantiated directly by users.
    """

    # Fields
    comptime key_type = String
    """The key type for this dictionary (always String)."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _DictKeyIter[
        Self.key_type, Self.V, default_comp_time_hasher, iterable_origin
    ]
    """The iterator type for this dictionary.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    var _dict: Dict[Self.key_type, Self.V, default_comp_time_hasher]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Initialize an empty keyword dictionary."""
        self._dict = Dict[Self.key_type, Self.V, default_comp_time_hasher]()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__(
        ref self, ref key: Self.key_type
    ) raises DictKeyError[Self.key_type] -> ref[self._dict[key]] Self.V:
        """Retrieve a value out of the keyword dictionary.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, if it's present.

        Raises:
            `DictKeyError` if the key isn't present.
        """
        return self._dict[key]

    @always_inline
    fn __setitem__(mut self, key: Self.key_type, var value: Self.V):
        """Set a value in the keyword dictionary by key.

        Args:
            key: The key to associate with the specified value.
            value: The data to store in the dictionary.
        """
        self._dict[key] = value^

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __contains__(self, key: Self.key_type) -> Bool:
        """Check if a given key is in the keyword dictionary or not.

        Args:
            key: The key to check.

        Returns:
            True if there key exists in the keyword dictionary, False
            otherwise.
        """
        return key in self._dict

    @always_inline
    fn __len__(self) -> Int:
        """The number of elements currently stored in the keyword dictionary.

        Returns:
            The number of elements currently stored in the keyword dictionary.
        """
        return len(self._dict)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn find(self, key: Self.key_type) -> Optional[Self.V]:
        """Find a value in the keyword dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        return self._dict.find(key)

    @always_inline
    fn pop(mut self, key: self.key_type, var default: Self.V) -> Self.V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.
            default: A default value to return if the key
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the dictionary.
            If it wasn't, return the provided default value instead.
        """
        return self._dict.pop(key, default^)

    @always_inline
    fn pop(
        mut self, ref key: self.key_type
    ) raises DictKeyError[Self.key_type] -> Self.V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.

        Returns:
            The value associated with the key, if it was in the dictionary.
            Raises otherwise.

        Raises:
            `DictKeyError` if the key was not present in the dictionary.
        """
        return self._dict.pop(key)

    fn __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the keyword dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return rebind[Self.IteratorType[origin_of(self)]](self._dict.keys())

    fn keys(
        ref self,
    ) -> _DictKeyIter[
        Self.key_type, Self.V, default_comp_time_hasher, origin_of(self._dict)
    ]:
        """Iterate over the keyword dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return self._dict.keys()

    fn values(
        ref self,
    ) -> _DictValueIter[
        Self.key_type, Self.V, default_comp_time_hasher, origin_of(self._dict)
    ]:
        """Iterate over the keyword dict's values as references.

        Returns:
            An iterator of references to the dictionary values.
        """
        return self._dict.values()

    fn items(
        ref self,
    ) -> _DictEntryIter[
        Self.key_type, Self.V, default_comp_time_hasher, origin_of(self._dict)
    ]:
        """Iterate over the keyword dictionary's entries as immutable
        references.

        Returns:
            An iterator of immutable references to the dictionary entries.

        Examples:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2

        for e in my_dict.items():
            print(e.key, e.value)
        ```

        Notes:
            These can't yet be unpacked like Python dict items, but you can
            access the key and value as attributes.
        """

        # TODO(#36448): Use this instead of the current workaround
        # return self[]._dict.items()
        return _DictEntryIter(0, 0, self._dict)

    @always_inline
    fn _insert(mut self, var key: Self.key_type, var value: Self.V):
        self._dict._insert(key^, value^)

    @always_inline
    fn _insert(mut self, key: StringLiteral, var value: Self.V):
        self._insert(String(key), value^)
