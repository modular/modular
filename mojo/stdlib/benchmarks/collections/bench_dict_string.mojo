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

from collections.string.string_slice import _to_string_list
from hashlib import default_comp_time_hasher, default_hasher
from os import abort
from pathlib import _dir_of_current_file
from sys import stderr

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    Format,
    Unit,
    keep,
    run,
)
from memory import memcpy, memset_zero
from testing import assert_equal


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
fn make_small_keys(filename: String = "UN_charter_EN.txt") -> List[String]:
    """Make a `String` made of items in the `./data` directory.

    Args:
        filename: The name of the file inside the `./data` directory.
    """

    try:
        directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")
        var content = f.read()
        return _to_string_list(content.split())
    except e:
        print(e, file=stderr)
    abort()


# ===-----------------------------------------------------------------------===#
# Long Key Data
# ===-----------------------------------------------------------------------===#
fn make_long_keys(filename: String = "UN_charter_EN.txt") -> List[String]:
    """Make a `String` made of items in the `./data` directory.

    Args:
        filename: The name of the file inside the `./data` directory.
    """

    try:
        directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")
        var content = f.read()
        return _to_string_list(content.split("\n"))
    except e:
        print(e, file=stderr)
    abort()


# ===-----------------------------------------------------------------------===#
# String Dict implementation for benchmarking baseline against Dict
# ===-----------------------------------------------------------------------===#

from bit import bit_width, pop_count


struct KeysContainer[KeyEndType: DType = DType.uint32](
    ImplicitlyCopyable, Sized
):
    var keys: UnsafePointer[UInt8, MutAnyOrigin]
    var allocated_bytes: Int
    var keys_end: UnsafePointer[Scalar[Self.KeyEndType], MutAnyOrigin]
    var count: Int
    var capacity: Int

    fn __init__(out self, capacity: Int):
        comptime assert (
            Self.KeyEndType == DType.uint8
            or Self.KeyEndType == DType.uint16
            or Self.KeyEndType == DType.uint32
            or Self.KeyEndType == DType.uint64
        ), "KeyEndType needs to be an unsigned integer"
        self.allocated_bytes = capacity << 3
        self.keys = alloc[UInt8](self.allocated_bytes)
        self.keys_end = alloc[Scalar[Self.KeyEndType]](capacity)
        self.count = 0
        self.capacity = capacity

    fn __copyinit__(out self, copy: Self):
        self.allocated_bytes = copy.allocated_bytes
        self.count = copy.count
        self.capacity = copy.capacity
        self.keys = alloc[UInt8](self.allocated_bytes)
        memcpy(dest=self.keys, src=copy.keys, count=self.allocated_bytes)
        self.keys_end = alloc[Scalar[Self.KeyEndType]](self.capacity)
        memcpy(dest=self.keys_end, src=copy.keys_end, count=self.capacity)

    fn __del__(deinit self):
        self.keys.free()
        self.keys_end.free()

    @always_inline
    fn add(mut self, key: StringSlice):
        var prev_end = 0 if self.count == 0 else self.keys_end[self.count - 1]
        var key_length = len(key)
        var new_end = prev_end + Scalar[Self.KeyEndType](key_length)

        var needs_realocation = False
        while new_end > Scalar[Self.KeyEndType](self.allocated_bytes):
            self.allocated_bytes += self.allocated_bytes >> 1
            needs_realocation = True

        if needs_realocation:
            var keys = alloc[UInt8](self.allocated_bytes)
            memcpy(dest=keys, src=self.keys, count=Int(prev_end))
            self.keys.free()
            self.keys = keys

        memcpy(
            dest=self.keys + prev_end,
            src=UnsafePointer(key.unsafe_ptr()),
            count=key_length,
        )
        var count = self.count + 1
        if count >= self.capacity:
            var new_capacity = self.capacity + (self.capacity >> 1)
            var keys_end = alloc[Scalar[Self.KeyEndType]](new_capacity)
            memcpy(dest=keys_end, src=self.keys_end, count=self.capacity)
            self.keys_end.free()
            self.keys_end = keys_end
            self.capacity = new_capacity

        self.keys_end.store(self.count, new_end)
        self.count = count

    @always_inline
    fn get(self, index: Int) -> StringSlice[ImmutAnyOrigin]:
        if index < 0 or index >= self.count:
            return StringSlice(unsafe_from_utf8=Span(ptr=self.keys, length=0))
        var start = 0 if index == 0 else Int(self.keys_end[index - 1])
        var length = Int(self.keys_end[index]) - start
        return StringSlice(
            unsafe_from_utf8=Span(ptr=self.keys + start, length=length)
        )

    @always_inline
    fn clear(mut self):
        self.count = 0

    @always_inline
    fn __getitem__(self, index: Int) -> StringSlice[ImmutAnyOrigin]:
        return self.get(index)

    @always_inline
    fn __len__(self) -> Int:
        return self.count

    fn keys_vec(self) -> List[StringSlice[ImmutAnyOrigin]]:
        var keys = List[StringSlice[ImmutAnyOrigin]](capacity=self.count)
        for i in range(self.count):
            keys.append(self[i])
        return keys^

    fn print_keys(self):
        print("(" + String(self.count) + ")[", end="")
        for i in range(self.count):
            var end = ", " if i < self.count - 1 else ""
            print(self[i], end=end)
        print("]")


struct StringDict[
    V: Copyable & ImplicitlyDestructible,
    KeyCountType: DType = DType.uint32,
    KeyOffsetType: DType = DType.uint32,
    destructive: Bool = True,
    caching_hashes: Bool = True,
](Sized):
    var keys: KeysContainer[Self.KeyOffsetType]
    var key_hashes: UnsafePointer[Scalar[Self.KeyCountType], MutAnyOrigin]
    var values: List[Self.V]
    var slot_to_index: UnsafePointer[Scalar[Self.KeyCountType], MutAnyOrigin]
    var deleted_mask: UnsafePointer[UInt8, MutAnyOrigin]
    var count: Int
    var capacity: Int

    fn __init__(out self, capacity: Int = 16):
        comptime assert (
            Self.KeyCountType == DType.uint8
            or Self.KeyCountType == DType.uint16
            or Self.KeyCountType == DType.uint32
            or Self.KeyCountType == DType.uint64
        ), "KeyCountType needs to be an unsigned integer"
        self.count = 0
        if capacity <= 8:
            self.capacity = 8
        else:
            var icapacity = Int64(capacity)
            self.capacity = capacity if pop_count(icapacity) == 1 else 1 << Int(
                bit_width(icapacity)
            )
        self.keys = KeysContainer[Self.KeyOffsetType](capacity)

        comptime if Self.caching_hashes:
            self.key_hashes = alloc[Scalar[Self.KeyCountType]](self.capacity)
        else:
            self.key_hashes = alloc[Scalar[Self.KeyCountType]](0)
        self.values = List[Self.V](capacity=capacity)
        self.slot_to_index = alloc[Scalar[Self.KeyCountType]](self.capacity)
        memset_zero(self.slot_to_index, self.capacity)

        comptime if Self.destructive:
            self.deleted_mask = alloc[UInt8](self.capacity >> 3)
            memset_zero(self.deleted_mask, self.capacity >> 3)
        else:
            self.deleted_mask = alloc[UInt8](0)

    fn __copyinit__(out self, copy: Self):
        self.count = copy.count
        self.capacity = copy.capacity
        self.keys = copy.keys

        comptime if Self.caching_hashes:
            self.key_hashes = alloc[Scalar[Self.KeyCountType]](self.capacity)
            memcpy(
                dest=self.key_hashes,
                src=copy.key_hashes,
                count=self.capacity,
            )
        else:
            self.key_hashes = alloc[Scalar[Self.KeyCountType]](0)
        self.values = copy.values.copy()
        self.slot_to_index = alloc[Scalar[Self.KeyCountType]](self.capacity)
        memcpy(
            dest=self.slot_to_index,
            src=copy.slot_to_index,
            count=self.capacity,
        )

        comptime if Self.destructive:
            self.deleted_mask = alloc[UInt8](self.capacity >> 3)
            memcpy(
                dest=self.deleted_mask,
                src=copy.deleted_mask,
                count=self.capacity >> 3,
            )
        else:
            self.deleted_mask = alloc[UInt8](0)

    fn __del__(deinit self):
        self.slot_to_index.free()
        self.deleted_mask.free()
        self.key_hashes.free()

    fn __len__(self) -> Int:
        return self.count

    @always_inline
    fn __contains__(self, key: StringSlice) -> Bool:
        return self._find_key_index(key) != 0

    fn put(mut self, key: StringSlice, value: Self.V):
        if self.count >= self.capacity - (self.capacity >> 3):
            self._rehash()

        var key_hash = hash(key).cast[Self.KeyCountType]()
        var modulo_mask = self.capacity - 1
        var slot = Int(key_hash & Scalar[Self.KeyCountType](modulo_mask))
        while True:
            var key_index = Int(self.slot_to_index.load(slot))
            if key_index == 0:
                self.keys.add(key)

                comptime if Self.caching_hashes:
                    self.key_hashes.store(slot, key_hash)
                self.values.append(value.copy())
                self.count += 1
                self.slot_to_index.store(
                    slot, Scalar[Self.KeyCountType](self.keys.count)
                )
                return

            comptime if Self.caching_hashes:
                var other_key_hash = self.key_hashes[slot]
                if other_key_hash == key_hash:
                    var other_key = self.keys[key_index - 1]
                    if other_key == key:
                        # replace value
                        self.values[key_index - 1] = value.copy()

                        comptime if Self.destructive:
                            if self._is_deleted(key_index - 1):
                                self.count += 1
                                self._not_deleted(key_index - 1)
                        return
            else:
                var other_key = self.keys[key_index - 1]
                if other_key == key:
                    # replace value
                    self.values[key_index - 1] = value.copy()

                    comptime if Self.destructive:
                        if self._is_deleted(key_index - 1):
                            self.count += 1
                            self._not_deleted(key_index - 1)
                    return

            slot = (slot + 1) & modulo_mask

    @always_inline
    fn _is_deleted(self, index: Int) -> Bool:
        var offset = index >> 3
        var bit_index = index & 7
        return (self.deleted_mask + offset).load() & UInt8(1 << bit_index) != 0

    @always_inline
    fn _deleted(self, index: Int):
        var offset = index >> 3
        var bit_index = index & 7
        var p = self.deleted_mask + offset
        var mask = p.load()
        p.store(mask | UInt8((1 << bit_index)))

    @always_inline
    fn _not_deleted(self, index: Int):
        var offset = index >> 3
        var bit_index = index & 7
        var p = self.deleted_mask + offset
        var mask = p.load()
        p.store(mask & UInt8(~(1 << bit_index)))

    @always_inline
    fn _rehash(mut self):
        var old_slot_to_index = self.slot_to_index
        var old_capacity = self.capacity
        self.capacity <<= 1
        var mask_capacity = self.capacity >> 3
        self.slot_to_index = alloc[Scalar[Self.KeyCountType]](self.capacity)
        memset_zero(self.slot_to_index, self.capacity)

        var key_hashes = self.key_hashes

        comptime if Self.caching_hashes:
            key_hashes = alloc[Scalar[Self.KeyCountType]](self.capacity)

        comptime if Self.destructive:
            var deleted_mask = alloc[UInt8](mask_capacity)
            memset_zero(deleted_mask, mask_capacity)
            memcpy(
                dest=deleted_mask,
                src=self.deleted_mask,
                count=old_capacity >> 3,
            )
            self.deleted_mask.free()
            self.deleted_mask = deleted_mask

        var modulo_mask = self.capacity - 1
        for i in range(old_capacity):
            if old_slot_to_index[i] == 0:
                continue
            var key_hash = Scalar[Self.KeyCountType](0)

            comptime if Self.caching_hashes:
                key_hash = self.key_hashes[i]
            else:
                key_hash = hash(self.keys[Int(old_slot_to_index[i] - 1)]).cast[
                    Self.KeyCountType
                ]()

            var slot = Int(key_hash & Scalar[Self.KeyCountType](modulo_mask))

            # var searching = True
            while True:
                var key_index = Int(self.slot_to_index.load(slot))

                if key_index == 0:
                    self.slot_to_index.store(slot, old_slot_to_index[i])
                    break
                    # searching = False

                else:
                    slot = (slot + 1) & modulo_mask

            comptime if Self.caching_hashes:
                key_hashes[slot] = key_hash

        comptime if Self.caching_hashes:
            self.key_hashes.free()
            self.key_hashes = key_hashes
        old_slot_to_index.free()

    fn get(self, key: StringSlice, default: Self.V) -> Self.V:
        var key_index = self._find_key_index(key)
        if key_index == 0:
            return default.copy()

        comptime if Self.destructive:
            if self._is_deleted(key_index - 1):
                return default.copy()
        return self.values[key_index - 1].copy()

    fn delete(mut self, key: StringSlice):
        comptime if not Self.destructive:
            return

        var key_index = self._find_key_index(key)
        if key_index == 0:
            return
        if not self._is_deleted(key_index - 1):
            self.count -= 1
        self._deleted(key_index - 1)

    fn upsert(
        mut self,
        key: StringSlice,
        update: fn(value: Optional[Self.V]) -> Self.V,
    ):
        var key_index = self._find_key_index(key)
        if key_index == 0:
            var value = update(None)
            self.put(key, value)
        else:
            key_index -= 1

            comptime if Self.destructive:
                if self._is_deleted(key_index):
                    self.count += 1
                    self._not_deleted(key_index)
                    self.values[key_index] = update(None)
                    return

            self.values[key_index] = update(self.values[key_index].copy())

    fn clear(mut self):
        self.values.clear()
        self.keys.clear()
        memset_zero(self.slot_to_index, self.capacity)

        comptime if Self.destructive:
            memset_zero(self.deleted_mask, self.capacity >> 3)
        self.count = 0

    @always_inline
    fn _find_key_index(self, key: StringSlice) -> Int:
        var key_hash = hash(key).cast[Self.KeyCountType]()
        var modulo_mask = self.capacity - 1

        var slot = Int(key_hash & Scalar[Self.KeyCountType](modulo_mask))
        while True:
            var key_index = Int(self.slot_to_index.load(slot))
            if key_index == 0:
                return key_index

            comptime if Self.caching_hashes:
                var other_key_hash = self.key_hashes[slot]
                if key_hash == other_key_hash:
                    var other_key = self.keys[key_index - 1]
                    if other_key == key:
                        return key_index
            else:
                var other_key = self.keys[key_index - 1]
                if other_key == key:
                    return key_index

            slot = (slot + 1) & modulo_mask


# ===-----------------------------------------------------------------------===#
# Benchmark Dict init
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_dict_init_with_short_keys[file_name: String](mut b: Bencher) raises:
    var keys = make_small_keys(file_name)

    @always_inline
    @parameter
    fn call_fn():
        var d = Dict[String, Int]()
        for i, key in enumerate(keys):
            d[key] = i
        keep(d._ctrl)

    b.iter[call_fn]()


@parameter
fn bench_dict_init_with_long_keys[file_name: String](mut b: Bencher) raises:
    var keys = make_long_keys(file_name)

    @always_inline
    @parameter
    fn call_fn():
        var d = Dict[String, Int, default_hasher]()
        for i, key in enumerate(keys):
            d[key] = i
        keep(d._ctrl)

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark StringDict init
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_dict_init_with_short_keys[
    file_name: String
](mut b: Bencher) raises:
    var keys = make_small_keys(file_name)

    @always_inline
    @parameter
    fn call_fn():
        var d = StringDict[Int]()
        for i, key in enumerate(keys):
            d.put(key, i)
        keep(d.keys.keys)

    b.iter[call_fn]()


@parameter
fn bench_string_dict_init_with_long_keys[
    file_name: String
](mut b: Bencher) raises:
    var keys = make_long_keys(file_name)

    @always_inline
    @parameter
    fn call_fn():
        var d = StringDict[Int]()
        for i, key in enumerate(keys):
            d.put(key, i)
        keep(d.keys.keys)

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Validate StringDict against Dict
# ===-----------------------------------------------------------------------===#


def validate_dicts(
    file_name: String = "UN_charter_EN.txt", small_keys: Bool = True
):
    var keys = make_small_keys(file_name) if small_keys else make_long_keys(
        file_name
    )
    print(
        "Number of keys:",
        len(keys),
        "small" if small_keys else "long",
        file_name,
    )
    var d = Dict[String, Int]()
    for i, key in enumerate(keys):
        d[key] = i

    var sd = StringDict[Int]()
    for i, key in enumerate(keys):
        sd.put(key, i)

    assert_equal(len(d), len(sd), "Length mismatch between Dict and StringDict")
    print("Length match between Dict and StringDict", len(d))


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    validate_dicts("UN_charter_EN.txt", small_keys=True)
    validate_dicts("UN_charter_EN.txt", small_keys=False)
    validate_dicts("UN_charter_AR.txt", small_keys=True)
    validate_dicts("UN_charter_AR.txt", small_keys=False)
    validate_dicts("UN_charter_ES.txt", small_keys=True)
    validate_dicts("UN_charter_ES.txt", small_keys=False)
    validate_dicts("UN_charter_RU.txt", small_keys=True)
    validate_dicts("UN_charter_RU.txt", small_keys=False)
    validate_dicts("UN_charter_zh-CN.txt", small_keys=True)
    validate_dicts("UN_charter_zh-CN.txt", small_keys=False)

    var m = Bench(
        BenchConfig(
            # out_file=_dir_of_current_file() / "bench_dict_string.csv",
            num_repetitions=5,
        )
    )
    m.bench_function[bench_dict_init_with_short_keys["UN_charter_EN.txt"]](
        BenchId("bench_dict_init_with_short_keys EN")
    )
    m.bench_function[bench_dict_init_with_short_keys["UN_charter_AR.txt"]](
        BenchId("bench_dict_init_with_short_keys AR")
    )
    m.bench_function[bench_dict_init_with_short_keys["UN_charter_ES.txt"]](
        BenchId("bench_dict_init_with_short_keys ES")
    )
    m.bench_function[bench_dict_init_with_short_keys["UN_charter_RU.txt"]](
        BenchId("bench_dict_init_with_short_keys RU")
    )
    m.bench_function[bench_dict_init_with_short_keys["UN_charter_zh-CN.txt"]](
        BenchId("bench_dict_init_with_short_keys zh-CN")
    )
    m.bench_function[bench_dict_init_with_long_keys["UN_charter_EN.txt"]](
        BenchId("bench_dict_init_with_long_keys EN")
    )
    m.bench_function[bench_dict_init_with_long_keys["UN_charter_AR.txt"]](
        BenchId("bench_dict_init_with_long_keys AR")
    )
    m.bench_function[bench_dict_init_with_long_keys["UN_charter_ES.txt"]](
        BenchId("bench_dict_init_with_long_keys ES")
    )
    m.bench_function[bench_dict_init_with_long_keys["UN_charter_RU.txt"]](
        BenchId("bench_dict_init_with_long_keys RU")
    )
    m.bench_function[bench_dict_init_with_long_keys["UN_charter_zh-CN.txt"]](
        BenchId("bench_dict_init_with_long_keys zh-CN")
    )

    m.bench_function[
        bench_string_dict_init_with_short_keys["UN_charter_EN.txt"]
    ](BenchId("bench_string_dict_init_with_short_keys EN"))
    m.bench_function[
        bench_string_dict_init_with_short_keys["UN_charter_AR.txt"]
    ](BenchId("bench_string_dict_init_with_short_keys AR"))
    m.bench_function[
        bench_string_dict_init_with_short_keys["UN_charter_ES.txt"]
    ](BenchId("bench_string_dict_init_with_short_keys ES"))
    m.bench_function[
        bench_string_dict_init_with_short_keys["UN_charter_RU.txt"]
    ](BenchId("bench_string_dict_init_with_short_keys RU"))
    m.bench_function[
        bench_string_dict_init_with_short_keys["UN_charter_zh-CN.txt"]
    ](BenchId("bench_string_dict_init_with_short_keys zh-CN"))
    m.bench_function[
        bench_string_dict_init_with_long_keys["UN_charter_EN.txt"]
    ](BenchId("bench_string_dict_init_with_long_keys EN"))
    m.bench_function[
        bench_string_dict_init_with_long_keys["UN_charter_AR.txt"]
    ](BenchId("bench_string_dict_init_with_long_keys AR"))
    m.bench_function[
        bench_string_dict_init_with_long_keys["UN_charter_ES.txt"]
    ](BenchId("bench_string_dict_init_with_long_keys ES"))
    m.bench_function[
        bench_string_dict_init_with_long_keys["UN_charter_RU.txt"]
    ](BenchId("bench_string_dict_init_with_long_keys RU"))
    m.bench_function[
        bench_string_dict_init_with_long_keys["UN_charter_zh-CN.txt"]
    ](BenchId("bench_string_dict_init_with_long_keys zh-CN"))

    m.dump_report()
