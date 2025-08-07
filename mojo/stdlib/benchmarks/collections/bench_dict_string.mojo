# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    Unit,
    Format,
    keep,
    run,
)
from collections.string.string_slice import _to_string_list
from hashlib import default_comp_time_hasher, default_hasher
from memory import memcpy, memset_zero
from os import abort
from pathlib import _dir_of_current_file
from sys import stderr
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
    return abort[List[String]]()


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
    return abort[List[String]]()


# ===-----------------------------------------------------------------------===#
# String Dict implementation for benchmarking baseline against Dict
# ===-----------------------------------------------------------------------===#

from bit import pop_count, bit_width


struct KeysContainer[KeyEndType: DType = DType.uint32](Sized):
    var keys: UnsafePointer[UInt8]
    var allocated_bytes: Int
    var keys_end: UnsafePointer[Scalar[KeyEndType]]
    var count: Int
    var capacity: Int

    fn __init__(out self, capacity: Int):
        constrained[
            KeyEndType == DType.uint8
            or KeyEndType == DType.uint16
            or KeyEndType == DType.uint32
            or KeyEndType == DType.uint64,
            "KeyEndType needs to be an unsigned integer",
        ]()
        self.allocated_bytes = capacity << 3
        self.keys = UnsafePointer[UInt8].alloc(self.allocated_bytes)
        self.keys_end = UnsafePointer[SIMD[KeyEndType, 1]].alloc(capacity)
        self.count = 0
        self.capacity = capacity

    fn __copyinit__(out self, existing: Self):
        self.allocated_bytes = existing.allocated_bytes
        self.count = existing.count
        self.capacity = existing.capacity
        self.keys = UnsafePointer[UInt8].alloc(self.allocated_bytes)
        memcpy(self.keys, existing.keys, self.allocated_bytes)
        self.keys_end = UnsafePointer[Scalar[KeyEndType]].alloc(self.capacity)
        memcpy(self.keys_end, existing.keys_end, self.capacity)

    fn __moveinit__(out self, deinit existing: Self):
        self.allocated_bytes = existing.allocated_bytes
        self.count = existing.count
        self.capacity = existing.capacity
        self.keys = existing.keys
        self.keys_end = existing.keys_end

    fn __del__(deinit self):
        self.keys.free()
        self.keys_end.free()

    @always_inline
    fn add(mut self, key: StringSlice):
        var prev_end = 0 if self.count == 0 else self.keys_end[self.count - 1]
        var key_length = len(key)
        var new_end = prev_end + key_length

        var needs_realocation = False
        while new_end > self.allocated_bytes:
            self.allocated_bytes += self.allocated_bytes >> 1
            needs_realocation = True

        if needs_realocation:
            var keys = UnsafePointer[UInt8].alloc(self.allocated_bytes)
            memcpy(keys, self.keys, Int(prev_end))
            self.keys.free()
            self.keys = keys

        memcpy(
            self.keys.offset(prev_end),
            UnsafePointer(key.unsafe_ptr()),
            key_length,
        )
        var count = self.count + 1
        if count >= self.capacity:
            var new_capacity = self.capacity + (self.capacity >> 1)
            var keys_end = UnsafePointer[SIMD[KeyEndType, 1]].alloc(
                new_capacity
            )
            memcpy(keys_end, self.keys_end, self.capacity)
            self.keys_end.free()
            self.keys_end = keys_end
            self.capacity = new_capacity

        self.keys_end.store(self.count, new_end)
        self.count = count

    @always_inline
    fn get(self, index: Int) -> StringSlice[ImmutableAnyOrigin]:
        if index < 0 or index >= self.count:
            return StringSlice(unsafe_from_utf8=Span(ptr=self.keys, length=0))
        var start = 0 if index == 0 else Int(self.keys_end[index - 1])
        var length = Int(self.keys_end[index]) - start
        return StringSlice(
            unsafe_from_utf8=Span(ptr=self.keys.offset(start), length=length)
        )

    @always_inline
    fn clear(mut self):
        self.count = 0

    @always_inline
    fn __getitem__(self, index: Int) -> StringSlice[ImmutableAnyOrigin]:
        return self.get(index)

    @always_inline
    fn __len__(self) -> Int:
        return self.count

    fn keys_vec(self) -> List[StringSlice[ImmutableAnyOrigin]]:
        var keys = List[StringSlice[ImmutableAnyOrigin]](capacity=self.count)
        for i in range(self.count):
            keys.append(self[i])
        return keys

    fn print_keys(self):
        print("(" + String(self.count) + ")[", end="")
        for i in range(self.count):
            var end = ", " if i < self.count - 1 else ""
            print(self[i], end=end)
        print("]")


struct StringDict[
    V: Copyable & Movable,
    KeyCountType: DType = DType.uint32,
    KeyOffsetType: DType = DType.uint32,
    destructive: Bool = True,
    caching_hashes: Bool = True,
](Sized):
    var keys: KeysContainer[KeyOffsetType]
    var key_hashes: UnsafePointer[Scalar[KeyCountType]]
    var values: List[V]
    var slot_to_index: UnsafePointer[Scalar[KeyCountType]]
    var deleted_mask: UnsafePointer[UInt8]
    var count: Int
    var capacity: Int

    fn __init__(out self, capacity: Int = 16):
        constrained[
            KeyCountType == DType.uint8
            or KeyCountType == DType.uint16
            or KeyCountType == DType.uint32
            or KeyCountType == DType.uint64,
            "KeyCountType needs to be an unsigned integer",
        ]()
        self.count = 0
        if capacity <= 8:
            self.capacity = 8
        else:
            var icapacity = Int64(capacity)
            self.capacity = capacity if pop_count(icapacity) == 1 else 1 << Int(
                bit_width(icapacity)
            )
        self.keys = KeysContainer[KeyOffsetType](capacity)

        @parameter
        if caching_hashes:
            self.key_hashes = UnsafePointer[Scalar[KeyCountType]].alloc(
                self.capacity
            )
        else:
            self.key_hashes = UnsafePointer[Scalar[KeyCountType]].alloc(0)
        self.values = List[V](capacity=capacity)
        self.slot_to_index = UnsafePointer[Scalar[KeyCountType]].alloc(
            self.capacity
        )
        memset_zero(self.slot_to_index, self.capacity)

        @parameter
        if destructive:
            self.deleted_mask = UnsafePointer[UInt8].alloc(self.capacity >> 3)
            memset_zero(self.deleted_mask, self.capacity >> 3)
        else:
            self.deleted_mask = UnsafePointer[UInt8].alloc(0)

    fn __copyinit__(out self, existing: Self):
        self.count = existing.count
        self.capacity = existing.capacity
        self.keys = existing.keys

        @parameter
        if caching_hashes:
            self.key_hashes = UnsafePointer[Scalar[KeyCountType]].alloc(
                self.capacity
            )
            memcpy(self.key_hashes, existing.key_hashes, self.capacity)
        else:
            self.key_hashes = UnsafePointer[Scalar[KeyCountType]].alloc(0)
        self.values = existing.values
        self.slot_to_index = UnsafePointer[Scalar[KeyCountType]].alloc(
            self.capacity
        )
        memcpy(self.slot_to_index, existing.slot_to_index, self.capacity)

        @parameter
        if destructive:
            self.deleted_mask = UnsafePointer[UInt8].alloc(self.capacity >> 3)
            memcpy(self.deleted_mask, existing.deleted_mask, self.capacity >> 3)
        else:
            self.deleted_mask = UnsafePointer[UInt8].alloc(0)

    fn __moveinit__(out self, deinit existing: Self):
        self.count = existing.count
        self.capacity = existing.capacity
        self.keys = existing.keys^
        self.key_hashes = existing.key_hashes
        self.values = existing.values^
        self.slot_to_index = existing.slot_to_index
        self.deleted_mask = existing.deleted_mask

    fn __del__(deinit self):
        self.slot_to_index.free()
        self.deleted_mask.free()
        self.key_hashes.free()

    fn __len__(self) -> Int:
        return self.count

    @always_inline
    fn __contains__(self, key: StringSlice) -> Bool:
        return self._find_key_index(key) != 0

    fn put(mut self, key: StringSlice, value: V):
        if self.count >= self.capacity - (self.capacity >> 3):
            self._rehash()

        var key_hash = hash(key).cast[KeyCountType]()
        var modulo_mask = self.capacity - 1
        var slot = Int(key_hash & modulo_mask)
        while True:
            var key_index = Int(self.slot_to_index.load(slot))
            if key_index == 0:
                self.keys.add(key)

                @parameter
                if caching_hashes:
                    self.key_hashes.store(slot, key_hash)
                self.values.append(value)
                self.count += 1
                self.slot_to_index.store(
                    slot, SIMD[KeyCountType, 1](self.keys.count)
                )
                return

            @parameter
            if caching_hashes:
                var other_key_hash = self.key_hashes[slot]
                if other_key_hash == key_hash:
                    var other_key = self.keys[key_index - 1]
                    if other_key == key:
                        self.values[key_index - 1] = value  # replace value

                        @parameter
                        if destructive:
                            if self._is_deleted(key_index - 1):
                                self.count += 1
                                self._not_deleted(key_index - 1)
                        return
            else:
                var other_key = self.keys[key_index - 1]
                if other_key == key:
                    self.values[key_index - 1] = value  # replace value

                    @parameter
                    if destructive:
                        if self._is_deleted(key_index - 1):
                            self.count += 1
                            self._not_deleted(key_index - 1)
                    return

            slot = (slot + 1) & modulo_mask

    @always_inline
    fn _is_deleted(self, index: Int) -> Bool:
        var offset = index >> 3
        var bit_index = index & 7
        return self.deleted_mask.offset(offset).load() & (1 << bit_index) != 0

    @always_inline
    fn _deleted(self, index: Int):
        var offset = index >> 3
        var bit_index = index & 7
        var p = self.deleted_mask.offset(offset)
        var mask = p.load()
        p.store(mask | (1 << bit_index))

    @always_inline
    fn _not_deleted(self, index: Int):
        var offset = index >> 3
        var bit_index = index & 7
        var p = self.deleted_mask.offset(offset)
        var mask = p.load()
        p.store(mask & ~(1 << bit_index))

    @always_inline
    fn _rehash(mut self):
        var old_slot_to_index = self.slot_to_index
        var old_capacity = self.capacity
        self.capacity <<= 1
        var mask_capacity = self.capacity >> 3
        self.slot_to_index = UnsafePointer[Scalar[KeyCountType]].alloc(
            self.capacity
        )
        memset_zero(self.slot_to_index, self.capacity)

        var key_hashes = self.key_hashes

        @parameter
        if caching_hashes:
            key_hashes = UnsafePointer[Scalar[KeyCountType]].alloc(
                self.capacity
            )

        @parameter
        if destructive:
            var deleted_mask = UnsafePointer[UInt8].alloc(mask_capacity)
            memset_zero(deleted_mask, mask_capacity)
            memcpy(deleted_mask, self.deleted_mask, old_capacity >> 3)
            self.deleted_mask.free()
            self.deleted_mask = deleted_mask

        var modulo_mask = self.capacity - 1
        for i in range(old_capacity):
            if old_slot_to_index[i] == 0:
                continue
            var key_hash = SIMD[KeyCountType, 1](0)

            @parameter
            if caching_hashes:
                key_hash = self.key_hashes[i]
            else:
                key_hash = hash(self.keys[Int(old_slot_to_index[i] - 1)]).cast[
                    KeyCountType
                ]()

            var slot = Int(key_hash & modulo_mask)

            # var searching = True
            while True:
                var key_index = Int(self.slot_to_index.load(slot))

                if key_index == 0:
                    self.slot_to_index.store(slot, old_slot_to_index[i])
                    break
                    # searching = False

                else:
                    slot = (slot + 1) & modulo_mask

            @parameter
            if caching_hashes:
                key_hashes[slot] = key_hash

        @parameter
        if caching_hashes:
            self.key_hashes.free()
            self.key_hashes = key_hashes
        old_slot_to_index.free()

    fn get(self, key: StringSlice, default: V) -> V:
        var key_index = self._find_key_index(key)
        if key_index == 0:
            return default

        @parameter
        if destructive:
            if self._is_deleted(key_index - 1):
                return default
        return self.values[key_index - 1]

    fn delete(mut self, key: StringSlice):
        @parameter
        if not destructive:
            return

        var key_index = self._find_key_index(key)
        if key_index == 0:
            return
        if not self._is_deleted(key_index - 1):
            self.count -= 1
        self._deleted(key_index - 1)

    fn upsert(mut self, key: StringSlice, update: fn (value: Optional[V]) -> V):
        var key_index = self._find_key_index(key)
        if key_index == 0:
            var value = update(None)
            self.put(key, value)
        else:
            key_index -= 1

            @parameter
            if destructive:
                if self._is_deleted(key_index):
                    self.count += 1
                    self._not_deleted(key_index)
                    self.values[key_index] = update(None)
                    return

            self.values[key_index] = update(self.values[key_index])

    fn clear(mut self):
        self.values.clear()
        self.keys.clear()
        memset_zero(self.slot_to_index, self.capacity)

        @parameter
        if destructive:
            memset_zero(self.deleted_mask, self.capacity >> 3)
        self.count = 0

    @always_inline
    fn _find_key_index(self, key: StringSlice) -> Int:
        var key_hash = hash(key).cast[KeyCountType]()
        var modulo_mask = self.capacity - 1

        var slot = Int(key_hash & modulo_mask)
        while True:
            var key_index = Int(self.slot_to_index.load(slot))
            if key_index == 0:
                return key_index

            @parameter
            if caching_hashes:
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
        for i in range(len(keys)):
            d[keys[i]] = i
        keep(d._entries.unsafe_ptr())

    b.iter[call_fn]()


@parameter
fn bench_dict_init_with_long_keys[file_name: String](mut b: Bencher) raises:
    var keys = make_long_keys(file_name)

    @always_inline
    @parameter
    fn call_fn():
        var d = Dict[String, Int, default_hasher]()
        for i in range(len(keys)):
            d[keys[i]] = i
        keep(d._entries.unsafe_ptr())

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
        for i in range(len(keys)):
            d.put(keys[i], i)
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
        for i in range(len(keys)):
            d.put(keys[i], i)
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
    for i in range(len(keys)):
        d[keys[i]] = i

    var sd = StringDict[Int]()
    for i in range(len(keys)):
        sd.put(keys[i], i)

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
