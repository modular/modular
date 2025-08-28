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

from collections.dict import DictEntry
from hashlib import Hasher
from random.random import random_si64, seed
from sys import size_of

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
fn make_dict[size: Int, *, random: Bool = False]() -> Dict[Int, Int]:
    var d = Dict[Int, Int]()
    for i in range(0, size):

        @parameter
        if random:
            d[i] = Int(random_si64(0, size))
        else:
            d[i] = i
    return d


# ===-----------------------------------------------------------------------===#
# Benchmark Dict init
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_dict_init(mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        for _ in range(1000):
            var d = Dict[Int, Int]()
            keep(d._entries.unsafe_ptr())
            keep(d._index.data)

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Insert
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_dict_insert[size: Int](mut b: Bencher) raises:
    """Insert 10 new items 100_000 times."""
    var items = make_dict[size]()

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(10_000):
            for key in range(size, size + 10):
                items[key] = Int(random_si64(0, size))

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Lookup
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_dict_lookup[size: Int](mut b: Bencher) raises:
    """Lookup 10 items 100_000 times."""
    var items = make_dict[size]()

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(10_000):
            for key in range(10):
                var res = items[key]
                keep(res)

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark Dict contains
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_dict_contains[size: Int](mut b: Bencher) raises:
    """Check if the dict contains 10 keys 100_000 times."""
    var items = make_dict[size]()

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(100_000):
            for key in range(10):
                var res = key in items
                keep(res)

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Memory Footprint
# ===-----------------------------------------------------------------------===#


fn total_bytes_used[H: Hasher](items: Dict[Int, Int, H]) -> Int:
    # the allocated memory by entries:
    var entry_size = size_of[Optional[DictEntry[Int, Int, H]]]()
    var amnt_bytes = items._entries.capacity * entry_size
    amnt_bytes += size_of[Dict[Int, Int]]()

    # the allocated memory by index table:
    var reserved = items._reserved()
    if reserved <= 128:
        amnt_bytes += size_of[Int8]() * reserved
    elif reserved <= 2**16 - 2:
        amnt_bytes += size_of[Int16]() * reserved
    elif reserved <= 2**32 - 2:
        amnt_bytes += size_of[Int32]() * reserved
    else:
        amnt_bytes += size_of[Int64]() * reserved

    return amnt_bytes


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=5))
    m.bench_function[bench_dict_init](BenchId("bench_dict_init"))
    alias sizes = (10, 30, 50, 100, 1000, 10_000, 100_000, 1_000_000)

    @parameter
    for i in range(len(sizes)):
        alias size = sizes[i]
        m.bench_function[bench_dict_insert[size]](
            BenchId(String("bench_dict_insert[", size, "]"))
        )
        m.bench_function[bench_dict_lookup[size]](
            BenchId(String("bench_dict_lookup[", size, "]"))
        )
        m.bench_function[bench_dict_contains[size]](
            BenchId(String("bench_dict_contains[", size, "]"))
        )

    results = Dict[String, (Float64, Int)]()
    for info in m.info_vec:
        n = info.name
        time = info.result.mean("ms")
        avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = ((avg * amnt + time) / (amnt + 1), amnt + 1)
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0], sep=",")

    @parameter
    for i in range(len(sizes)):
        alias size = sizes[i]
        var mem_s = total_bytes_used(make_dict[size]())
        print("dict_memory_size[", size, "]: ", mem_s, sep="")
