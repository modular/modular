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

from collections import BitSet
from random import *

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep


alias INIT_LOOP_SIZE: UInt = 1000000
"""Bench loop size for BitSet init tests."""

alias OP_LOOP_SIZE: UInt = 1000
"""Bench loop size for BitSet operation tests."""


@parameter
fn bench_empty_bitset_init[size: Int](mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        for _ in range(0, INIT_LOOP_SIZE):
            var b = BitSet[size]()
            keep(len(b))

    b.iter[call_fn]()


@parameter
fn bench_bitset_init_from[width: Int](mut b: Bencher) raises:
    var initial = SIMD[DType.bool, width](fill=True)

    @__copy_capture(initial)
    @always_inline
    @parameter
    fn call_fn():
        for _ in range(0, INIT_LOOP_SIZE):
            var b = BitSet(initial)
            keep(len(b))

    b.iter[call_fn]()


@parameter
fn bench_bitset_set[size: Int](mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        var bitset = BitSet[size]()
        for _ in range(0, OP_LOOP_SIZE):

            @parameter
            for i in range(0, bitset.size):
                bitset.set(i)
        keep(len(bitset))

    b.iter[call_fn]()


@parameter
fn bench_bitset_clear[width: Int](mut b: Bencher) raises:
    var initial = SIMD[DType.bool, width](fill=True)

    @__copy_capture(initial)
    @always_inline
    @parameter
    fn call_fn() raises:
        var bitset = BitSet[width](initial)
        for _ in range(0, OP_LOOP_SIZE):

            @parameter
            for i in range(0, bitset.size):
                bitset.clear(i)

        keep(len(bitset))

    b.iter[call_fn]()


@parameter
fn bench_bitset_toggle[width: Int](mut b: Bencher) raises:
    var initial = SIMD[DType.bool, width](fill=True)

    @__copy_capture(initial)
    @always_inline
    @parameter
    fn call_fn() raises:
        var bitset = BitSet[width](initial)
        for _ in range(0, OP_LOOP_SIZE):

            @parameter
            for i in range(0, bitset.size):
                bitset.toggle(i)

        keep(len(bitset))

    b.iter[call_fn]()


@parameter
fn bench_bitset_test[width: Int](mut b: Bencher) raises:
    var initial = SIMD[DType.bool, width](fill=True)

    @__copy_capture(initial)
    @always_inline
    @parameter
    fn call_fn() raises:
        var bitset = BitSet[width](initial)
        for _ in range(0, OP_LOOP_SIZE):

            @parameter
            for i in range(0, bitset.size):
                keep(bitset.test(i))

    b.iter[call_fn]()


@parameter
fn bench_bitset_union[width: Int](mut b: Bencher) raises:
    var lhs_init = SIMD[DType.bool, width](True)
    var rhs_init = SIMD[DType.bool, width](False)

    @__copy_capture(lhs_init)
    @__copy_capture(rhs_init)
    @always_inline
    @parameter
    fn call_fn() raises:
        var lhs = BitSet[width](lhs_init)
        var rhs = BitSet[width](rhs_init)

        for _ in range(0, OP_LOOP_SIZE):
            var new = lhs.union(rhs)
            keep(len(new))

    b.iter[call_fn]()


@parameter
fn bench_bitset_intersection[width: Int](mut b: Bencher) raises:
    var lhs_init = SIMD[DType.bool, width](True)
    var rhs_init = SIMD[DType.bool, width](False)

    @__copy_capture(lhs_init)
    @__copy_capture(rhs_init)
    @always_inline
    @parameter
    fn call_fn() raises:
        var lhs = BitSet[width](lhs_init)
        var rhs = BitSet[width](rhs_init)

        for _ in range(0, OP_LOOP_SIZE):
            var new = lhs.intersection(rhs)
            keep(len(new))

    b.iter[call_fn]()


@parameter
fn bench_bitset_difference[width: Int](mut b: Bencher) raises:
    var lhs_init = SIMD[DType.bool, width](True)
    var rhs_init = SIMD[DType.bool, width](False)

    @__copy_capture(lhs_init)
    @__copy_capture(rhs_init)
    @always_inline
    @parameter
    fn call_fn() raises:
        var lhs = BitSet[width](lhs_init)
        var rhs = BitSet[width](rhs_init)

        for _ in range(0, OP_LOOP_SIZE):
            var new = lhs.difference(rhs)
            keep(len(new))

    b.iter[call_fn]()


def main():
    seed()
    alias widths = (1, 2, 4, 8, 16)
    alias sizes = (10, 30, 50, 100, 1000)
    var m = Bench(BenchConfig(num_repetitions=1))

    @parameter
    for i in range(len(sizes)):
        alias size = sizes[i]
        m.bench_function[bench_empty_bitset_init[size]](
            BenchId(String("bench_empty_bitset_init[", size, "]"))
        )

    @parameter
    for width_idx in range(0, len(widths)):
        alias width = widths[width_idx]
        m.bench_function[bench_bitset_init_from[width]](
            BenchId(String("bench_bitset_init_from[", width, "]"))
        )

    @parameter
    for width_idx in range(0, len(widths)):
        alias width = widths[width_idx]
        m.bench_function[bench_bitset_set[width]](
            BenchId(String("bench_bitset_set[", width, "]"))
        )

    @parameter
    for width_idx in range(0, len(widths)):
        alias width = widths[width_idx]
        m.bench_function[bench_bitset_clear[width]](
            BenchId(String("bench_bitset_clear[", width, "]"))
        )

    @parameter
    for width_idx in range(0, len(widths)):
        alias width = widths[width_idx]
        m.bench_function[bench_bitset_clear[width]](
            BenchId(String("bench_bitset_test[", width, "]"))
        )

    @parameter
    for width_idx in range(0, len(widths)):
        alias width = widths[width_idx]
        m.bench_function[bench_bitset_clear[width]](
            BenchId(String("bench_bitset_toggle[", width, "]"))
        )

    @parameter
    for width_idx in range(0, len(widths)):
        alias width = widths[width_idx]
        m.bench_function[bench_bitset_clear[width]](
            BenchId(String("bench_bitset_union[", width, "]"))
        )

    @parameter
    for width_idx in range(0, len(widths)):
        alias width = widths[width_idx]
        m.bench_function[bench_bitset_clear[width]](
            BenchId(String("bench_bitset_intersection[", width, "]"))
        )

    @parameter
    for width_idx in range(0, len(widths)):
        alias width = widths[width_idx]
        m.bench_function[bench_bitset_clear[width]](
            BenchId(String("bench_bitset_difference[", width, "]"))
        )

    m.dump_report()
