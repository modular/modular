# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max, factorial, div_ceil, min
from time import now
from utils.index import StaticIntTuple
from algorithm import parallelize, async_parallelize
from runtime.llcl import num_cores, Runtime, OwningOutputChainPtr
from collections.vector import UnsafeFixedVector

alias n = 12


@always_inline
fn swap[type: AnyRegType](inout lhs: type, inout rhs: type):
    let tmp = lhs
    lhs = rhs
    rhs = tmp


@always_inline
fn perm(i0: Int) -> StaticIntTuple[n]:
    var p = StaticIntTuple[n]()
    var i = i0

    for k in range(n):
        let f = factorial(n - 1 - k)
        p[k] = i // f
        i = i % f

    for k in range(n - 1, -1, -1):
        for j in range(k - 1, -1, -1):
            if p[j] <= p[k]:
                p[k] += 1

    return p


fn main():
    let core_count = 8  # num_cores()

    let t0 = now()

    var max_vals = UnsafeFixedVector[Int](core_count)
    for i in range(max_vals.capacity):
        max_vals.append(0)

    let num_work_items = factorial(n)

    let chunk_size = max(div_ceil(num_work_items, core_count), 1)

    @parameter
    @always_inline
    fn _do_parallel(thread_idx: Int):
        var max_flips = 0

        for idx in range(
            chunk_size * thread_idx,
            min(chunk_size * (thread_idx + 1), num_work_items),
        ):
            var p = perm(idx)
            var flips = 0
            var k = p[0]

            while k:
                var i = 0
                var j = k
                while i < j:
                    swap(p[i], p[j])
                    i += 1
                    j -= 1

                k = p[0]
                flips += 1
            max_flips = max(max_flips, flips)
        max_vals[thread_idx] = max_flips

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        async_parallelize[_do_parallel](out_chain.borrow(), core_count)
        out_chain.wait()

    var max_flips = max_vals[0]
    for i in range(len(max_vals)):
        # print("i = ", i, " max_vals[i] = ", max_vals[i])
        max_flips = max(max_flips, max_vals[i])

    print("fannkuchen(", n, ") = ", max_flips)
    let t1 = now()
    print((t1 - t0) / 1e9)
