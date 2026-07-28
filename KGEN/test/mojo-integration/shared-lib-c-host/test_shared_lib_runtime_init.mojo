# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Tests that `runtime.initialize_runtime()` lets an exported function in a
# `--emit shared-lib` library use `parallelize` when the process main is a
# C program rather than Mojo (GEX-3993). Without the initialize_runtime()
# call, the async runtime is never created and the first parallelize
# segfaults on a null runtime.
#
# COM: For sake of keeping the test commands simple, this hard-codes the use
# COM:   of `.dylib`, even on platforms like Linux where .so would otherwise
# COM:   be used. dlopen takes a literal path, so the extension is cosmetic.
# RUN: mkdir -p %t.dir
# RUN: %mojo-build %s -o %t.dir/libtestlib.dylib --emit shared-lib
# RUN: $(dirname %s)/c_host %t.dir/libtestlib.dylib | FileCheck %s

# CHECK: sum1=499500
# CHECK: sum2=499500

from max.algorithm import parallelize
from std.runtime import initialize_runtime


@export("mojo_parallel_sum")
def mojo_parallel_sum(n: Int64) abi("C") -> Int64:
    initialize_runtime()

    var count = Int(n)
    var results = List[Int64](length=count, fill=0)

    @parameter
    def fill(i: Int):
        results[i] = Int64(i)

    parallelize[fill](count)

    var total = Int64(0)
    for r in results:
        total += r
    return total
