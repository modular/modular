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
# The IR dump is produced only when the pass pipeline actually runs, and a
# warm KGEN transform cache serves the result without running it, so give
# the run an empty cache of its own. `raise-for-loops` runs on each function,
# so its dumps come out in completion order while the checks below expect
# source order: run the pipeline single-threaded.
# RUN: rm -rf %t.cache && env MODULAR_CACHE_DIR=%t.cache kgen --emit=llvm %s \
# RUN:   -mllvm --mlir-print-ir-after=raise-for-loops \
# RUN:   -mllvm --mlir-disable-threading -o %t.ll 2>&1 | FileCheck %s

# Verify that mojo's `for i in range(...)` is raised from `hlcf.loop` to
# `hlcf.for`. `range(end)` uses the `_ZeroStartingRange` iterator and
# `range(start, end)` the `_SequentialRange` one; both count up, so they raise
# to a half-open `slt add` loop. The strided range (with a runtime stride)
# goes through `_StridedRange`, whose MLIR shape `inferLoopCount` does not
# match, so it stays as `hlcf.loop`.


# CHECK-LABEL: kgen.func export @loop_runtime_ub(
# CHECK: hlcf.for [{{.*}} to {{.*}} step {{.*}} slt add]
@export
def loop_runtime_ub(end: Int) -> Int:
    var x: Int = 0
    for i in range(end):
        x += i
    return x


# CHECK-LABEL: kgen.func export @loop_runtime_lb(
# CHECK: hlcf.for [{{.*}} to {{.*}} step {{.*}} slt add]
@export
def loop_runtime_lb(start: Int) -> Int:
    var x: Int = 0
    for i in range(start, 100):
        x += i
    return x


# CHECK-LABEL: kgen.func export @loop_runtime_stride(
# CHECK: hlcf.loop
@export
def loop_runtime_stride(step: Int) -> Int:
    var x: Int = 0
    for i in range(0, 100, step):
        x += i
    return x
