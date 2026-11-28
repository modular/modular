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
# A cycle of functions that cannot be split must still be emitted when the
# module is split per function.
#
# Two exports make the compiler outline `walk`, which both of them reach, into
# a module of its own. `walk` then uses a private switch table, which makes it
# unable to split, and it calls `back`, which calls it back. Neither `walk` nor
# `back` has a user outside the cycle, so before the per-function splitter
# rooted such cycles, neither ended up in any split. The object kept an
# undefined reference to `walk`, which failed the link of a dynamic library.
#
# RUN: mkdir -p %t.dir
# RUN: %mojo-build %s -o %t.dir/output.o --emit object
# RUN: llvm-nm -U %t.dir/output.o | FileCheck %s

# CHECK-DAG: T {{_?lib_a}}
# CHECK-DAG: T {{_?lib_b}}
# CHECK-DAG: {{[TWt]}} {{_?.*walk.*}}


@no_inline
def walk(n: Int) -> Int:
    if n <= 0:
        var k = n & 3
        if k == 0:
            return 11
        elif k == 1:
            return 23
        elif k == 2:
            return 37
        return 41
    return back(n - 1) + leaf_a(n) + leaf_b(n)


@no_inline
def back(n: Int) -> Int:
    return walk(n - 1) * 3


@no_inline
def leaf_a(n: Int) -> Int:
    return n * 5 + 1


@no_inline
def leaf_b(n: Int) -> Int:
    return n * 7 + 2


@export("lib_a")
def lib_a(n: Int) abi("C") -> Int:
    return walk(n)


@export("lib_b")
def lib_b(n: Int) abi("C") -> Int:
    return walk(n + 1)
