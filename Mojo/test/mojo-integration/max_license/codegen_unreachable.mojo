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

# RUN: mojo -debug-level full %s | FileCheck %s
# RUN: not mojo -debug-level full -D use_alias=0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-UNREACHABLE
# RUN: not mojo -debug-level full -D ENABLE_GPU_MALLOC=false %s 2>&1 | FileCheck %s --check-prefix=CHECK-GPU-MALLOC

from std.compile import compile_info
from std.memory.memory import _free, _malloc
from std.sys import (
    codegen_unreachable,
    get_defined_bool,
)
from std.sys.info import CompilationTarget
from max.gpu.host.info import _a100_target


def foo[T: Bool]() -> Int:
    return bar[T]()


def bar[T: Bool]() -> Int:
    codegen_unreachable[T, "this function can only be compile time evaluated"]()
    return 27


def test_malloc_no_recursing_parent(i: Int):
    test_malloc(i)


def test_malloc(i: Int):
    if i > 0:
        var ptr = _malloc[Int32](i)
        _free(ptr)
    else:
        # This runtime recursion creates a cycle in call instantiation graph
        # in the elaborator so that we can test out that error emitting can
        # handle this correctly.
        test_malloc(i + 2)


def test_gpu_malloc():
    _ = compile_info[
        test_malloc_no_recursing_parent,
        emission_kind="llvm",
        target=_a100_target,
    ]()


def main():
    comptime use_alias = get_defined_bool["use_alias", True]()

    comptime if use_alias:
        comptime t = foo[True]()
        # CHECK: 27
        print(t)
    else:
        var t = foo[True]()
        # CHECK-UNREACHABLE: this function can only be compile time evaluated
        print(t)

    # COM: check non-recursing parent also gets printed in the error.
    # CHECK-GPU-MALLOC: error: call expansion failed
    # CHECK-GPU-MALLOC:  test_malloc(i)
    # CHECK-GPU-MALLOC: runtime allocation on GPU not allowed
    test_gpu_malloc()
