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

from std.atomic import Atomic, Ordering, fence

from std.compile import compile_info
from std.testing import TestSuite, assert_false, assert_true


def test_compile_atomic() raises:
    @parameter
    def my_add_function[
        dtype: DType
    ](mut x: Atomic[dtype, scope="agent"]) -> Scalar[dtype]:
        return x.fetch_add(1)

    var asm = compile_info[
        my_add_function[DType.float32], emission_kind="llvm"
    ]()

    assert_true(
        'atomicrmw fadd ptr %2, float 1.000000e+00 syncscope("agent") seq_cst'
        in asm
    )


def test_compile_fence() raises:
    @parameter
    def my_fence_function():
        fence[scope="agent"]()

    var asm = compile_info[my_fence_function, emission_kind="llvm"]()

    assert_true('fence syncscope("agent") seq_cst' in asm)


def test_compile_compare_exchange() raises:
    def my_cmpxchg_function(
        mut atm: Atomic[DType.int32, scope="agent"]
    ) -> Bool:
        var expected = Int32(0)
        return atm.compare_exchange(expected, 42)

    var asm = compile_info[my_cmpxchg_function, emission_kind="llvm"]()

    assert_true(
        'cmpxchg ptr %3, i32 %4, i32 42 syncscope("agent") seq_cst seq_cst'
        in asm
    )
    assert_false("cmpxchg weak" in asm)


def test_compile_store() raises:
    def my_store_function(
        mut atm: Atomic[DType.int32, scope="agent"], v: Int32
    ):
        Atomic[DType.int32, scope="agent"].store[ordering=Ordering.RELEASE](
            UnsafePointer(to=atm.value), v
        )

    var asm = compile_info[my_store_function, emission_kind="llvm"]()

    assert_true('store atomic i32 %1, ptr %3 syncscope("agent") release' in asm)
    assert_false("atomicrmw xchg" in asm)


def test_compile_store_default_scope() raises:
    def my_store_function(mut atm: Atomic[DType.int64], v: Int64):
        Atomic[DType.int64].store[ordering=Ordering.RELEASE](
            UnsafePointer(to=atm.value), v
        )

    var asm = compile_info[my_store_function, emission_kind="llvm"]()

    assert_true("store atomic i64 %1, ptr %3 release" in asm)
    assert_false("syncscope" in asm)
    assert_false("atomicrmw xchg" in asm)


def test_compile_xchg() raises:
    def my_xchg_function(
        mut atm: Atomic[DType.int32, scope="agent"], v: Int32
    ) -> Int32:
        return Atomic[DType.int32, scope="agent"]._xchg[
            ordering=Ordering.SEQUENTIAL
        ](UnsafePointer(to=atm.value), v)

    var asm = compile_info[my_xchg_function, emission_kind="llvm"]()

    assert_true(
        'atomicrmw xchg ptr %3, i32 %1 syncscope("agent") seq_cst' in asm
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
