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

from os.atomic import Atomic, fence

from compile import compile_info
from testing import TestSuite, assert_false, assert_true


def test_compile_atomic():
    @parameter
    fn my_add_function[
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


def test_compile_fence():
    @parameter
    fn my_fence_function():
        fence[scope="agent"]()

    var asm = compile_info[my_fence_function, emission_kind="llvm"]()

    assert_true('fence syncscope("agent") seq_cst' in asm)


def test_compile_compare_exchange():
    fn my_cmpxchg_function(atom: Atomic[DType.int32, scope="agent"]) -> Bool:
        var expected = Int32(0)
        return atom.compare_exchange(expected, 42)

    var asm = compile_info[my_cmpxchg_function, emission_kind="llvm"]()

    assert_true(
        'cmpxchg ptr %3, i32 %4, i32 42 syncscope("agent") seq_cst seq_cst'
        in asm
    )
    assert_false("cmpxchg weak" in asm)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
