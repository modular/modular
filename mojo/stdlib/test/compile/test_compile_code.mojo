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

from compile import compile_info
from testing import *


def test_compile_llvm():
    @parameter
    fn my_add_function[
        type: DType, size: Int
    ](x: SIMD[type, size], y: SIMD[type, size]) -> SIMD[type, size]:
        return x + y

    alias func = my_add_function[DType.float32, 4]
    var asm = compile_info[func, emission_kind="llvm"]()

    assert_true("fadd" in asm)


def main():
    test_compile_llvm()
