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

from compile.reflection import get_linkage_name
from testing import assert_equal


fn foo():
    pass


fn bar(x: Int) -> Int:
    return 1


# NOTE: this is intentionally in the middle here, to ensure that the intrinsic
# correctly resolves signatures that are declared after the call.
alias funcs = __functions_in_module()


def main():
    var expected_names = [
        "test_functions_in_module::foo()",
        "test_functions_in_module::bar(::Int)",
        "test_functions_in_module::bar(::Int,::Int)",
        "test_functions_in_module::foobar(::SIMD[::DType(float64), ::Int(1)])",
    ]

    @parameter
    for i in range(len(funcs)):
        alias name = get_linkage_name[funcs[i]]()
        assert_equal(name, expected_names[i])


fn bar(y: Int, z: Int):
    pass


def foobar(*, z: Float64 = 1.6):
    pass
