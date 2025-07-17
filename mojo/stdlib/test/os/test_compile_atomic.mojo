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

from os import Atomic

from compile import compile_info
from testing import assert_true


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


def main():
    test_compile_atomic()
