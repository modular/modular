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


def main():
    var a_int: Int = -7
    var b_int: Int = 4
    sum_int = a_int + b_int  # Result is type Int
    print("Int sum:", sum_int)

    var i_uint: UInt = 9
    var j_uint: UInt = 8
    sum_uint = i_uint + j_uint  # Result is type UInt
    print("UInt sum:", sum_uint)

    sum_mixed = a_int + i_uint  # Result is type Int
    print("Mixed sum:", sum_mixed)

    quotient_int = a_int / b_int  # Result is type Float64
    print("Int quotient:", quotient_int)
    quotient_uint = i_uint / j_uint  # Result is type Float64
    print("UInt quotient:", quotient_uint)
