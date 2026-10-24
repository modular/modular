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
#
# A runtime integer exponent's sign is not known until it is evaluated, so the
# guard is a trap rather than a compile error. It aborts, so verify it with
# FileCheck rather than a regular unit test.
#
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: test_simd_length_pow_negative_exponent_aborts
def main():
    print("== test_simd_length_pow_negative_exponent_aborts")
    var a = SIMDLength(2)
    var b = SIMDLength(-1)
    # CHECK: an integer cannot be raised to a negative power
    var n = a**b
    # CHECK-NOT: is never reached
    print("is never reached", Int(n))
