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

# RUN: not %mojo %s 2>&1 | FileCheck %s


fn test_cannot_cast_immutable_to_mutable[
    T: AnyType
](p: UnsafePointer[mut=True, T, ...]):
    pass


def main():
    var x = 42

    var p = UnsafePointer(to=x).as_immutable()
    # FIXME: this probably should be a parameter inference error, not the type
    # conversion error.

    # CHECK: invalid call to 'test_cannot_cast_immutable_to_mutable': value passed to 'p' cannot be converted from 'UnsafePointer[Int, {{.*}}]' to 'UnsafePointer[T, {{.*}}]'
    test_cannot_cast_immutable_to_mutable(p)
