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


fn test_cannot_cast_between_different_address_spaces[
    T: AnyType
](p: UnsafePointer[T, address_space = AddressSpace(1), ...]):
    pass


def main():
    var x = 42

    var p = UnsafePointer(to=x).address_space_cast[AddressSpace(2)]()
    # CHECK: value passed to 'p' cannot be converted from 'UnsafePointer[Int, origin_of(x), address_space=AddressSpace(2)]' to 'UnsafePointer[T, p.origin, address_space=AddressSpace(1)]'
    test_cannot_cast_between_different_address_spaces(p)
