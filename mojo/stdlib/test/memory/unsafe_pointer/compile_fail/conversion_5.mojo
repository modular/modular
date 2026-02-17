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


fn test_cannot_cast_from_mutable_any_to_named[
    T: AnyType, mut: Bool, //, origin: Origin[mut=mut]
](p: UnsafePointer[T, origin]):
    pass


def main():
    var x = 42

    var p = UnsafePointer(to=x).as_any_origin()
    # CHECK: value passed to 'p' cannot be converted from 'UnsafePointer[Int, MutAnyOrigin]' to 'UnsafePointer[T, origin_of(x)]'
    test_cannot_cast_from_mutable_any_to_named[origin_of(x)](p)
