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

# RUN: not %mojo %s 2>&1 | FileCheck %s


fn test_cannot_cast_between_different_types[T: AnyType](p: UnsafePointer[T]):
    pass


def main():
    var x = 42

    var p = UnsafePointer(to=x)
    # CHECK: invalid call to 'test_cannot_cast_between_different_types'
    test_cannot_cast_between_different_types[String](p)
