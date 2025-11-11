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


fn test_cannot_cast_immutable_any_to_mutable_any[
    T: AnyType
](p: UnsafePointer[T, MutAnyOrigin, **_]):
    pass


def main():
    var x = 42

    var p = UnsafePointer(to=x).as_immutable().as_any_origin()
    # CHECK: constraint failed: Invalid UnsafePointer conversion from immutable to mutable
    test_cannot_cast_immutable_any_to_mutable_any(p)
