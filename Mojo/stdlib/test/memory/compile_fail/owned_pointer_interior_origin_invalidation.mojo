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

# A pointer vended by `OwnedPointer.ptr()` carries an interior origin, so
# reassigning or consuming the `OwnedPointer` invalidates it at compile time.

from std.memory import OwnedPointer


def ptr_then_reassign():
    var owned = OwnedPointer(String("hello"))
    var p = owned.ptr()
    owned = OwnedPointer(String("goodbye"))
    # CHECK: owned_pointer_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'owned["value"]'
    print(p[])


def ptr_then_into_inner_and_reinit():
    var consumed = OwnedPointer(String("hello"))
    var p = consumed.ptr()
    var s = consumed^.into_inner()
    consumed = OwnedPointer(String("goodbye"))
    # CHECK: owned_pointer_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'consumed["value"]'
    print(p[], s)


def main():
    ptr_then_reassign()
    ptr_then_into_inner_and_reinit()
