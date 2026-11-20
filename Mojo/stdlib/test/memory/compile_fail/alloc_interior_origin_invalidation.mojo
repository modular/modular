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

# A pointer or span vended by an allocation handle carries an interior origin,
# so consuming or reassigning the handle invalidates the view at compile time.

from std.memory import Layout, alloc, dealloc


def allocation_unsafe_span_then_reassign():
    var a = alloc(Layout[Int].single())
    a.unsafe_ptr().write(42)
    ref first = a.unsafe_span()[0]
    dealloc(a^)
    a = alloc(Layout[Int].single())
    # CHECK: alloc_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'a["element"]'
    print(first)
    dealloc(a^)


def allocation_unsafe_ptr_then_reassign():
    var a = alloc(Layout[Int].single())
    var ptr = a.unsafe_ptr()
    dealloc(a^)
    a = alloc(Layout[Int].single())
    # CHECK: alloc_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'a["element"]'
    ptr.write(1)
    dealloc(a^)


def allocation_storage_then_reassign():
    var a = alloc(Layout[Int](length=4))
    var storage = a.storage()
    dealloc(a^)
    a = alloc(Layout[Int](length=4))
    # CHECK: alloc_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'a["element"]'
    print(len(storage))
    dealloc(a^)


def managed_unsafe_ptr_then_reassign():
    var managed = alloc(Layout[Int].single()).into_managed()
    var ptr = managed.unsafe_ptr()
    managed = alloc(Layout[Int].single()).into_managed()
    # CHECK: alloc_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'managed["element"]'
    ptr.write(1)


def managed_unsafe_span_then_reassign():
    var managed = alloc(Layout[Int].single()).into_managed()
    var span = managed.unsafe_span()
    managed = alloc(Layout[Int].single()).into_managed()
    # CHECK: alloc_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'managed["element"]'
    print(len(span))


def thin_unsafe_ptr_then_reassign():
    var layout = Layout[Int].single()
    var thin = alloc(layout).into_thin()
    var ptr = thin.unsafe_ptr()
    dealloc(thin^.unsafe_with_layout(layout))
    thin = alloc(layout).into_thin()
    # CHECK: alloc_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'thin["element"]'
    ptr.write(1)
    dealloc(thin^.unsafe_with_layout(layout))


def main():
    allocation_unsafe_span_then_reassign()
    allocation_unsafe_ptr_then_reassign()
    allocation_storage_then_reassign()
    managed_unsafe_ptr_then_reassign()
    managed_unsafe_span_then_reassign()
    thin_unsafe_ptr_then_reassign()
