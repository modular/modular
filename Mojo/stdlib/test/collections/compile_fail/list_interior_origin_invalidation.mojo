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

# The pointer from `List.unsafe_ptr()` carries the list's interior origin. A
# `mut self` method or a reassignment can move the list's storage, so it
# invalidates the pointer at compile time.


def unsafe_ptr_then_append():
    var appended: List = [1]
    var ptr = appended.unsafe_ptr()
    for i in range(100):
        appended.append(i)
    # CHECK: use of invalidated interior reference 'appended["element"]'
    print(ptr[])


def unsafe_ptr_then_reassign():
    var reassigned: List = [1]
    var ptr = reassigned.unsafe_ptr()
    reassigned = [2, 3]
    # CHECK: use of invalidated interior reference 'reassigned["element"]'
    print(ptr[])


def unsafe_ptr_then_clear():
    var cleared: List = [String("hello")]
    var ptr = cleared.unsafe_ptr()
    cleared.clear()
    # CHECK: use of invalidated interior reference 'cleared["element"]'
    print(ptr[])


def main():
    unsafe_ptr_then_append()
    unsafe_ptr_then_reassign()
    unsafe_ptr_then_clear()
