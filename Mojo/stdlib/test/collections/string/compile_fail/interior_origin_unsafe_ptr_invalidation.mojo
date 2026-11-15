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

# A pointer vended by a `String` carries an interior origin, so mutating the
# string (which may reallocate its buffer) invalidates the pointer at compile
# time.


def unsafe_ptr_then_append():
    var s = String("hello world")
    var ptr = s.unsafe_ptr()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    print(ptr[])


def unsafe_ptr_mut_then_reserve():
    var s = String("hello world")
    var ptr = s.unsafe_ptr_mut()
    s.reserve_bytes(1000)
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    ptr[] = 72


def unsafe_ptr_then_unsafe_ptr_mut():
    var s = String("hello world")
    var ptr = s.unsafe_ptr()
    var grown = s.unsafe_ptr_mut(capacity=1000)
    grown[] = 72
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    print(ptr[])


def unsafe_ptr_then_reassign():
    var s = String("hello world, long enough to live on the heap")
    var ptr = s.unsafe_ptr()
    s = String("a different string, long enough to live on the heap")
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    print(ptr[])


def main():
    unsafe_ptr_then_append()
    unsafe_ptr_mut_then_reserve()
    unsafe_ptr_then_unsafe_ptr_mut()
    unsafe_ptr_then_reassign()
