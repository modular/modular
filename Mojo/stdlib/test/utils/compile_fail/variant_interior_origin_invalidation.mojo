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

# A reference vended by `Variant.unsafe_get()` carries an interior origin, so
# reassigning the `Variant` or calling a `mut self` method on it invalidates
# the reference at compile time.

from std.utils import Variant


def unsafe_get_then_reassign():
    var reassigned = Variant[Int, String](String("hello"))
    ref s = reassigned.unsafe_get[String]()
    reassigned = Variant[Int, String](42)
    # CHECK: variant_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'reassigned["value"]'
    print(s)


def unsafe_get_then_set():
    var set_in_place = Variant[Int, String](String("hello"))
    ref s = set_in_place.unsafe_get[String]()
    set_in_place.set[Int](42)
    # CHECK: variant_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'set_in_place["value"]'
    print(s)


def unsafe_get_then_replace():
    var replaced = Variant[Int, String](String("hello"))
    ref s = replaced.unsafe_get[String]()
    var old = replaced.replace[Int, String](42)
    # CHECK: variant_interior_origin_invalidation.mojo:[[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 'replaced["value"]'
    print(s, old)


def main():
    unsafe_get_then_reassign()
    unsafe_get_then_set()
    unsafe_get_then_replace()
