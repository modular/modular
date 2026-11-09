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

# A reference vended by an `Optional` carries an interior origin, so
# reassigning the `Optional` or calling a `mut self` method on it invalidates
# the reference at compile time.


def value_then_reassign():
    var reassigned: Optional[Int] = 5
    ref i = reassigned.value()
    reassigned = None
    # CHECK: use of invalidated interior reference 'reassigned["value"]'
    print(i)


def getitem_then_reassign() raises:
    var replaced = Optional(String("hello"))
    ref s = replaced[]
    replaced = String("goodbye")
    # CHECK: use of invalidated interior reference 'replaced["value"]'
    print(s)


def unsafe_value_then_take():
    var taken = Optional(String("hello"))
    ref s = taken.unsafe_value()
    var owned = taken.take()
    # CHECK: use of invalidated interior reference 'taken["value"]'
    print(s, owned)


def main() raises:
    value_then_reassign()
    getitem_then_reassign()
    unsafe_value_then_take()
