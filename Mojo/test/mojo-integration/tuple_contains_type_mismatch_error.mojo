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

# Test that `Tuple.__contains__` rejects a value whose type is not one of the
# tuple's element types. Such a value is never compared against the elements,
# so without the `where` clause the result would silently be False.


def main():
    var ints = (1, 2, 3)
    var bools = (False, True)
    var mixed = (123, True, "Mojo is awesome")

    # CHECK: [[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__contains__': violated constraint
    # CHECK: note: constraint declared here evaluated to False, expected 'TypeList.contains[T]()'
    _ = UInt8(1) in ints

    # `Int` and `Bool` are distinct types, even though 1 == True in Python.
    # CHECK: [[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__contains__': violated constraint
    # CHECK: note: constraint declared here evaluated to False, expected 'TypeList.contains[T]()'
    _ = 1 in bools

    # CHECK: [[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__contains__': violated constraint
    # CHECK: note: constraint declared here evaluated to False, expected 'TypeList.contains[T]()'
    _ = True in ints

    # The string element is a `String`, not a `StaticString`.
    # CHECK: [[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__contains__': violated constraint
    # CHECK: note: constraint declared here evaluated to False, expected 'TypeList.contains[T]()'
    _ = StaticString("Mojo is awesome") in mixed
