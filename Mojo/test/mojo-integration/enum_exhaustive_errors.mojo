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

# Only the `EnumLike` conformance reports an `_enum_is_exhaustive` that is not a
# `Bool`; the members that read the conformance report nothing.


# CHECK-NOT: error:
# CHECK: error: 'Mistyped' does not implement all requirements for 'EnumLike'
# CHECK: note: comptime member '_enum_is_exhaustive' type 'IntLiteral[5]' does not conform to trait's required type 'Bool'
# CHECK: note: trait 'EnumLike' declared here
# CHECK-NOT: error:
# CHECK: error: failed to parse the provided Mojo source module
# TODO: `ImplicitlyCopyable` only so that `Mistyped.a` can be used as a runtime
# value; a `comptime` value of another type does not materialize.
__enum Mistyped(ImplicitlyCopyable, Writable):
    case a
    case b(Int)

    comptime _enum_is_exhaustive = 5


def main():
    var value: Mistyped = .b(1)
    print(value, Mistyped.a)
