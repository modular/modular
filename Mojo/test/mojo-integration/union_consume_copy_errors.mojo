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

# Checks:
# 1. Reject an implicit copy of a payload binding of a `__union`.
# TODO(MOCO-5454): A payload `var` requires an implicit copy until moving a
# part out of a value is sound.


struct MoveOnly(Movable):
    pass


__union MovePayload(Movable):
    case value(MoveOnly)


def moveonly_payload(var u: MovePayload):
    match u^:
        # CHECK: error: value of type 'MoveOnly' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'
        # CHECK-NEXT: case .value(var value):
        case .value(var value):
            _ = value^


__union TwoMoveOnly(Movable):
    case two(MoveOnly, MoveOnly)


def multi_moveonly_payload(var u: TwoMoveOnly):
    match u^:
        # CHECK: error: value of type 'MoveOnly' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'
        # CHECK-NEXT: case .two(var a, var b):
        case .two(var a, var b):
            _ = a^
            _ = b^


def main():
    pass
