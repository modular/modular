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
# RUN: %mojo %s | FileCheck %s

# Ensure the extension types generated to support imm to owned argument
# convention conversions are proper type views and not considered different
# types.


@fieldwise_init
struct Holder[F: def(var Int) -> Int](Movable):
    var f: Self.F


def add_one(x: Int) -> Int:
    return x + 1


def main():
    # COM: def(Int) -> Int requires extension to convert to
    # def(var Int) -> Int
    var h = Holder(add_one)
    # CHECK: 42
    print(h.f(41))
