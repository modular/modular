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
# RUN: %mojo -O3 %s | FileCheck %s

from std.memory import alloc
from std.utils import Variant


@fieldwise_init
struct Callback(ImplicitlyCopyable):
    var function: def(Int) thin -> Int


def increment(value: Int) -> Int:
    return value + 1


def double(value: Int) -> Int:
    return value * 2


# Assign 42 over a callback in a list; a later comptime read must give 42.
def number_over_callback_in_list() -> List[Variant[Callback, Int]]:
    var values: List[Variant[Callback, Int]] = [Callback(increment)]
    values[0] = 42
    return values^


# Store a callback, then 42 in the same allocation; a later read must give 42.
def overwrite_function_with_integer() -> Pointer[Int, MutUntrackedOrigin]:
    var allocation = alloc[Callback]({length = 1}).unsafe_leak()
    allocation.unsafe_write(Callback(increment))
    var integers = allocation.unsafe_bitcast[Int]()
    integers[] = 42
    return integers


# Store increment, 42, then double; a later read must give 11, 42 and 34.
def function_integer_function_round_trip() -> Tuple[Int, Int, Callback]:
    var allocation = alloc[Callback]({length = 1}).unsafe_leak()
    allocation.unsafe_write(Callback(increment))
    var first = allocation[].function(10)
    var integers = allocation.unsafe_bitcast[Int]()
    integers[] = 42
    var second = integers[]
    allocation.unsafe_write(Callback(double))
    return (first, second, allocation[])


def main():
    comptime values = number_over_callback_in_list()
    comptime number = values[0][Int]
    # CHECK: list: 42
    print("list:", number)

    comptime pointer = overwrite_function_with_integer()
    comptime value = pointer[]
    # CHECK-NEXT: allocation: 42
    print("allocation:", value)

    comptime result = function_integer_function_round_trip()
    comptime last = result[2].function(17)
    # CHECK-NEXT: round trip: 11 42 34
    print("round trip:", result[0], result[1], last)
