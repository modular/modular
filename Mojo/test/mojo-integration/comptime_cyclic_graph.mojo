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

# Reference-counting the interpreter's blob graph used to recurse until the
# stack was gone on any cycle (MOCO-5028). The hand-built graphs below cover
# the cycle shapes a doubly-linked list is not; `LinkedList` covers the
# two-node cycle that first surfaced the bug.

from std.collections.linked_list import LinkedList
from std.memory import Layout, alloc


@fieldwise_init
struct Node(Movable):
    var value: Int
    var next: Pointer[Self, MutUntrackedOrigin]


@fieldwise_init
struct Graph(Movable):
    var head: Pointer[Node, MutUntrackedOrigin]


def node() -> Pointer[Node, MutUntrackedOrigin]:
    return alloc(Layout[Node](length=1)).unsafe_leak()


# The graph has to arrive as an argument to a comptime call: that is what
# serializes it into the blob graph reference-counting then walks.
def walk(var g: Graph, steps: Int) -> Int:
    var total = 0
    var cur = g.head
    for _ in range(steps):
        total += cur[].value
        cur = cur[].next
    return total


def self_loop() -> Graph:
    # a -> a
    var a = node()
    a.unsafe_write(Node(7, a))
    return Graph(a)


def three_cycle() -> Graph:
    # a -> b -> c -> a
    var a = node()
    var b = node()
    var c = node()
    a.unsafe_write(Node(1, b))
    b.unsafe_write(Node(2, c))
    c.unsafe_write(Node(4, a))
    return Graph(a)


def rho() -> Graph:
    # a -> b -> c -> d -> b
    var a = node()
    var b = node()
    var c = node()
    var d = node()
    a.unsafe_write(Node(1, b))
    b.unsafe_write(Node(2, c))
    c.unsafe_write(Node(4, d))
    d.unsafe_write(Node(8, b))
    return Graph(a)


def sum_of(var l: LinkedList[Int]) -> Int:
    var total = 0
    for v in l:
        total += v
    return total


def grown() -> Int:
    var l = LinkedList[Int]()
    for i in range(10):
        l.append(i)
    return sum_of(l^)


def main():
    # CHECK: 21
    comptime looped = walk(self_loop(), 3)
    print(looped)

    # CHECK-NEXT: 14
    comptime cycled = walk(three_cycle(), 6)
    print(cycled)

    # CHECK-NEXT: 29
    comptime tailed = walk(rho(), 7)
    print(tailed)

    # CHECK-NEXT: 3
    comptime size = len(LinkedList[Int](1, 2, 3))
    print(size)

    # CHECK-NEXT: 6
    comptime total = sum_of(LinkedList[Int](1, 2, 3))
    print(total)

    # CHECK-NEXT: 45
    comptime grown_total = grown()
    print(grown_total)
