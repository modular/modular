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

# As in Python, an unpacked `*pack` can be one of many operands of a variadic
# pack argument. The callee gets the concatenation.

# RUN: mojo %s | FileCheck %s


def show[*Ts: Writable](*args: *Ts):
    comptime for i in range(args.__len__()):
        print(args[i], end=" ")
    print()


def count[*Ts: AnyType](*args: *Ts) -> Int:
    return args.__len__()


def first[*Ts: Writable](*args: *Ts) -> type_of(args):
    return args.copy()


def fwd[*Ts: Writable](*args: *Ts):
    # CHECK: front: 0 1 a 2.5
    print("front:", end=" ")
    show(0, *args)
    # CHECK: back: 1 a 2.5 9
    print("back:", end=" ")
    show(*args, 9)
    # CHECK: both: 1 a 2.5 1 a 2.5
    print("both:", end=" ")
    show(*args, *args)
    # CHECK: middle: 0 1 a 2.5 9
    print("middle:", end=" ")
    show(0, *args, 9)

    # A temporary pack can be unpacked.
    var seven = 7
    # CHECK: temporary: 7 1 a 2.5
    print("temporary:", end=" ")
    show(*first(seven), *args)

    # A local copy can be unpacked alone or with other operands.
    var copy = args.copy()
    # CHECK: copy: 1 a 2.5
    print("copy:", end=" ")
    show(*copy)
    # CHECK: copy concat: 1 a 2.5 0
    print("copy concat:", end=" ")
    show(*copy, 0)

    # A pack of a refining trait gets an upcast to the trait of the callee.
    # CHECK: upcast: 4
    print("upcast:", count(0, *args))

    # A different spelling of the same trait also gets an upcast.
    # CHECK: alias: 0 1 a 2.5
    print("alias:", end=" ")
    show_alias(0, *args)


comptime W = Writable


def show_alias[*Ts: W](*args: *Ts):
    show(*args)


def total[*Ts: Intable](*args: *Ts) -> Int:
    var sum = 0
    comptime for i in range(args.__len__()):
        sum += Int(args[i])
    return sum


def total_with_ends[*Ts: Intable](*args: *Ts) -> Int:
    return total(100, *args, 1000)


trait Bumpable:
    def bump(mut self):
        ...


@fieldwise_init
struct Counter(Bumpable, Writable):
    var n: Int

    def bump(mut self):
        self.n += 1


def bump_all[*Ts: Bumpable](mut *args: *Ts):
    comptime for i in range(args.__len__()):
        args[i].bump()


# Writes through a concatenated `mut` pack reach each element.
def fwd_mut[*Ts: Bumpable](mut *args: *Ts):
    var x = Counter(10)
    var y = Counter(20)
    bump_all(x, *args, y)
    # CHECK: mut ends: Counter(n=11) Counter(n=21)
    print("mut ends:", x, y)


def main():
    fwd(1, "a", 2.5)

    var a = Counter(1)
    var b = Counter(2)
    fwd_mut(a, b)
    # CHECK: mut pack: Counter(n=2) Counter(n=3)
    print("mut pack:", a, b)

    # The comptime interpreter also does the concatenation.
    comptime at_comptime = total_with_ends(1, 2)
    # CHECK: comptime: 1103
    print("comptime:", at_comptime)
    # CHECK: runtime: 1103
    print("runtime:", total_with_ends(1, 2))
