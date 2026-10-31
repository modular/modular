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

"""Moving an rvalue payload out of a `var` match binding must not destroy the
subject's storage before the move, and must leave exactly one destruction."""

from std.os import abort


struct Tracked(Deinitable, Movable):
    """A payload that reports every move and destruction."""

    var val: Int

    def __init__(out self, val: Int):
        self.val = val
        print("init", self.val)

    def __init__(out self, *, deinit move: Self):
        self.val = move.val
        print("move", self.val)

    def __deinit__(deinit self):
        print("deinit", self.val)


__enum MyEnum:
    case Empty
    case Val(Tracked)

    def take(var self) -> Tracked:
        match self^:
            case .Empty:
                abort("empty")
            case .Val(var v):
                return v^


def take_from_enum() -> Int:
    var e = MyEnum.Val(Tracked(1))
    var v = e^.take()
    print("got v:", v.val)
    return v.val


def take_from_optional(var opt: Optional[Tracked]) -> Int:
    if case .Some(var v) = opt^:
        return v.val
    abort("none")


def take_from_some() -> Int:
    var opt = Optional(Tracked(2))
    var v = take_from_optional(opt^)
    print("got v:", v)
    return v


# The subject's storage used to be destroyed before the payload was moved into
# the binding, which printed a `deinit` before `got v` and destroyed the moved
# value a second time at the end of its lifetime.
# CHECK: init 1{{$}}
# CHECK-NOT: deinit
# CHECK: got v: 1{{$}}
# CHECK-NOT: deinit
# CHECK: deinit 1{{$}}
# CHECK: init 2{{$}}
# CHECK: deinit 2{{$}}
# CHECK: got v: 2{{$}}
# CHECK-NOT: deinit
def main():
    _ = take_from_enum()
    _ = take_from_some()
