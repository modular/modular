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

# Checks:
# 1. Copy the payload elements of a consumed `__enum` or `__union`, at every
#    payload size and below the payload, in a `match`, an `if case` and a
#    `while case`.
# 2. Move a binding of the whole subject and of a direct element of a tuple
#    subject.
# 3. Borrow a `ref` binding; copy a merged or-binding; an arm whose guard
#    fails destroys its copy and leaves the subject whole.
# 4. Keep `main`'s move for a hand-written `EnumLike` conformer: a payload
#    that is only `Movable` still binds with `var`.
# TODO(MOCO-5454): Move payload elements out once moving a part out of a
# value is sound.


struct Tracked(ImplicitlyCopyable, Movable):
    var id: Int

    def __init__(out self, id: Int):
        self.id = id

    def __init__(out self, *, copy: Self):
        self.id = copy.id + 1000
        print("copy", copy.id, self.id)

    def __init__(out self, *, deinit move: Self):
        self.id = move.id

    def __deinit__(deinit self):
        print("destroy", self.id)


__union Inner(Copyable):
    case t(Tracked)
    case none()


__union U(Copyable):
    case t(Tracked)
    case two(Tracked, Tracked)
    case inner(Inner)
    case none()


__union WholeU(ImplicitlyCopyable):
    case one(Tracked)


def show(u: WholeU):
    match u:
        case .one(t):
            print("use", t.id)


def consume_var(var u: U):
    # CHECK-LABEL: == var
    print("== var")
    # CHECK-NEXT: copy 1 1001
    # CHECK-NEXT: destroy 1
    # CHECK-NEXT: bound 1001
    # CHECK-NEXT: destroy 1001
    match u^:
        case .t(var x):
            print("bound", x.id)
        case _:
            pass
    # CHECK-NOT: {{copy|destroy}}


def consume_two_vars(var u: U):
    # CHECK-LABEL: == two vars
    print("== two vars")
    # CHECK-NEXT: copy 2 1002
    # CHECK-NEXT: copy 3 1003
    # CHECK-NEXT: destroy 2
    # CHECK-NEXT: destroy 3
    # CHECK-NEXT: bound 1002 1003
    # CHECK-NEXT: destroy 1003
    # CHECK-NEXT: destroy 1002
    match u^:
        case .two(var a, var b):
            print("bound", a.id, b.id)
        case _:
            pass
    # CHECK-NOT: {{copy|destroy}}


def consume_nested(var u: U):
    # CHECK-LABEL: == nested
    print("== nested")
    # CHECK-NEXT: copy 8 1008
    # CHECK-NEXT: destroy 8
    # CHECK-NEXT: bound 1008
    # CHECK-NEXT: destroy 1008
    match u^:
        case .inner(.t(var x)):
            print("bound", x.id)
        case _:
            pass
    # CHECK-NOT: {{copy|destroy}}


def consume_ref(var u: U):
    # CHECK-LABEL: == ref
    print("== ref")
    # CHECK-NEXT: bound 10
    # CHECK-NEXT: destroy 10
    match u^:
        case .t(ref x):
            print("bound", x.id)
        case _:
            pass
    # CHECK-NOT: {{copy|destroy}}


def consume_tag_only(var u: U):
    # CHECK-LABEL: == tag-only match
    print("== tag-only match")
    # CHECK-NEXT: tag empty
    # CHECK-NEXT: == tag-only match
    # CHECK-NEXT: destroy 42
    # CHECK-NEXT: tag payload
    match u^:
        case .t(_):
            print("tag payload")
        case .none():
            print("tag empty")
        case _:
            pass
    # CHECK-NOT: {{copy|destroy}}


def consume_whole(var u: U):
    # CHECK-LABEL: == whole union
    print("== whole union")
    # CHECK-NEXT: bound whole
    # CHECK-NEXT: destroy 48
    match u^:
        case var whole:
            print("bound whole")
            _ = whole^
    # CHECK-NOT: {{copy|destroy}}


def direct_tuple():
    # CHECK-LABEL: == direct tuple
    print("== direct tuple")
    # CHECK-NEXT: use 70
    # CHECK-NEXT: destroy 70
    # CHECK-NEXT: use 71
    # CHECK-NEXT: destroy 71
    match (WholeU.one(Tracked(70)), Tracked(71)):
        case (var u, var n):
            show(u)
            print("use", n.id)
    # CHECK-NOT: {{copy|destroy}}


def consume_or_var(var u: U):
    # CHECK-LABEL: == or vars
    print("== or vars")
    # CHECK-NEXT: copy 50 1050
    # CHECK-NEXT: destroy 50
    # CHECK-NEXT: bound 1050
    # CHECK-NEXT: destroy 1050
    # CHECK-NEXT: == or vars
    # CHECK-NEXT: copy 52 1052
    # CHECK-NEXT: destroy 52
    # CHECK-NEXT: bound 1052
    # CHECK-NEXT: destroy 1052
    match u^:
        case .t(var x) | .inner(.t(var x)):
            print("bound", x.id)
        case _:
            pass
    # CHECK-NOT: {{copy|destroy}}


def consume_guarded(var u: U):
    # CHECK-LABEL: == guarded
    print("== guarded")
    # CHECK-NEXT: copy 60 1060
    # CHECK-NEXT: destroy 1060
    # CHECK-NEXT: copy 60 1060
    # CHECK-NEXT: destroy 60
    # CHECK-NEXT: second 1060
    # CHECK-NEXT: destroy 1060
    match u^:
        case .t(var x) if x.id < 100:
            print("first", x.id)
        case .t(var x):
            print("second", x.id)
        case _:
            pass
    # CHECK-NOT: {{copy|destroy}}


def consume_if_case():
    # CHECK-LABEL: == if case
    print("== if case")
    # CHECK-NEXT: copy 30 1030
    # CHECK-NEXT: destroy 30
    # CHECK-NEXT: if bound 1030
    # CHECK-NEXT: destroy 1030
    if case .t(var x) = U.t(Tracked(30)):
        print("if bound", x.id)
    # CHECK-NOT: {{copy|destroy}}


def next_t(i: Int) -> U:
    if i < 2:
        return U.t(Tracked(40 + i))
    return U.none()


def consume_while_case():
    # CHECK-LABEL: == while case
    print("== while case")
    # CHECK-NEXT: copy 40 1040
    # CHECK-NEXT: destroy 40
    # CHECK-NEXT: while bound 1040
    # CHECK-NEXT: destroy 1040
    # CHECK-NEXT: copy 41 1041
    # CHECK-NEXT: destroy 41
    # CHECK-NEXT: while bound 1041
    # CHECK-NEXT: destroy 1041
    var i = 0
    while case .t(var x) = next_t(i):
        print("while bound", x.id)
        i += 1
    # CHECK-NOT: {{copy|destroy}}


# A hand-written `EnumLike` conformer keeps `main`'s move: a payload that is
# only `Movable` still binds with `var`.
struct MoveOnly(Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


def make_moveonly() -> Optional[MoveOnly]:
    var value: Optional[MoveOnly] = MoveOnly(1)
    return value^


def consume_optional_moveonly(var o: Optional[MoveOnly]):
    match o^:
        case .Some(var x):
            _ = x^
        case .None:
            pass
    # CHECK-LABEL: == optional move-only
    print("== optional move-only")


def main():
    consume_var(U.t(Tracked(1)))
    consume_two_vars(U.two(Tracked(2), Tracked(3)))
    consume_nested(U.inner(Inner.t(Tracked(8))))
    consume_ref(U.t(Tracked(10)))
    consume_tag_only(U.none())
    consume_tag_only(U.t(Tracked(42)))
    consume_whole(U.t(Tracked(48)))
    direct_tuple()
    consume_or_var(U.t(Tracked(50)))
    consume_or_var(U.inner(Inner.t(Tracked(52))))
    consume_guarded(U.t(Tracked(60)))
    consume_if_case()
    consume_while_case()
    consume_optional_moveonly(make_moveonly())
