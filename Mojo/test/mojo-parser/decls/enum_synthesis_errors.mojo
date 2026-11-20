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

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# An enum holds the payload of its active case, so every payload type must
# support each conformance of the enum that copies, moves or destroys it.
__enum Copies(ImplicitlyCopyable):
    case empty
    # expected-error @+1 {{case 'list' has non-'ImplicitlyCopyable' payload type 'List[Int]'}}
    case list(List[Int])


__enum Generic[T: AnyType]:
    # expected-error @+1 {{case 'value' has non-'Deinitable' payload type 'T'}}
    case value(Self.T)


# A case constructor moves its payload into the enum, so every payload type
# must be `Movable`, even in an enum that is not.
struct Pinned(not Movable):
    var x: Int


__enum Unmovable(not Movable):
    case empty
    # expected-error @+1 {{case 'pinned' has non-'Movable' payload type 'Pinned'}}
    case pinned(Pinned)
    # expected-error @+1 {{case 'pair' has non-'Movable' payload type 'Pinned'}}
    case pair(Int, Pinned)


# A payload that is `Movable` only under a condition is accepted where the
# enum's `where` clause proves it; otherwise a note points at the condition.
# expected-note @+1 {{unproven constraint}}
struct MovableIfPositive[n: Int](Deinitable, Movable where n > 0):
    var x: Int


__enum UnprovenMovable[n: Int](not Movable):
    # expected-error @+1 {{case 'v' has non-'Movable' payload type 'MovableIfPositive[n]'}}
    case v(MovableIfPositive[Self.n])


__enum ProvenMovable[n: Int](not Movable) where n > 0:
    case v(MovableIfPositive[Self.n])


# A payload type that is not a struct is `Movable` as well; each enum writes
# its `__deinit__`, so `Movable` is all its cases require of their payloads.
comptime Callback = def(Int) thin -> Int


__enum WithCallback(not Movable):
    case call(Callback)

    def __deinit__(deinit self):
        pass


__enum WithIndex(not Movable):
    case index(__mlir_type.index)

    def __deinit__(deinit self):
        pass


# A trivial enum is copied bit for bit, so every payload type must be
# `TrivialRegisterPassable`, as every field of a trivial struct must be. A
# payload type that is not a struct is `TrivialRegisterPassable` as well. A
# nonmaterializable payload type is checked against `TrivialRegisterPassable`
# as itself, not as the type it materializes to.
struct CopiedByHand(ImplicitlyCopyable, RegisterPassable):
    var x: Int

    def __init__(out self, *, copy: Self):
        self.x = copy.x


@__nonmaterializable(Int)
struct LiteralCopiedByHand(ImplicitlyCopyable, RegisterPassable):
    var x: Int

    def __init__(out self, *, copy: Self):
        self.x = copy.x


__enum Trivial[T: AnyType, U: TrivialRegisterPassable](
    TrivialRegisterPassable
):
    case a
    case b(Int)
    case u(Self.U)
    case index(__mlir_type.index)
    case call(Callback)
    case lit(type_of("x"))
    # expected-error @+1 {{case 's' has non-'TrivialRegisterPassable' payload type 'String'}}
    case s(String)
    # expected-error @+1 {{case 't' has non-'TrivialRegisterPassable' payload type 'T'}}
    case t(Self.T)
    # expected-error @+1 {{case 'c' has non-'TrivialRegisterPassable' payload type 'CopiedByHand'}}
    case c(CopiedByHand)
    # expected-error @+1 {{case 'n' has non-'TrivialRegisterPassable' payload type 'LiteralCopiedByHand'}}
    case n(LiteralCopiedByHand)


# A case's constructor moves a nonmaterializable payload as the type it
# materializes to, so that type must be `Movable` in a trivial enum too.
@__nonmaterializable(Pinned)
struct PinnedLiteral(TrivialRegisterPassable):
    var x: Int


__enum TrivialPinnedLiteral(TrivialRegisterPassable):
    # expected-error @+1 {{case 'p' has payload type 'PinnedLiteral', which materializes to non-'Movable' type 'Pinned'}}
    case p(PinnedLiteral)


# As for a struct's field, the enum's `where` clause does not bound a payload.
__enum TrivialWhere[T: AnyType](TrivialRegisterPassable) where conforms_to(
    T, TrivialRegisterPassable
):
    # expected-error @+1 {{case 'v' has non-'TrivialRegisterPassable' payload type 'T'}}
    case v(Self.T)


__enum HoldsCopiedByHand(ImplicitlyCopyable, RegisterPassable):
    case c(CopiedByHand)


# A payload of several types is checked type by type; a `Tuple` is checked as
# one type, as a trivial struct's `Tuple` field is.
__enum TrivialSiblings(TrivialRegisterPassable):
    case pair(Int, Int)
    # expected-error @+1 {{case 'q' has non-'TrivialRegisterPassable' payload type 'String'}}
    case q(Int, String)
    # expected-error @+1 {{case 'tup' has non-'TrivialRegisterPassable' payload type 'Tuple[Int, Int]'}}
    case tup(Tuple[Int, Int])
    # expected-error @+1 {{case 'h' has non-'TrivialRegisterPassable' payload type 'HoldsCopiedByHand'}}
    case h(HoldsCopiedByHand)


# Like a trivial struct, a trivial enum cannot declare a destructor, a copy
# constructor or a move constructor.
__enum WithDeinit(TrivialRegisterPassable):
    case a
    case b(Int)

    # expected-error @+2 {{trivial types must not declare '__deinit__' methods; they are trivially destroyable}}
    # expected-note @+1 {{trivial types have no identity; the compiler destroys them automatically with no observable effect}}
    def __deinit__(deinit self):
        pass


__enum WithCopy(TrivialRegisterPassable):
    case a
    case b(Int)

    # expected-error @+1 {{trivial types must not declare explicit copy constructors; they are trivially copyable}}
    def __init__(out self, *, copy: Self):
        self = .a


__enum WithMove(TrivialRegisterPassable):
    case a
    case b(Int)

    # expected-error @+1 {{'RegisterPassable' types must not declare explicit move constructors; values of these types have no identity, and the compiler can freely move them between registers}}
    def __init__(out self, *, deinit move: Self):
        self = .a


# An enum cannot hold itself by value.
__enum Chain:
    case end
    # expected-error @+1 {{case 'next' has recursive payload type 'Chain'}}
    case next(Int, Self)


__enum Clash:
    # expected-error @+1 {{enum case 'red' conflicts with a member of the same name}}
    case red
    # expected-error @+1 {{enum case 'blue' conflicts with a member of the same name}}
    case blue(Int)

    # expected-note @+1 {{member declared here}}
    def red(self) -> Int:
        return 1

    # expected-note @+1 {{member declared here}}
    comptime blue = 3

    # expected-error @+1 {{'_get_enum_discriminant' is synthesized for enums and cannot be declared}}
    def _get_enum_discriminant(self) -> Int:
        return 0

    # expected-error @+1 {{'_enum_elt_type_for_case' is synthesized for enums and cannot be declared}}
    comptime _enum_elt_type_for_case[id: Int]: AnyType = Int


# A case's name is checked as the case resolves, so an error in another case
# does not hide a clash.
__enum ClashBesideError:
    # expected-error @+1 {{use of unknown declaration 'NotAType'}}
    case a(NotAType)
    # expected-error @+1 {{enum case 'b' conflicts with a member of the same name}}
    case b

    # expected-note @+1 {{member declared here}}
    comptime b = 3


# An enum may declare that values other than its cases exist.
__enum Open:
    case a
    case b

    comptime _enum_is_exhaustive = False


# A case generates a member, which must not collide with a member that is
# generated for every enum.
__enum ReservedByEnum:
    case a
    # expected-error @+1 {{invalid redefinition of '_get_enum_discriminant'}}
    case _get_enum_discriminant(Int)
    # expected-error @+1 {{invalid redefinition of '_enum_is_exhaustive'}}
    case _enum_is_exhaustive
    # expected-error @+1 {{invalid redefinition of '__enum_storage'}}
    case __enum_storage


__enum ReservedByStruct(Copyable):
    case a
    # expected-error @+1 {{invalid redefinition of '__init__'}}
    case __init__(Int)
    # expected-error @+1 {{invalid redefinition of '__copy_ctor_is_trivial'}}
    case __copy_ctor_is_trivial
    # expected-error @+1 {{invalid redefinition of '__del__'}}
    case __del__
    # expected-error @+1 {{case '__eq__' must not use the name of a special method that takes 'self'; rename the case}}
    case __eq__(Int)
    # expected-error @+1 {{case '__ne__' must not use the name of a special method that takes 'self'; rename the case}}
    case __ne__


# A `__` prefix is an ordinary case name, as it is an ordinary member name.
__enum Underscored:
    case __first
    case __second(Int)


# expected-error @+1 {{'@fieldwise_init' cannot be applied to an enum}}
@fieldwise_init
__enum Fieldwise:
    case a


# `match` checks enum cases through the synthesized `EnumLike` conformance.
__enum Light:
    case on
    case off
    case dimmed(Int)


def match_cases(light: Light) -> Int:
    # expected-warning @+1 {{'match' is not exhaustive; missing case for 'off'}}
    match light:
        case .on:
            return 1
        case .dimmed(level):
            return level
    match light:
        # expected-error @+1 {{'blinking' is not a case of 'Light'}}
        case .blinking:
            return 2
        # expected-error @+1 {{enum case 'off' has no associated value}}
        case .off(_):
            return 3
        case _:
            return 4


# A copy constructor, move constructor or `__deinit__` written in the enum
# lifts the payload requirement of its own conformance only.
struct NoCopy(Movable):
    var x: Int

    def __init__(out self):
        self.x = 0


__enum MovesByHand(Copyable):
    # expected-error @+1 {{case 'held' has non-'Copyable' payload type 'NoCopy'}}
    case held(NoCopy)

    def __init__(out self, *, deinit move: Self):
        self = Self.held(NoCopy())
