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

# Checks:
# 1. Require payload conformances and reject reserved names.
# 2. Honor manually supplied lifecycle methods.

# An enum declared in `std.builtin.stubs`, which `std.builtin._enum`
# imports, still compiles: demanding its members resolves `_enum`, which
# imports that module.
from std.builtin.stubs import _StubsEnum

comptime _stubs_enum_case = _StubsEnum.a


# Payloads must satisfy the union's derived lifecycle conformances.
__union Copies(ImplicitlyCopyable):
    case empty()
    # expected-error @+1 {{case 'list' has non-'ImplicitlyCopyable' payload type 'List[Int]'}}
    case list(List[Int])


__union Generic[T: AnyType]:
    # expected-error @+1 {{case 'value' has non-'Deinitable' payload type 'T'}}
    case value(Self.T)


# A case constructor moves its payload into the union, so every payload type
# must be `Movable`, even in a union that is not.
struct Pinned(not Movable):
    var x: Int


__union Unmovable(not Movable):
    case empty()
    # expected-error @+1 {{case 'pinned' has non-'Movable' payload type 'Pinned'}}
    case pinned(Pinned)
    # expected-error @+1 {{case 'pair' has non-'Movable' payload type 'Pinned'}}
    case pair(Int, Pinned)


# A where clause must prove a conditional payload conformance.
# expected-note @+1 {{unproven constraint}}
struct MovableIfPositive[n: Int](Deinitable, Movable where n > 0):
    var x: Int


__union UnprovenMovable[n: Int](not Movable):
    # expected-error @+1 {{case 'v' has non-'Movable' payload type 'MovableIfPositive[n]'}}
    case v(MovableIfPositive[Self.n])


__union ProvenMovable[n: Int](not Movable) where n > 0:
    case v(MovableIfPositive[Self.n])


# A custom destructor lifts the payload's Deinitable requirement.
comptime Callback = def(Int) thin -> Int


__union WithCallback(not Movable):
    case call(Callback)

    def __deinit__(deinit self):
        pass


__union WithIndex(not Movable):
    case index(__mlir_type.index)

    def __deinit__(deinit self):
        pass


# Trivial unions require `TrivialRegisterPassable` of each declared payload type.
# A nonmaterializable type must satisfy that requirement before materialization.
struct CopiedByHand(ImplicitlyCopyable, RegisterPassable):
    var x: Int

    def __init__(out self, *, copy: Self):
        self.x = copy.x


@__nonmaterializable(Int)
struct LiteralCopiedByHand(ImplicitlyCopyable, RegisterPassable):
    var x: Int

    def __init__(out self, *, copy: Self):
        self.x = copy.x


__union Trivial[T: AnyType, U: TrivialRegisterPassable](
    TrivialRegisterPassable
):
    case a()
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
# materializes to, so that type must be `Movable` in a trivial union too.
@__nonmaterializable(Pinned)
struct PinnedLiteral(TrivialRegisterPassable):
    var x: Int


__union TrivialPinnedLiteral(TrivialRegisterPassable):
    # expected-error @+1 {{case 'p' has payload type 'PinnedLiteral', which materializes to non-'Movable' type 'Pinned'}}
    case p(PinnedLiteral)


# As for a struct's field, the union's `where` clause does not bound a payload.
__union TrivialWhere[T: AnyType](TrivialRegisterPassable) where conforms_to(
    T, TrivialRegisterPassable
):
    # expected-error @+1 {{case 'v' has non-'TrivialRegisterPassable' payload type 'T'}}
    case v(Self.T)


__union HoldsCopiedByHand(ImplicitlyCopyable, RegisterPassable):
    case c(CopiedByHand)


# Check a tuple as one declared type, separately from multiple payload types.
__union TrivialSiblings(TrivialRegisterPassable):
    case pair(Int, Int)
    # expected-error @+1 {{case 'q' has non-'TrivialRegisterPassable' payload type 'String'}}
    case q(Int, String)
    # expected-error @+1 {{case 'tup' has non-'TrivialRegisterPassable' payload type 'Tuple[Int, Int]'}}
    case tup(Tuple[Int, Int])
    # expected-error @+1 {{case 'h' has non-'TrivialRegisterPassable' payload type 'HoldsCopiedByHand'}}
    case h(HoldsCopiedByHand)


# Like a trivial struct, a trivial union cannot declare a destructor, a copy
# constructor or a move constructor.
__union WithDeinit(TrivialRegisterPassable):
    case a()
    case b(Int)

    # expected-error @+2 {{trivial types must not declare '__deinit__' methods; they are trivially destroyable}}
    # expected-note @+1 {{trivial types have no identity; the compiler destroys them automatically with no observable effect}}
    def __deinit__(deinit self):
        pass


__union WithCopy(TrivialRegisterPassable):
    case a()
    case b(Int)

    # expected-error @+1 {{trivial types must not declare explicit copy constructors; they are trivially copyable}}
    def __init__(out self, *, copy: Self):
        self = .a()


__union WithMove(TrivialRegisterPassable):
    case a()
    case b(Int)

    # expected-error @+1 {{'RegisterPassable' types must not declare explicit move constructors; values of these types have no identity, and the compiler can freely move them between registers}}
    def __init__(out self, *, deinit move: Self):
        self = .a()


# A union cannot hold itself by value.
__union Chain:
    case end()
    # expected-error @+1 {{case 'next' has recursive payload type 'Chain'}}
    case next(Int, Self)


__union Clash:
    # expected-error @+1 {{union case 'red' conflicts with a member of the same name}}
    case red()
    # expected-error @+1 {{union case 'blue' conflicts with a member of the same name}}
    case blue(Int)

    # expected-note @+1 {{member declared here}}
    def red(self) -> Int:
        return 1

    # expected-note @+1 {{member declared here}}
    comptime blue = 3

    # expected-error @+1 {{'_get_enum_discriminant' is synthesized for unions and cannot be declared}}
    def _get_enum_discriminant(self) -> Int:
        return 0

    # expected-error @+1 {{'_enum_elt_type_for_case' is synthesized for unions and cannot be declared}}
    comptime _enum_elt_type_for_case[id: Int]: AnyType = Int


# A case's name is checked as the case resolves, so an error in another case
# does not hide a clash.
__union ClashBesideError:
    # expected-error @+1 {{use of unknown declaration 'NotAType'}}
    case a(NotAType)
    # expected-error @+1 {{union case 'b' conflicts with a member of the same name}}
    case b()

    # expected-note @+1 {{member declared here}}
    comptime b = 3


# An enum may declare that values other than its cases exist.
__enum Open:
    case a
    case b

    comptime _enum_is_exhaustive = False


# The support trait's associated comptimes are reserved, like its methods.
__enum ReservedComptimes:
    case a

    # expected-error @+1 {{'_EnumStorage' is synthesized for enums and cannot be declared}}
    comptime _EnumStorage = 1

    # expected-error @+1 {{'_enum_has_no_payloads' is synthesized for enums and cannot be declared}}
    comptime _enum_has_no_payloads = True


# A case's name is reserved the same way: the member it generates would
# collide with the support trait's.
__enum ReservedCaseName:
    case a

    # expected-error @+1 {{'_EnumStorage' is synthesized for enums and cannot be declared}}
    case _EnumStorage


# A case generates a member, which must not collide with a member that is
# generated for every union.
__union ReservedByEnum:
    case a()
    # expected-error @+1 {{invalid redefinition of '_get_enum_discriminant'}}
    case _get_enum_discriminant(Int)
    # expected-error @+1 {{invalid redefinition of '_enum_is_exhaustive'}}
    case _enum_is_exhaustive()
    # expected-error @+1 {{invalid redefinition of '__union_storage'}}
    case __union_storage()


__union ReservedByStruct(Copyable):
    case a()
    # expected-error @+1 {{invalid redefinition of '__init__'}}
    case __init__(Int)
    # expected-error @+1 {{invalid redefinition of '__copy_ctor_is_trivial'}}
    case __copy_ctor_is_trivial()
    # expected-error @+1 {{invalid redefinition of '__del__'}}
    case __del__()
    # expected-error @+1 {{case '__eq__' must not use the name of a special method that takes 'self'; rename the case}}
    case __eq__(Int)
    # expected-error @+1 {{case '__ne__' must not use the name of a special method that takes 'self'; rename the case}}
    case __ne__()


# A `__` prefix is an ordinary case name, as it is an ordinary member name.
__union Underscored:
    case __first()
    case __second(Int)


# expected-error @+1 {{'@fieldwise_init' cannot be applied to a union}}
@fieldwise_init
__union Fieldwise:
    case a(Int)


# A copy constructor, move constructor or `__deinit__` written in the union
# lifts the payload requirement of its own conformance only.
struct NoCopy(Movable):
    var x: Int

    def __init__(out self):
        self.x = 0


__union MovesByHand(Copyable):
    # expected-error @+1 {{case 'held' has non-'Copyable' payload type 'NoCopy'}}
    case held(NoCopy)

    def __init__(out self, *, deinit move: Self):
        self = Self.held(NoCopy())


# expected-error @+2 {{conditional conformance to 'TrivialRegisterPassable' is not supported}}
__union Conditional[T: Movable & Deinitable](
    TrivialRegisterPassable where conforms_to(T, TrivialRegisterPassable)
):
    case value(Self.T)


__union NonRegister(RegisterPassable):
    # expected-error @below {{case 'text' has non-'RegisterPassable' payload type 'String'}}
    case text(String)


__union Synthesized:
    case a(Int)
    # expected-error @+1 {{'_enum_case_names' is synthesized for unions and cannot be declared}}
    comptime _enum_case_names = 1
