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


# The storage of an enum has non-trivial copy, move and destruction. As for a
# struct whose header is rejected, the body is not resolved.
# expected-error @+1 {{enums cannot conform to 'TrivialRegisterPassable'}}
__enum Trivial(TrivialRegisterPassable):
    case a
    case b(NotAType)


# TODO(MOCO-5080): `NoneType` is `EnumLike`'s "no payload" marker, so a case
# cannot carry one; the rejection goes with the marker.
__enum NonePayload:
    # expected-error @+1 {{enum case payload must not be 'NoneType'; a case without a payload is written 'case nothing'}}
    case nothing(NoneType)


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
