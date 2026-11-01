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

# Only `case name` and `case name(T, ...)` declare a case; every other spelling
# after `case` is rejected.
__enum Reserved:
    # expected-note @+1 {{previous definition here}}
    case a
    # expected-error @+1 {{invalid redefinition of 'a'}}
    case a
    """A doc string after a rejected case is still consumed."""
    # expected-error @+1 {{labeled payloads in enum cases are not supported; use positional types, e.g. 'case C(T, U)'}}
    case labeled(x: Int)
    # expected-error @+1 {{labeled payloads in enum cases are not supported; use positional types, e.g. 'case C(T, U)'}}
    case keyword(Int, y=Int)
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case struct Nested:
        pass
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case typed: Int
    """A doc string after a rejected case is still consumed."""
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case valued = 1
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case several, names
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case 1
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case empty()
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case trailing(Int) junk
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case missing_comma(Int Int)
    # expected-error @+2 {{'case' statement does not support decorators; remove the decorator}}
    @always_inline
    case decorated


# A rejected case still takes its name, so a later case with that name is a
# redefinition. A duplicate is rejected before its payload is resolved.
__enum Dropped:
    # expected-error @+2 {{labeled payloads in enum cases are not supported; use positional types, e.g. 'case C(T, U)'}}
    # expected-note @+1 {{previous definition here}}
    case a(x: Int)
    # expected-error @+1 {{invalid redefinition of 'a'}}
    case a
    # expected-error @+2 {{use of unknown declaration 'NotAType'}}
    # expected-note @+1 {{previous definition here}}
    case b(NotAType)
    # expected-error @+1 {{invalid redefinition of 'b'}}
    case b(AlsoNotAType)


struct Box[T: AnyType]:
    pass


__enum Payloads:
    # expected-error @+1 {{enum case payloads do not support trait types; 'AnyType' is a trait, use a concrete type or compile-time generic}}
    case trait_payload(AnyType)
    # expected-error @+1 {{use of unknown declaration 'NotAType'}}
    case unknown_payload(NotAType)
    # expected-error @+1 {{'Box[_]' is not concrete, use '[]' to bind missing parameters}}
    case unbound_payload(Box)


__enum Stateful:
    case a
    # expected-error @+1 {{enums do not support 'var' fields; store state in a case payload}}
    var x: Int
    """A doc string after a rejected field is still consumed."""


# A method body of only `...` is reserved in an enum.
__enum Bodies:
    case a

    def no_result(self):
        # expected-error @+1 {{enum methods must have a body; '...' is reserved}}
        ...

    def with_result(self) -> Int:
        # expected-error @+1 {{enum methods must have a body; '...' is reserved}}
        ...

    @staticmethod
    def static_method():
        """A doc string does not count as a body."""
        # expected-error @+1 {{enum methods must have a body; '...' is reserved}}
        ...

    # expected-error @+1 {{enum methods must have a body; '...' is reserved}}
    def one_line(self): ...

    # expected-error @+1 {{enum methods must have a body; '...' is reserved}}
    def with_semicolon(self): ...;

    def two_ellipses(self):
        # expected-error @+1 {{enum methods must have a body; '...' is reserved}}
        ...
        ...

    def more_than_ellipsis(self):
        ...
        pass

    def outer(self):
        def nested():
            ...

        nested()

    @unavailable("use 'a'")
    def unavailable(self):
        ...


__extension Bodies:
    def extension_method(self):
        # expected-error @+1 {{enum methods must have a body; '...' is reserved}}
        ...

    @unavailable("use 'a'")
    def extension_unavailable(self):
        ...


# expected-error @+1 {{expected ':' in enum definition}}
__enum NoColon
    case a


struct OuterStruct:
    # expected-error @+1 {{nested enum not supported here}}
    __enum Inner:
        case a


trait OuterTrait:
    # expected-error @+1 {{nested enum in a trait not supported here}}
    __enum Inner:
        case a


def outer_function():
    # expected-error @+1 {{enum inside a function not supported here}}
    __enum Inner:
        case a


def comptime_enum():
    # expected-error @+1 {{'comptime' cannot be used with '__enum'}}
    comptime __enum = 1


# expected-error @+1 {{expected enum name}}
__enum
