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
# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s -o /dev/null 2>&1 | FileCheck %s --check-prefix PAYLOAD

# Checks:
# 1. Reject `()` on an enum case without a payload.
# 2. Select enum wording for shared diagnostics.

__enum EmptyParens:
    # PAYLOAD: "fixIts":[],"location":{"column":14,"line":[[#@LINE+3]]}
    # PAYLOAD-SAME: "message":"enum case 'empty' must not have a payload; declare 'EmptyParens' with '__union'"
    # expected-error @+1 {{enum case 'empty' must not have a payload; declare 'EmptyParens' with '__union'}}
    case empty()


__enum Payload:
    # expected-error @+1 {{enum case 'data' must not have a payload; declare 'Payload' with '__union'}}
    case data(Int)


__enum Malformed:
    case value
    # expected-error @+1 {{enum case must have the form 'case name'}}
    case typed: Int


__enum Stateful:
    case value
    # expected-error @+1 {{enums do not support 'var' fields; use '__union' to store case payloads}}
    var x: Int


__enum Bodies:
    case value

    def method(self):
        # expected-error @+1 {{enum methods must have a body; '...' is reserved}}
        ...


# expected-error @+1 {{'@fieldwise_init' cannot be applied to an enum}}
@fieldwise_init
__enum Fieldwise:
    case value


# expected-error @+1 {{expected ':' in enum definition}}
__enum NoColon
    case value


struct OuterStruct:
    # expected-error @+1 {{nested enum not supported here}}
    __enum Inner:
        case value


trait OuterTrait:
    # expected-error @+1 {{nested enum in a trait not supported here}}
    __enum Inner:
        case value


def outer_function():
    # expected-error @+1 {{enum inside a function not supported here}}
    __enum Inner:
        case value


def comptime_enum():
    # expected-error @+1 {{'comptime' cannot be used with '__enum'}}
    comptime __enum = 1


__enum NotImplicitlyCopyable(not ImplicitlyCopyable):
    case a


def materialize_not_implicitly_copyable() -> NotImplicitlyCopyable:
    # expected-error @+2 {{cannot materialize comptime value of type 'NotImplicitlyCopyable' to runtime because it is not 'ImplicitlyCopyable'}}
    # expected-note @+1 {{use 'materialize' to explicitly materialize the value}}
    return NotImplicitlyCopyable.a


# expected-error @+2 {{trait 'TrivialRegisterPassable' requires ancestor trait 'Deinitable', which is opted out with 'not'; remove one of the entries}}
# expected-note @+1 {{opted out here}}
__enum NotDeinitableTrivial(not Deinitable, TrivialRegisterPassable):
    case a


# expected-error @+1 {{expected enum name}}
__enum
