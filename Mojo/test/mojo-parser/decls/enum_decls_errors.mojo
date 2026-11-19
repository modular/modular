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
# 1. Reject `()` on an enum case without a payload.
# 2. Select enum wording for shared diagnostics.

__enum EmptyParens:
    # expected-error @+1 {{enum case must have the form 'case name' or 'case name(T, ...)'}}
    case empty()


__enum Stateful:
    case value
    # expected-error @+1 {{enums do not support 'var' fields; store state in a case payload}}
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


# expected-error @+1 {{expected enum name}}
__enum
