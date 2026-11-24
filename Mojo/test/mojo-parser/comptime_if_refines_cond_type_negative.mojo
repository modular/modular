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

# A conditional type `A if c else B` must refine only when the assumptions
# decide `c`, and only to the arm they select.
#
# RUN: %parse-mojo-isolated -verify-diagnostics %s


@fieldwise_init
struct Foo(Movable):
    def foo(self):
        pass


@fieldwise_init
struct Bar(Movable):
    def bar(self):
        pass


def unguarded[c: Bool](arg: Foo if c else Bar):
    arg.foo()  # expected-error {{has no attribute 'foo'}}


def unrelated_condition[c: Bool, d: Bool](arg: Foo if c else Bar):
    comptime if d:
        arg.foo()  # expected-error {{has no attribute 'foo'}}


def wrong_arm[c: Bool](arg: Foo if c else Bar):
    comptime if c:
        arg.bar()  # expected-error {{has no attribute 'bar'}}
    else:
        arg.foo()  # expected-error {{has no attribute 'foo'}}


# Converting a concrete arm into the conditional type needs the assumptions to
# select that same arm.
# expected-note @below {{function declared here}}
def take[c: Bool](var arg: Foo if c else Bar):
    pass


def introduce_unguarded[c: Bool]():
    take[c](Foo())  # expected-error {{cannot be converted from 'Foo'}}


def introduce_wrong_arm[c: Bool]():
    comptime if c:
        take[c](Bar())  # expected-error {{cannot be converted from 'Bar'}}
    else:
        take[c](Foo())  # expected-error {{cannot be converted from 'Foo'}}
