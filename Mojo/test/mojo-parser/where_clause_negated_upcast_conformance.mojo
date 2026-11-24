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

##===----------------------------------------------------------------------===##
# Tests that a lone negated conformance assumption discharges the same
# requirement when an upcast wraps the conformance's type value. Passing a
# `Derived` parameter where a `Base` one is expected upcasts it, and the
# assumption and goal must still agree on one representation.
##===----------------------------------------------------------------------===##


trait Base:
    pass


trait Derived(Base):
    pass


trait Other:
    pass


comptime NotOther[T: Base] = not conforms_to(T, Other)
comptime NotConforming[T: Base, Tr: type_of(AnyType)] = not conforms_to(T, Tr)


# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{constraint declared here needs evidence for 'not conforms_to(T, Other).__bool__()'}}
def needs_not_other[T: Base]() where not conforms_to(T, Other):
    pass


# COM: Only the goal carries the upcast, from binding `T` to `needs_not_other`.
def upcast_goal[T: Derived]() where not conforms_to(T, Other):
    needs_not_other[T]()


# COM: The `Base`-typed alias wraps `T` in an upcast on both the assumption and
# COM: the goal, so the goal restates the assumption.
# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{constraint declared here needs evidence for 'not conforms_to(T, Other).__bool__()'}}
def needs_wrapped[T: Derived]() where NotOther[T]:
    pass


def wrapped_restated[T: Derived]() where NotOther[T]:
    needs_wrapped[T]()


# COM: As above, against a trait parameter that names no trait symbols.
# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @+3 {{constraint declared here needs evidence for 'not conforms_to(T, Tr).__bool__()'}}
def needs_wrapped_opaque[
    T: Derived, Tr: type_of(AnyType)
]() where NotConforming[T, Tr]:
    pass


def wrapped_opaque_restated[
    T: Derived, Tr: type_of(AnyType)
]() where NotConforming[T, Tr]:
    needs_wrapped_opaque[T, Tr]()


# COM: Without the assumption each requirement is unproven, so the calls above
# COM: are discharged by their assumptions rather than skipped.
def unproven[T: Derived]():
    # expected-error @below {{invalid call to 'needs_not_other': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_not_other[T]()


def wrapped_unproven[T: Derived]():
    # expected-error @below {{invalid call to 'needs_wrapped': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_wrapped[T]()


def wrapped_opaque_unproven[T: Derived, Tr: type_of(AnyType)]():
    # expected-error @below {{invalid call to 'needs_wrapped_opaque': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_wrapped_opaque[T, Tr]()
