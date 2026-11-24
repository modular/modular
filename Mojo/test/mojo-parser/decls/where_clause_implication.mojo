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
# Tests which callee constraints a caller's where clause proves, disproves, or
# leaves unproven. Each failing call is the control for the case above it: it
# shows the passing call is discharged by its assumption, not skipped.
##===----------------------------------------------------------------------===##


trait A:
    pass


trait B:
    pass


# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{constraint declared here needs evidence for 'a or b'}}
def needs_either[a: Bool, b: Bool]() where a or b:
    pass


def or_weakening[a: Bool, b: Bool]() where a:
    needs_either[a, b]()


def or_unproven[a: Bool, b: Bool]():
    # expected-error @below {{invalid call to 'needs_either': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_either[a, b]()


# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, A & B)'}}
def needs_joint[T: AnyType]() where conforms_to(T, A & B):
    pass


# COM: Neither bound alone proves the packed one.
def joint_conformance[
    T: AnyType
]() where conforms_to(T, A) and conforms_to(T, B):
    needs_joint[T]()


def joint_unproven[T: AnyType]() where conforms_to(T, A):
    # expected-error @below {{invalid call to 'needs_joint': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_joint[T]()


# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{constraint declared here needs evidence for 'a and b'}}
def needs_both[a: Bool, b: Bool]() where a and b:
    pass


# COM: Each conjunct of the goal is already known, so the whole goal is.
def restated_conjunction[a: Bool, b: Bool]() where a and b:
    needs_both[a, b]()


def conjunction_unproven[a: Bool, b: Bool]() where a:
    # expected-error @below {{invalid call to 'needs_both': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_both[a, b]()


# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{constraint declared here needs evidence for '(N == K)'}}
def needs_equal[N: Int, K: Int]() where N == K:
    pass


def restated_equal[N: Int, K: Int]() where N == K:
    needs_equal[N, K]()


def equal_unproven[N: Int, K: Int]():
    # expected-error @below {{invalid call to 'needs_equal': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_equal[N, K]()


# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{constraint declared here needs evidence for '(N != K)'}}
def needs_not_equal[N: Int, K: Int]() where N != K:
    pass


def restated_not_equal[N: Int, K: Int]() where N != K:
    needs_not_equal[N, K]()


def not_equal_unproven[N: Int, K: Int]():
    # expected-error @below {{invalid call to 'needs_not_equal': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_not_equal[N, K]()


# expected-note @below {{function declared here}}
# expected-note @below {{constraint declared here evaluated to False, expected 'not a'}}
def needs_not[a: Bool]() where not a:
    pass


# COM: The assumption contradicts the constraint, which is a harder error than
# COM: the unproven ones above.
def disproved[a: Bool]() where a:
    # expected-error @below {{invalid call to 'needs_not': violated constraint}}
    needs_not[a]()
