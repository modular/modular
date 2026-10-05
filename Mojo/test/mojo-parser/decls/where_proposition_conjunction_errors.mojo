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
#
# Diagnostics for undischarged identity and conformance constraints name the
# proposition the constraint folded into, so the caller is told what evidence
# is missing.
#
# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-note @below {{cannot prove constraint for candidate}}
def needs_same[A: AnyType, B: AnyType]()
    # expected-note @below {{constraint declared here needs evidence for 'Bool(identical(A, B))'}}
    where A == B:
    pass


def missing_identity[A: AnyType, B: AnyType]():
    # expected-error @below {{invalid call to 'needs_same': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_same[A, B]()


# expected-note @below {{cannot prove constraint for candidate}}
def needs_intable[T: AnyType]()
    # expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, Intable)'}}
    where conforms_to(T, Intable):
    pass


def missing_conformance[T: AnyType]():
    # expected-error @below {{invalid call to 'needs_intable': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_intable[T]()


# expected-note @below {{cannot prove constraint for candidate}}
def needs_chain[A: AnyType, B: AnyType, C: AnyType]()
    # expected-note @below {{constraint declared here needs evidence for 'Bool(identical(B, C)) if Bool(identical(A, C)) else Bool(identical(A, C))'}}
    where A == C and B == C:
    pass


def missing_chain[A: AnyType, B: AnyType, C: AnyType]():
    # expected-error @below {{invalid call to 'needs_chain': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    needs_chain[A, B, C]()
