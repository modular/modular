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

# A copy into or out of a non-default address space must be trivial. In generic
# code, the `where` clauses of the function can prove that it is.


def take[T: ImplicitlyCopyable](value: T):
    pass


def copy_out[
    T: ImplicitlyCopyable, AS: AddressSpace
](ref[AS] x: T) -> T where downcast[T, Copyable].__copy_ctor_is_trivial:
  # FIXME: MOCO-5480
  # expected-error @+1 {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    take(x)
  # FIXME: MOCO-5480
  # expected-error @+1 {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    return x


def copy_in[
    T: ImplicitlyCopyable, o: Origin[mut=True], AS: AddressSpace
](ref[o, AS] dst: T, src: T) where downcast[
    T, Copyable
].__copy_ctor_is_trivial:
  # FIXME: MOCO-5480
  # expected-error @+1 {{value of type 'T' cannot be copied or moved into a non-default address space}}
    dst = src


def copy_out_conjunction[
    T: ImplicitlyCopyable, AS: AddressSpace
](ref[AS] x: T) -> T where conforms_to(T, Copyable) and downcast[
    T, Copyable
].__copy_ctor_is_trivial:
  # FIXME: MOCO-5480
  # expected-error @+1 {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    return x


def copy_out_trivial_register_passable[
    T: ImplicitlyCopyable, AS: AddressSpace
](ref[AS] x: T) -> T where conforms_to(T, TrivialRegisterPassable):
    return x


def copy_out_unproven[T: ImplicitlyCopyable, AS: AddressSpace](ref[AS] x: T):
    # expected-error @+1 {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    take(x)


def copy_in_unproven[
    T: ImplicitlyCopyable, o: Origin[mut=True], AS: AddressSpace
](ref[o, AS] dst: T, src: T):
    # expected-error @+1 {{cannot be copied or moved into a non-default address space}}
    dst = src


# Each disjunct proves the copy trivial, but the prover does not split a
# disjunction in the assumptions into cases.
def copy_out_disjunction[
    T: ImplicitlyCopyable, AS: AddressSpace
](ref[AS] x: T) where conforms_to(T, TrivialRegisterPassable) or downcast[
    T, Copyable
].__copy_ctor_is_trivial:
    # expected-error @+1 {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    take(x)
