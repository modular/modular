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

# RUN: %parse-mojo-isolated -split-input-file %s | FileCheck %s

# Calling a constructor through a parametric alias whose parameters are not all
# bound, so the struct's parameters come from the alias rather than from the
# call's result type.

# An alias that binds a leading struct parameter: its generator inputs don't
# line up index-for-index with the struct's parameters (`T` is input 0 of `A`
# but parameter 1 of `S`), as in `ImmPointer[origin=...](_mlir_value=...)`.


struct S[a: Int, T: AnyType]:
    def __init__(out self, v: Self.T):
        pass


comptime A[T: AnyType] = S[0, T]


# CHECK-LABEL: lit.fn @"test_partially_bound_alias
def test_partially_bound_alias(x: Int):
    # CHECK: lit.call {{.*}}__init__
    _ = A(x)


# // -----

# An `__init__` with its own parameters (`U`): the full signature has inputs for
# both the struct parameters and `U`, as in the `Pointer.__init__` overloads
# that retype `out self`.


struct S[T: AnyType]:
    def __init__[U: AnyType](out self: S[U], u: U):
        pass


comptime A[T: AnyType] = S[T]


# CHECK-LABEL: lit.fn @"test_parametric_init
def test_parametric_init(x: Int):
    # CHECK: lit.call {{.*}}__init__
    _ = A(x)


# // -----

# An unbound parameter used only inside a struct parameter value: `tabulate`
# leaves its infer-only `ToT` unbound, and `ToT` appears only inside the
# `#kgen.param_list.tabulate` attribute, as in `Args.map[ToPointer]()` in
# `_create_dynamic_pack`.


struct Box[T: AnyType](ImplicitlyCopyable):
    pass


# CHECK-LABEL: lit.fn @"test_nested_unbound_param
def test_nested_unbound_param[
    ArgTrait: type_of(AnyType), //, Args: TypeList[Trait=ArgTrait, ...]
]():
    comptime ToBox[T: ArgTrait]: ImplicitlyCopyable = Box[T]
    comptime L = Args.map[ToBox]()


# // -----

# A struct parameter whose type depends on another (`origin` on `mut`), with an
# overload taking an auto-parameterized `other: P`, whose own parameters'
# types refer to each other, as in `UnsafePointer(to=x)`.


struct P[mut: Bool, //, T: AnyType, origin: Origin[mut=mut]]:
    def __init__(out self, *, ref[Self.origin] to: Self.T):
        pass

    def __init__(out self, other: P):
        pass


comptime A[mut: Bool, //, T: AnyType, origin: Origin[mut=mut]] = P[T, origin]


# CHECK-LABEL: lit.fn @"test_dependent_param_type
def test_dependent_param_type():
    var x = 1
    # CHECK: lit.call {{.*}}__init__
    _ = A(to=x)


# // -----

# A method parameter whose type refers to a struct parameter (`o`'s type uses
# `Self.mut`), through an alias with fewer inputs than the struct has
# parameters, as in `SMemTile[...](...)` against the `TileTensor.__init__`
# overloads.


struct P[mut: Bool, //, T: AnyType, origin: Origin[mut=mut]]:
    def __init__(out self, v: Self.T):
        pass

    def __init__[o: Origin[mut=Self.mut]](out self, ref[o] v: Self.T, k: Int):
        pass


comptime A[T: AnyType] = P[T, MutAnyOrigin]


# CHECK-LABEL: lit.fn @"test_method_param_refers_to_struct_param
def test_method_param_refers_to_struct_param(x: Int):
    # CHECK: lit.call {{.*}}__init__
    _ = A(x)


# // -----

# A defaulted alias parameter keeps its default, so it doesn't need to be
# inferred.


struct Q[T: AnyType, *, a: Int = 0]:
    def __init__(out self, v: Self.T):
        pass


comptime B[T: AnyType, *, a: Int = 0] = Q[T, a=a]


# CHECK-LABEL: lit.fn @"test_defaulted_alias_param
def test_defaulted_alias_param():
    # CHECK: lit.call {{.*}}__init__
    _ = B(1)
