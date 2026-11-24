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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s

# COM: Verify generated wrapper structure

# CHECK: lit.struct.decl @"inflated$def(x: ::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]|{{[0-9a-f]+}}"<*"#__CALL__#": !lit.generator<("x": !Int) -> !Int>

# CHECK: kgen.conformance @{{.*}}::@AnyType {
# CHECK-NEXT: }

# CHECK: lit.fn @"__call__
# CHECK-SAME: kgen.transparent_thunk_callee_expr = #kgen.param.decl.ref<"#__CALL__#">
# CHECK: %1 = lit.call tail[!lit.generator<("x": !Int) -> !Int>: *"#__CALL__#"](%x)
# CHECK: lit.return %1 : !Int
# CHECK: lit.end_fn

# CHECK: kgen.conformance @"##__mojo_closure__##"
# CHECK: kgen.witness "__call__"

# CHECK: lit.trait.decl @"##__mojo_closure__##"

# CHECK: lit.fn @"wrap_fn()"
# CHECK: %__call_result_tmp__ = lit.var.decl "__call_result_tmp__" synth
# CHECK: %0 = lit.call @"inflated$def(x: ::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]|{{[0-9a-f]+}}"::@"__init__()"
# CHECK: %1 = lit.ref.immut %__call_result_tmp__


def top_level(x: Int) -> Int:
    return x


def use_closure[Impl: def(x: Int) -> Int](cb: Impl) -> Int:
    return cb(1)


def wrap_fn() -> Int:
    return use_closure(top_level)


# // -----

# COM: Verify that wrappers are deduplicated

# CHECK-COUNT-1: lit.struct.decl @"inflated$def(x: ::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]|{{[0-9a-f]+}}"


def a(x: Int) -> Int:
    return x


def b(x: Int) -> Int:
    return x * x


def use_closure[Impl: def(x: Int) -> Int](cb: Impl) -> Int:
    return cb(1)


def wrap_fn() -> Int:
    return use_closure(a) + use_closure(b)


# // -----

# COM: fn literals can be converted to closure wrappers.

# COM: The wrapper carries a single conformance, instantiated with the named
# COM: signature it was inflated from, while this section's `use_closure`
# COM: declares the unnamed one - the two bridge through parametric-trait
# COM: matching rather than through a second conformance on the wrapper.
# CHECK: kgen.conformance @"##__mojo_closure__##"
# CHECK-SAME: #kgen.pog_list<[<"", pos, not_vararg>, <"x", pos_or_kw, not_vararg>]>
# CHECK: %cb: !lit.ref<:trait<@"##__mojo_closure__##"{{.*}}#kgen.pog_list<[<"", pos, not_vararg>, <"", pos, not_vararg>]>


def top_level(x: Int) -> Int:
    return x


# COM: Note the lack of an argument name in the signature.
def use_closure[Impl: def(Int) -> Int](cb: Impl) -> Int:
    return cb(1)


def wrap_fn() -> Int:
    var _ = use_closure(top_level)
    var _ = use_closure[type_of(top_level)](top_level)
