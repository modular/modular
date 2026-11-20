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

# Resolving `Chain`'s field evaluates `Box`'s conditional conformance, which
# resolves `Link`'s body and synthesizes its fieldwise init. That init moves a
# `Chain`, so `Chain`'s move and copy initializers must exist before its fields
# are resolved.


trait Tag:
    pass


struct Box[T: Movable](ImplicitlyCopyable, Tag where conforms_to(T, Tag)):
    var x: Int


@fieldwise_init
struct Chain[T: ImplicitlyCopyable & Deinitable](ImplicitlyCopyable):
    var first: Box[Link[Self.T]]


@fieldwise_init
struct Link[T: ImplicitlyCopyable & Deinitable](Movable):
    var head: Self.T
    var tail: Chain[Self.T]


# Resolving `DtorChain`'s field infers `Outer`'s parameter, which evaluates
# `Inner`'s conditional conformance and resolves `DtorLink`'s fields. One of
# those names `DtorChain`'s destructor, so it too must exist before
# `DtorChain`'s fields are resolved.


struct Inner[T: Movable](ImplicitlyCopyable, Tag where conforms_to(T, Tag)):
    var x: Int


struct Outer[T: AnyType](
    ImplicitlyCopyable where conforms_to(T, ImplicitlyCopyable)
):
    var x: Int


struct Holder[T: AnyType](Movable):
    var x: Int


struct DtorChain[T: ImplicitlyCopyable & Deinitable](ImplicitlyCopyable):
    var first: Outer[Inner[DtorLink[Self.T]]]


struct DtorLink[T: ImplicitlyCopyable & Deinitable](Movable):
    var head: Self.T
    var h: Holder[type_of(DtorChain[Self.T].__deinit__)]
