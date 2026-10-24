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

# RUN: mojo doc %s | FileCheck %s

# Signatures that name a parametric alias print the alias, not its expansion
# (MOCO-3617, MOCO-5063).

from layout import RowMajorLayout
from std.utils.coord import ComptimeInt


@fieldwise_init
struct Pair[A: AnyType, B: AnyType](Movable):
    pass


comptime _Hidden[T: AnyType] = Pair[T, T]
comptime Doubled[T: AnyType] = Pair[T, _Hidden[T]]
comptime Sized[T: AnyType, n: Int] = Pair[T, Pair[T, T]]
comptime Flagged[T: AnyType, *, flag: Bool] = Pair[T, T]
comptime Defaulted[T: AnyType, n: Int = 4] = Pair[T, T]
comptime Inferred[mut: Bool, //, o: Origin[mut=mut]] = Pair[Int, Int]
comptime Packed[*Ts: AnyType] = Pair[Int, Int]
comptime Same[T: AnyType] = T


# CHECK-DAG: "signature": "def make_doubled[T: AnyType]() -> Doubled[T]"
def make_doubled[T: AnyType]() -> Doubled[T]:
    return Doubled[T]()


# CHECK-DAG: "signature": "def make_sized[T: AnyType, n: Int]() -> Sized[T, n]"
def make_sized[T: AnyType, n: Int]() -> Sized[T, n]:
    return Sized[T, n]()


# CHECK-DAG: "signature": "def make_concrete() -> Doubled[Int]"
def make_concrete() -> Doubled[Int]:
    return Doubled[Int]()


# Alias parameters print like struct parameters: keyword-only ones are named,
# and defaulted and infer-only ones are elided.
# CHECK-DAG: "signature": "def make_flagged[T: AnyType]() -> Flagged[T, flag=True]"
def make_flagged[T: AnyType]() -> Flagged[T, flag=True]:
    return Flagged[T, flag=True]()


# CHECK-DAG: "signature": "def make_defaulted[T: AnyType]() -> Defaulted[T]"
def make_defaulted[T: AnyType]() -> Defaulted[T]:
    return Defaulted[T]()


# CHECK-DAG: "signature": "def make_inferred() -> Inferred[MutAnyOrigin]"
def make_inferred() -> Inferred[MutAnyOrigin]:
    return Inferred[MutAnyOrigin]()


# Aliases whose names start with `_` are implementation details and stay
# expanded.
# CHECK-DAG: "signature": "def make_hidden[T: AnyType]() -> Pair[T, T]"
def make_hidden[T: AnyType]() -> _Hidden[T]:
    return _Hidden[T]()


# The reported shape: an alias with a variadic parameter, applied to a pack.
# CHECK-DAG: "signature": "def make_packed[*Ts: AnyType]() -> Packed[*Ts]"
def make_packed[*Ts: AnyType]() -> Packed[*Ts]:
    return Packed[*Ts]()


# Argument types keep their alias too, parametric or not.
comptime IntPair = Pair[Int, Int]


# CHECK-DAG: "signature": "def take_doubled[T: AnyType](d: Doubled[T])"
def take_doubled[T: AnyType](d: Doubled[T]):
    pass


# CHECK-DAG: "signature": "def take_concrete(d: Doubled[Int], p: IntPair)"
def take_concrete(d: Doubled[Int], p: IntPair):
    pass


# CHECK-DAG: "signature": "def take_flagged[T: AnyType](f: Flagged[T, flag=False])"
def take_flagged[T: AnyType](f: Flagged[T, flag=False]):
    pass


# CHECK-DAG: "signature": "def take_hidden[T: AnyType](h: Pair[T, T])"
def take_hidden[T: AnyType](h: _Hidden[T]):
    pass


# Standard library and kernel aliases. Scalar aliases carry no sugar, since
# the printer spells `SIMD[_, 1]` as `Scalar[dtype]` or a named scalar anyway.
# CHECK-DAG: "signature": "def take_scalar[dtype: DType](x: Scalar[dtype], y: Float32)"
def take_scalar[dtype: DType](x: Scalar[dtype], y: Float32):
    pass


# CHECK-DAG: "signature": "def take_pointer[T: AnyType](p: UnsafePointer[T, MutAnyOrigin])"
def take_pointer[T: AnyType](p: UnsafePointer[T, MutAnyOrigin]):
    pass


# CHECK-DAG: "signature": "def take_layout(l: RowMajorLayout[ComptimeInt[Int(4)], ComptimeInt[Int(8)]])"
def take_layout(l: RowMajorLayout[ComptimeInt[4], ComptimeInt[8]]):
    pass


# TODO(MOCO-3617): Aliases used as parameters of another type expand, because
# the upcast to the parameter's trait type drops the alias.
# CHECK-DAG: "signature": "def take_nested(p: Pair[Pair[Int, Pair[Int, Int]], Pair[Int, Int]])"
def take_nested(p: Pair[Doubled[Int], IntPair]):
    pass


# Binding the `_` adds a parameter, so the bound type no longer matches the
# alias and prints expanded.
# CHECK-DAG: "signature": "def take_partial(p: Pointer[Int])"
def take_partial(p: UnsafePointer[mut=True, Int, _]):
    pass


# Aliases that don't name a struct type print what they compute.
# CHECK-DAG: "signature": "def make_same[T: Movable](var x: T) -> T"
def make_same[T: Movable](var x: T) -> Same[T]:
    return x^
