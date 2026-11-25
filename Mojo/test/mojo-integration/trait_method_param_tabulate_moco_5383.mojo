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

# RUN: %mojo %s

# Regression test for MOCO-5383: a trait method takes a `Tuple` whose length
# depends on `Self` and whose element type uses the method's own parameter.
# Calling it through the trait bound has to find the conformer's witness.

from std.testing import assert_equal


comptime Vecs[width: Int, T: Splatted] = Tuple[
    *TypeList.splat[Trait=AnyType, T.N, SIMD[DType.float32, width]]()
]


trait Splatted:
    comptime N: Int

    def total[width: Int](self, operands: Vecs[width, Self]) -> Float32:
        ...


@fieldwise_init
struct Three(Splatted):
    comptime N = 3

    def total[width: Int](self, operands: Vecs[width, Self]) -> Float32:
        comptime V = SIMD[DType.float32, width]
        return (
            rebind[V](operands[0]).reduce_add()
            + 10 * rebind[V](operands[1]).reduce_add()
            + 100 * rebind[V](operands[2]).reduce_add()
        )


def total_of[T: Splatted](t: T) -> Float32:
    return t.total[2](
        rebind[Vecs[2, T]](
            Tuple(
                SIMD[DType.float32, 2](1, 2),
                SIMD[DType.float32, 2](3, 4),
                SIMD[DType.float32, 2](5, 6),
            )
        )
    )


# The same shape built by mapping an associated `TypeList`.
comptime _Vec[width: Int, T: AnyType]: AnyType = SIMD[DType.float32, width]
comptime MappedVecs[width: Int, T: Mapped] = Tuple[
    *T.Types.map[_Vec[width, _]]()
]


trait Mapped:
    comptime __types: TypeList[Trait=AnyType, _]._mlir_type
    comptime Types: TypeList[Trait=AnyType, Self.__types]

    def first[width: Int](self, operands: MappedVecs[width, Self]) -> Float32:
        ...


@fieldwise_init
struct One(Mapped):
    comptime Types = TypeList.of[Trait=AnyType, Int]()
    comptime __types = Self.Types.values

    def first[width: Int](self, operands: MappedVecs[width, Self]) -> Float32:
        return rebind[SIMD[DType.float32, width]](operands[0]).reduce_add()


def first_of[T: Mapped](t: T) -> Float32:
    return t.first[4](
        rebind[MappedVecs[4, T]](Tuple(SIMD[DType.float32, 4](1, 2, 3, 4)))
    )


def main() raises:
    assert_equal(total_of(Three()), 1173)
    assert_equal(first_of(One()), 10)
