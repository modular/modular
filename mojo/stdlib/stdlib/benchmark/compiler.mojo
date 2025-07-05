# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from sys import sizeof
from sys._assembly import inlined_assembly


# ===-----------------------------------------------------------------------===#
# keep
# ===-----------------------------------------------------------------------===#


@always_inline
fn keep(val: Bool):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Args:
      val: The value to not optimize away.
    """
    keep(UInt8(val))


@always_inline
fn keep(val: Int):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Args:
      val: The value to not optimize away.
    """
    keep(Scalar[DType.index](val))


@always_inline
fn keep[dtype: DType, simd_width: Int](val: SIMD[dtype, simd_width]):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      dtype: The `dtype` of the input and output SIMD vector.
      simd_width: The width of the input and output SIMD vector.

    Args:
      val: The value to not optimize away.
    """

    @parameter
    if simd_width > 1:
        for i in range(simd_width):
            # TODO(#27998): Remove the temporary variable.
            var tmp = val[i]
            keep(tmp)
        return

    var tmp = val
    var tmp_ptr = UnsafePointer(to=tmp).origin_cast[
        mut=False, origin=ImmutableAnyOrigin
    ]()

    @parameter
    if (
        sizeof[dtype]()
        <= sizeof[UnsafePointer[SIMD[dtype, simd_width]]._mlir_type]()
    ):
        inlined_assembly[
            "",
            NoneType,
            constraints="+m,r,~{memory}",
            has_side_effect=True,
        ](tmp_ptr, val)
    else:
        inlined_assembly[
            "",
            NoneType,
            constraints="+m,~{memory}",
            has_side_effect=True,
        ](tmp_ptr, tmp_ptr)


@always_inline
fn keep[dtype: AnyType](val: UnsafePointer[dtype]):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      dtype: The type of the input.

    Args:
      val: The value to not optimize away.
    """
    var tmp = val
    var tmp_ptr = UnsafePointer(to=tmp)
    inlined_assembly[
        "",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](tmp_ptr)


@always_inline
fn keep[dtype: AnyTrivialRegType](mut val: dtype):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      dtype: The type of the input.

    Args:
      val: The value to not optimize away.
    """
    var tmp = val
    var tmp_ptr = UnsafePointer(to=tmp)
    inlined_assembly[
        "",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](tmp_ptr)
