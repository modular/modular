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

from std.math import ceil, iota

from register import *

from std.utils.index import IndexList

# ===----------------------------------------------------------------------=== #
# Arange op
# ===----------------------------------------------------------------------=== #


@always_inline
def arange[
    dtype: DType, simd_width: Int
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    step: Scalar[dtype],
    index: IndexList[1],
) -> SIMD[dtype, simd_width]:
    return start + (iota[dtype, simd_width](Scalar[dtype](index[0])) * step)


@always_inline
def arange_shape[
    dtype: DType
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    step: Scalar[dtype],
) raises -> IndexList[1]:
    if step == 0:
        raise Error("[range] step must be non-zero")

    comptime if start.dtype.is_integral():
        if step > 0 and stop < start:
            raise Error("[range] requires (start <= stop) for positive step")

        if step < 0 and start < stop:
            raise Error("[range] requires (stop <= start) for negative step")

        return IndexList[1](len(range(Int(start), Int(stop), Int(step))))
    else:
        return IndexList[1](Int(ceil(abs(stop - start) / abs(step))))
