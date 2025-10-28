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

from testing.prop.random import Rng

# ===----------------------------------------------------------------------=== #
# SIMD
# ===----------------------------------------------------------------------=== #


__extension SIMD:
    @staticmethod
    fn strategy(
        *,
        min: Scalar[dtype] = Scalar[dtype].MIN_FINITE,
        max: Scalar[dtype] = Scalar[dtype].MAX_FINITE,
    ) -> _SIMDStrategy[dtype, size]:
        """Returns a strategy for generating random SIMD values.

        Args:
            min: The minimum value for the SIMD vector.
            max: The maximum value for the SIMD vector.

        Returns:
            A strategy for generating random SIMD values.
        """
        return _SIMDStrategy[dtype, size](min=min, max=max)


struct _SIMDStrategy[dtype: DType, size: Int](Movable, Strategy):
    alias Value = SIMD[dtype, size]

    var _min: Scalar[dtype]
    var _max: Scalar[dtype]

    fn __init__(
        out self,
        *,
        min: Scalar[dtype] = Scalar[dtype].MIN_FINITE,
        max: Scalar[dtype] = Scalar[dtype].MAX_FINITE,
    ):
        self._min = min
        self._max = max

    # TODO: Provide better more consistent "corner case" values
    # e.g. 0, -1, 1, max, min, max-1, min+1, etc...
    fn value(mut self, mut rng: Rng) raises -> Self.Value:
        var result = SIMD[dtype, size](0)

        @parameter
        for i in range(size):
            result[i] = rng.rand_scalar[dtype](min=self._min, max=self._max)
        return result
