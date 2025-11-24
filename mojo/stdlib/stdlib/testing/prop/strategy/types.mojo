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
    comptime Value = SIMD[Self.dtype, Self.size]

    var _min: Scalar[Self.dtype]
    var _max: Scalar[Self.dtype]

    fn __init__(
        out self,
        *,
        min: Scalar[Self.dtype] = Scalar[Self.dtype].MIN_FINITE,
        max: Scalar[Self.dtype] = Scalar[Self.dtype].MAX_FINITE,
    ):
        self._min = min
        self._max = max

    # TODO: Provide better more consistent "corner case" values
    # e.g. 0, -1, 1, max, min, max-1, min+1, etc...
    fn value(mut self, mut rng: Rng) raises -> Self.Value:
        var result = SIMD[Self.dtype, Self.size](0)

        @parameter
        for i in range(Self.size):
            result[i] = rng.rand_scalar[Self.dtype](
                min=self._min, max=self._max
            )
        return result


# ===----------------------------------------------------------------------=== #
# List
# ===----------------------------------------------------------------------=== #


__extension List:
    fn strategy[
        StrategyType: Strategy
    ](
        var strategy: StrategyType,
        *,
        min_len: Int = 0,
        max_len: Int = Int.MAX,
    ) raises -> _ListStrategy[StrategyType]:
        """Returns a strategy for generating lists with random elements.

        Parameters:
            StrategyType: The type of the strategy to use for generating random elements.

        Args:
            strategy: The strategy to use for generating random elements.
            min_len: The minimum length of the list.
            max_len: The maximum length of the list.

        Returns:
            A strategy for generating lists with random elements.

        Raises:
            If the minimum length is greater than the maximum length.
        """
        return _ListStrategy(strategy^, min_len=min_len, max_len=max_len)


struct _ListStrategy[T: Strategy](Movable, Strategy):
    comptime Value = List[Self.T.Value]

    var _strat: Self.T
    var _min_len: Int
    var _max_len: Int

    fn __init__(
        out self,
        var strategy: Self.T,
        *,
        min_len: Int = 0,
        max_len: Int = Int.MAX,
    ) raises:
        if min_len < 0 or min_len > max_len:
            raise Error("Invalid min/max for list length")

        # TODO: Make this configurable for other collection types via
        # a property test config value.
        comptime MAX_LIST_SIZE = 100

        self._strat = strategy^
        self._min_len = min_len
        self._max_len = min(max_len, MAX_LIST_SIZE)

    # TODO: Provide more consistent "corner case" values.
    # Empty list, single element list, max size list, etc...
    fn value(mut self, mut rng: Rng) raises -> Self.Value:
        var result = List[Self.T.Value](capacity=self._min_len)

        while len(result) < self._min_len:
            result.append(self._strat.value(rng))

        var average_len = Float64(self._min_len + self._max_len) / 2.0

        # geometric distribution
        var probability = 1.0 - 1.0 / (1.0 + average_len)
        while len(result) < self._max_len:
            var should_append = rng.rand_bool(true_probability=probability)
            if not should_append:
                break

            result.append(self._strat.value(rng))

        return result^
