# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licenxse is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.testing.prop.random import Rng
from std.testing.prop.strategy import Strategy
from std.testing.prop._errors import PLAYBACK_EXHAUSTED
from std.collections import Set
from std.sys.info import simd_width_of
from std.os import abort

comptime Stream = List[UInt64]
comptime StreamChunk[width: Int] = SIMD[DType.uint64, width]


# TODO: add larger block sizes and handle mulitples of simd width
def block_sizes() -> List[Int]:
    var l = List[Int]()
    var start = simd_width_of[DType.uint64]()

    while start > 0:
        l.append(start)
        start >>= 1

    return l^


def stream_is_better(stream_a: Stream, stream_b: Stream) -> Bool:
    if len(stream_a) != len(stream_b):
        return len(stream_a) < len(stream_b)

    for a, b in zip(stream_a, stream_b):
        if a != b:
            return a < b

    # probably shouldn't happen that the streams are the same
    return False


@fieldwise_init
struct ShrinkResult[T: Copyable & ImplicitlyDestructible & Writable](Copyable):
    var value: Self.T
    var runs: Int


struct Shrinker[
    StrategyType: Strategy, f: def(var StrategyType.Value) capturing raises
]:

    """Property test input reducer.
    When a test fails, try and find a minimal reproduction of the failure.
    """

    var best: Stream
    var expected_error: Error
    var max_reductions: Int
    var cache: Set[Stream]
    var strategy: Self.StrategyType

    @doc_hidden
    def __init__(
        out self,
        var strategy: Self.StrategyType,
        var stream: Stream,
        var error: Error,
        max_reductions: Int = 1000,
    ):
        self.best = stream^
        self.expected_error = error^
        self.max_reductions = max_reductions
        self.strategy = strategy^

        self.cache = {}

    def shrink(mut self) -> ShrinkResult[Self.StrategyType.Value]:
        """Find the minimal interesting input.

        Returns:
            The minimal interesting input found during shrinking.
        """

        var reductions = 0
        var runs = 0
        while reductions < self.max_reductions:
            runs += 1
            var stream = self.best.copy()

            if not (
                self.delete_chunk_pass(stream)
                or self.zero_chunk_pass(stream)
                or self.reduce_chunk_pass(stream)
            ):
                break

            reductions += 1

        var rng = Rng(self.best.copy())

        try:
            return ShrinkResult(self.strategy.value(rng), runs)
        except:
            # We know best works so this shouldn't happen
            abort("Failed to build value from best stream")

    def try_stream(mut self, stream: Stream) -> Bool:
        if stream in self.cache:
            return False

        self.cache.add(stream)
        var rng = Rng(stream.copy())
        try:
            var val = self.strategy.value(rng)
            Self.f(val^)
            return False
        except e:
            var used = Stream(stream[: rng.playback_index])
            if e == self.expected_error and stream_is_better(used, self.best):
                self.best = used^
                return True
            return False

    def multi_element_pass[
        op: def[width: Int, //](StreamChunk[width]) thin -> StreamChunk[width]
    ](mut self, stream: Stream) -> Bool:
        comptime for size in block_sizes():
            if size <= len(self.best):
                for i in range(0, len(self.best) - size + 1, size):
                    var cpy = self.best.copy()
                    var ptr = cpy.unsafe_ptr() + i
                    ptr.store[width=size](0, op(ptr.load[width=size]()))

                    if self.try_stream(cpy):
                        return True

        return False

    def binary_search_value(
        mut self, i: Int, var hi: UInt64, mut improved: Bool
    ):
        var lo = UInt64(0)
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            var cpy = self.best.copy()
            cpy[i] = mid

            if self.try_stream(cpy):
                improved = True
                # best was updated by try_stream, use its value
                hi = self.best[i]
            else:
                if mid == lo:
                    break
                lo = mid

    def reduce_chunk_pass(mut self, stream: Stream) -> Bool:
        var improved = False

        # TODO: Use SIMD/chunked for this as well
        for i in range(len(self.best)):
            var hi = self.best[i]
            if hi == 0:
                continue

            self.binary_search_value(i, hi, improved)

        return improved

    def zero_chunk_pass(mut self, stream: Stream) -> Bool:
        @always_inline
        def zero[width: Int, //](s: StreamChunk[width]) -> StreamChunk[width]:
            return 0

        return self.multi_element_pass[zero](stream)

    @staticmethod
    def delete_chunk(stream: Stream, start: Int, chunk_size: Int) -> Stream:
        var cpy = Stream()
        for j in range(start):
            cpy.append(stream[j])
        for j in range(start + chunk_size, len(stream)):
            cpy.append(stream[j])
        return cpy^

    def delete_chunk_pass(mut self, stream: Stream) -> Bool:
        var n = len(stream)
        var chunk_size = n - 1

        while chunk_size >= 1:
            for i in range(n - chunk_size + 1):
                if self.try_stream(Self.delete_chunk(stream, i, chunk_size)):
                    return True

                # Fallback: binary search the predecessor value with the
                # chunk deleted. This handles "length + items" generation
                # patterns where deleting items without adjusting the
                # preceding length causes a playback overrun.
                if i > 0 and stream[i - 1] > 0:
                    var hi = stream[i - 1]
                    var improved = False

                    self.binary_search_value(i, hi, improved)

                    if improved:
                        return True

            chunk_size -= 1

        return False

    def sort_chunk_pass(mut self, stream: Stream) -> Bool:
        # TODO: Try and find a smaller ordering of the existing data
        return False

    def redistribute_pass(mut self, stream: Stream) -> Bool:
        # TODO: Adjust adjacent while keeping the sum of them constant
        return False

    def swap_chunk_pass(mut self, stream: Stream) -> Bool:
        # TODO: Swap adjacent chunks
        return False
