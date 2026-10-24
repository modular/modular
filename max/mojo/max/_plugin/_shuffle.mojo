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

"""Legalizes warp shuffles onto a 32-bit lane shuffle.

A GPU whose hardware moves one 32-bit word between lanes at a time can still
back every shuffle hook with it: its plugin passes the word shuffle to
`shuffle_words`, which splits, packs or widens any value the public
`shuffle_*` functions accept into whole words and reassembles the result. The
CUDA and HIP plugins build their shuffles this way, since `shfl.sync` and
`ds.bpermute` both move one 32-bit word. A plugin with native typed and vector
shuffles, such as Metal's `simd_shuffle`, calls them directly instead.
"""

from std.memory import bitcast
from std.sys import size_of

comptime WordShuffleFnType = def(
    mask: UInt, word: UInt32, offset: UInt32
) thin -> UInt32
"""Copies `word` from lane `offset`, honoring `mask` where the target can."""


@inline(.always)
def shuffle_words[
    dtype: DType, simd_width: SIMDLength, //, word_fn: WordShuffleFnType
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    """Shuffles `val` one 32-bit word at a time through `word_fn`.

    A 32-bit value, or a pair of 16-bit elements, is one word. A wider value
    is split into its even and odd words, which are shuffled separately and
    interleaved back. A lone element narrower than a word is widened to fill
    one. `Bool` travels as an `Int32`.

    Parameters:
        dtype: The element type of `val`.
        simd_width: The number of elements in `val`.
        word_fn: The lane shuffle for one 32-bit word.

    Args:
        mask: The lanes taking part; passed through to `word_fn`.
        val: The value each lane contributes.
        offset: The source lane; passed through to `word_fn`.

    Returns:
        The value read from the source lane.
    """
    comptime size = size_of[SIMD[dtype, simd_width]]()

    comptime if dtype == .bool:
        return shuffle_words[word_fn](mask, val.cast[.int32](), offset).cast[
            dtype
        ]()
    elif size == 4:
        return bitcast[dtype, simd_width](
            word_fn(mask, bitcast[.uint32, 1](val), offset)
        )
    elif size < 4:
        comptime if simd_width == 1:
            var widened = SIMD[dtype, 4 // size](val._refine[new_size=1]())
            return shuffle_words[word_fn](mask, widened, offset)[0]
        else:
            return shuffle_words[word_fn](mask, val.join(val), offset).slice[
                simd_width
            ]()
    else:
        var even, odd = bitcast[.uint32, size // 4](val).deinterleave()
        var result = shuffle_words[word_fn](mask, even, offset).interleave(
            shuffle_words[word_fn](mask, odd, offset)
        )
        return bitcast[dtype, simd_width](result)
