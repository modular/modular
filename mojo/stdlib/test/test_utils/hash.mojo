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

from hashlib.hasher import Hasher

from bit import pop_count
from builtin._location import __call_location
from testing import assert_true


def dif_bits(i1: UInt64, i2: UInt64) -> Int:
    """Computes the number of differing bits between two integers.

    Args:
        i1: First integer.
        i2: Second integer.

    Returns:
        The number of bits that differ between the two integers.
    """
    return Int(pop_count(i1 ^ i2))


@always_inline
def assert_dif_hashes(hashes: List[UInt64], upper_bound: Int):
    """Asserts that all pairs of hashes differ by more than the upper bound.

    Args:
        hashes: List of hash values to compare.
        upper_bound: Minimum number of differing bits required between hashes.
    """
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            var diff = dif_bits(hashes[i], hashes[j])
            assert_true(
                diff > upper_bound,
                "Index: {}:{}, diff between: {} and {} is: {}".format(
                    i, j, hashes[i], hashes[j], diff
                ),
                location=__call_location(),
            )


@always_inline
def assert_fill_factor[
    label: String, HasherType: Hasher
](words: List[String], num_buckets: Int, lower_bound: Float64):
    """Asserts that the hash function achieves a minimum fill factor.

    Parameters:
        label: Label for the test output.
        HasherType: Type of hasher to use.

    Args:
        words: List of strings to hash.
        num_buckets: Number of buckets to distribute hashes into.
        lower_bound: Minimum required fill factor (0.0 to 1.0).
    """
    # A perfect hash function is when the number of buckets is equal to number of words
    # and the fill factor results in 1.0
    var buckets = List[Int](0) * num_buckets
    for w in words:
        var h = hash[HasherType=HasherType](w)
        buckets[h % num_buckets] += 1
    var unfilled = 0
    for v in buckets:
        if v == 0:
            unfilled += 1

    var fill_factor = 1.0 - Float64(unfilled) / Float64(num_buckets)
    assert_true(
        fill_factor >= lower_bound,
        "Fill factor for {} is {}, provided lower bound was {}".format(
            label, fill_factor, lower_bound
        ),
        location=__call_location(),
    )
