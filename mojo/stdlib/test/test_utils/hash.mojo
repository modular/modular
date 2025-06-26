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

from bit import pop_count
from builtin._location import __call_location
from hashlib.hasher import Hasher
from testing import assert_true


def dif_bits(i1: UInt64, i2: UInt64) -> Int:
    return Int(pop_count(i1 ^ i2))


@always_inline
def assert_dif_hashes(hashes: List[UInt64], upper_bound: Int):
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            var diff = dif_bits(hashes[i], hashes[j])
            assert_true(
                diff > upper_bound,
                String("Index: {}:{}, diff between: {} and {} is: {}").format(
                    i, j, hashes[i], hashes[j], diff
                ),
                location=__call_location(),
            )


@always_inline
def assert_fill_factor[
    label: String, HasherType: Hasher
](words: List[String], num_buckets: Int, lower_bound: Float64):
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

    var fill_factor = 1 - unfilled / num_buckets
    assert_true(
        fill_factor >= lower_bound,
        String("Fill factor for {} is {}, provided lower bound was {}").format(
            label, fill_factor, lower_bound
        ),
        location=__call_location(),
    )
