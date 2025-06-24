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

from internal_utils import HostNDBuffer
from nn._ragged_utils import get_batch_from_row_offsets
from testing import assert_equal


def test_get_batch_from_row_offsets():
    batch_size = 9
    prefix_sums = HostNDBuffer[DType.uint32, 1]((batch_size + 1,))
    prefix_sums.tensor[0] = 0
    prefix_sums.tensor[1] = 100
    prefix_sums.tensor[2] = 200
    prefix_sums.tensor[3] = 300
    prefix_sums.tensor[4] = 400
    prefix_sums.tensor[5] = 500
    prefix_sums.tensor[6] = 600
    prefix_sums.tensor[7] = 700
    prefix_sums.tensor[8] = 800
    prefix_sums.tensor[9] = 900

    assert_equal(
        get_batch_from_row_offsets(prefix_sums.tensor, 100),
        1,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums.tensor, 0),
        0,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums.tensor, 899),
        8,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums.tensor, 555),
        5,
    )

    _ = prefix_sums^


def main():
    test_get_batch_from_row_offsets()
