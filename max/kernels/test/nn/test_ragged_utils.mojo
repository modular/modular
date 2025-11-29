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

from buffer import DimList
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from nn._ragged_utils import get_batch_from_row_offsets
from testing import assert_equal
from utils import Index


def test_get_batch_from_row_offsets():
    comptime batch_size = 9
    var storage = InlineArray[UInt32, batch_size + 1](uninitialized=True)
    prefix_sums = LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)](
        storage,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size + 1)),
    )
    prefix_sums[0] = 0
    prefix_sums[1] = 100
    prefix_sums[2] = 200
    prefix_sums[3] = 300
    prefix_sums[4] = 400
    prefix_sums[5] = 500
    prefix_sums[6] = 600
    prefix_sums[7] = 700
    prefix_sums[8] = 800
    prefix_sums[9] = 900

    assert_equal(
        get_batch_from_row_offsets(prefix_sums, 100),
        1,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums, 0),
        0,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums, 899),
        8,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums, 555),
        5,
    )


def main():
    test_get_batch_from_row_offsets()
