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

from buffer import NDBuffer


@always_inline
fn get_batch_from_row_offsets(
    row_offsets: NDBuffer[DType.uint32, 1, *_], tok_idx: Int
) -> Int:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    var row_offsets_size = row_offsets.dim[0]()

    debug_assert(
        tok_idx >= 0 and tok_idx < Int(row_offsets[row_offsets_size - 1]),
        "tok_idx is out of range of row_offsets",
    )

    var low: UInt = 0
    var high: UInt = row_offsets_size - 1
    while low + 1 != high:
        var mid = (low + high) // 2

        if tok_idx >= Int(row_offsets[mid]):
            low = mid
        elif tok_idx < Int(row_offsets[mid]):
            high = mid

    return Int(low)
