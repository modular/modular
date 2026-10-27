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
"""Orders selected sparse-attention entries by position."""

from max import functional as F
from max.dtype import DType
from max.nn.legacy.kernels import top_k_per_row as _top_k_per_row
from max.tensor import Tensor

top_k_per_row = F.functional(_top_k_per_row)
_POSITION_SENTINEL = (1 << 31) - 1


def position_order(indices: Tensor, *, invalid: int = -1) -> Tensor:
    """Reorders a fixed-width, selection-order id row ascending.

    Takes the descending-score picks a bounded selector produces
    (``invalid`` trailing) and returns the same ids in ascending id
    order, ``invalid`` still trailing. The sort key is the id itself
    (sentinel on invalid picks), negated and run through the block-wide
    ``top_k_per_row`` with every pick kept, which orders a row of this
    width in a fraction of the generic ``top_k``'s time. The keys are
    exact in float32 (ids stay below 2**24) and distinct except for the
    sentinels, which all gather ``invalid``.

    Args:
        indices: ``[..., width]`` int32 ids, ``invalid``-padded.
        invalid: The padding sentinel.

    Returns:
        ``[..., width]`` int32, ascending by id, ``invalid`` trailing.
    """
    idx = F.cast(indices, DType.int32)
    width = int(idx.shape[-1])
    if width == 0:
        return idx

    is_valid = idx != invalid
    sentinel = F.constant(_POSITION_SENTINEL, DType.int32, idx.device)
    sort_key = F.reshape(F.where(is_valid, idx, sentinel), [-1, width])
    # ``top_k_per_row`` ranks largest-first; negate so the smallest
    # (earliest) id comes out first.
    rows = sort_key.shape[0]
    _, order = top_k_per_row(
        -F.cast(sort_key, DType.float32),
        F.full([rows], width, dtype=DType.int64, device=idx.device),
        width,
    )
    order = F.reshape(order, idx.shape)
    batch_dims = idx.rank - 1
    gather_idx = F.unsqueeze(order, -1)
    return F.gather_nd(idx, gather_idx, batch_dims=batch_dims)
