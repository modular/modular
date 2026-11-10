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
"""The last-token gather and logits projection."""

from __future__ import annotations

from collections.abc import Callable

from max import functional as F
from max.dtype import DType
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

__all__ = ["gather_last_tokens", "project_to_logits"]


def _last_tokens(h: Tensor, input_row_offsets: Tensor) -> Tensor:
    """One device's last token of each request."""
    return F.gather(h, input_row_offsets[1:] - 1, axis=0)


def gather_last_tokens(h: Tensor, input_row_offsets: Tensor) -> Tensor:
    """Gathers the last token of each request from the ragged batch, per shard.

    Under data parallelism ``input_row_offsets`` is rebased into each replica's
    local frame, so its values only address that replica's own rows; the gather
    runs on each device's own rows, never through the auto-sharded
    ``F.gather``.
    """
    if not h.is_distributed:
        return _last_tokens(h, input_row_offsets)
    return F.call_on_mesh(_last_tokens, h.mesh, out_specs=h.mapping)(
        h, input_row_offsets
    )


def mask_padded_tail(
    logits: Tensor, vocab_size: int, unpadded_vocab_size: int | None
) -> Tensor:
    """Sends the dummy/padding rows of the vocabulary to negative infinity."""
    if unpadded_vocab_size is None or unpadded_vocab_size >= vocab_size:
        return logits
    device = logits.device
    # Two broadcast scalars rather than a materialized row: keeps a
    # vocab-sized fp32 constant out of the graph.
    keep = F.constant(0.0, DType.float32, device=device).broadcast_to(
        [unpadded_vocab_size]
    )
    drop = F.constant(float("-inf"), DType.float32, device=device).broadcast_to(
        [vocab_size - unpadded_vocab_size]
    )
    return logits + F.concat([keep, drop]).cast(logits.dtype)


def project_to_logits(
    h: Tensor,
    norm: Callable[[Tensor], Tensor],
    lm_head: Callable[[Tensor], Tensor],
    mesh: DeviceMesh | None,
    vocab_size: int,
    unpadded_vocab_size: int | None,
) -> Tensor:
    """Returns the float32 logits of ``h`` on the first device of ``mesh``.

    Vocabulary rows at or past ``unpadded_vocab_size`` are masked to negative
    infinity.
    """
    logits = lm_head(norm(h))
    if mesh is not None:
        logits = logits.to(mesh.devices[0])
    return mask_padded_tail(
        logits.cast(DType.float32), vocab_size, unpadded_vocab_size
    )
