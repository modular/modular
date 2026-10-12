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

"""Vocab-parallel embedding for multi-device tensor parallelism."""

from __future__ import annotations

import itertools

from max.experimental import functional as F
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.embedding import Embedding
from max.experimental.sharding import (
    DeviceMapping,
    NamedMapping,
    Partial,
    Replicated,
)
from max.experimental.sharding.action import PerShard
from max.experimental.tensor import Tensor
from max.graph import DimLike


def _masked_gather(
    weight: Tensor,
    indices: Tensor,
    vocab_start: int,
    vocab_end: int,
) -> Tensor:
    """Gathers the rows of ``indices`` from one vocabulary shard of ``weight``.

    Indices outside ``[vocab_start, vocab_end)`` read rows of zeros.
    """
    in_range = (indices >= vocab_start) & (indices < vocab_end)
    gathered = F.gather(weight, (indices - vocab_start) * in_range, axis=0)
    return gathered * in_range.unsqueeze(-1).cast(gathered.dtype)


class VocabParallelEmbedding(Embedding):
    """An embedding whose vocabulary is sharded across devices.

    On a single device this behaves identically to
    :class:`~max.experimental.nn.embedding.Embedding`.  On a multi-device
    mesh the vocabulary dimension (axis 0) is split so each device holds
    a contiguous range of rows.  A lookup gathers from the local shard,
    masks out-of-range indices, and all-reduces the results.
    """

    def __init__(
        self, vocab_size: DimLike, *, dim: DimLike, tp_axis: str | None = None
    ) -> None:
        super().__init__(vocab_size, dim=dim)
        if tp_axis is None:
            tp_axis = TP
        self.weight = self.weight.to(
            NamedMapping(self.weight.mesh, (tp_axis, None))
        )

    def forward(self, indices: Tensor) -> Tensor:
        """Gather the embeddings for the input indices."""
        if not self.weight.is_distributed:
            return F.gather(self.weight, indices, axis=0)
        return self._vocab_parallel_gather(indices)

    def _vocab_parallel_gather(self, indices: Tensor) -> Tensor:
        """Per-shard gather with masking and all-reduce."""
        mesh = self.weight.mesh
        # An uneven split gives the first devices one row more, so each
        # device's vocabulary range comes from its own shard.
        ends = list(
            itertools.accumulate(
                int(shard.shape[0]) for shard in self.weight.local_shards
            )
        )
        if not indices.is_distributed:
            indices = indices.to(
                DeviceMapping(mesh, (Replicated(),) * mesh.ndim)
            )
        # Each device gathers only its own vocabulary rows, so the rows
        # summed across devices are the embedding.
        partial = F.call_on_mesh(
            _masked_gather, mesh, out_specs=DeviceMapping(mesh, (Partial(),))
        )(
            self.weight,
            indices,
            PerShard([0, *ends[:-1]]),
            PerShard(ends),
        )
        return F.allreduce_sum(partial)
