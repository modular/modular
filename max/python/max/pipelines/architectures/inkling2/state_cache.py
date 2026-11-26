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
"""Per-request slot pools for Inkling 2's short-convolution state.

There is one pool per conv site and layer kind per device. A slot is a ring of
conv inputs indexed by position (see
:mod:`max.nn._graph_ops.state_space.short_conv_ring`) holding
``kernel_size - 1`` entries, plus the most positions a speculative verify can
roll back. Slots are never cleared: taps before position zero
read as zero, so a fresh request never sees the previous tenant's inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final

from max.dtype import DType
from max.nn.kv_cache import RecurrentStateRegion
from typing_extensions import Self

if TYPE_CHECKING:
    # `model_config` imports this module for the regions a cache declares.
    from .model_config import Inkling2TextConfig

# The conv kernel accumulates in float32 regardless of the model dtype.
CONV_STATE_DTYPE: Final = DType.float32


class ConvSite(IntEnum):
    """The four convolution sites of a layer, in pool order."""

    K = 0
    V = 1
    ATTN_OUT = 2
    MLP_OUT = 3


@dataclass(frozen=True)
class ConvStateLayout:
    """Per-device channel widths of each layer's four sites.

    The widths come in :class:`ConvSite` order.
    """

    ring_len: int
    """Entries per slot: ``kernel_size - 1`` plus the largest rollback."""

    layers: tuple[tuple[int, int, int, int], ...]
    is_local: tuple[bool, ...]
    """Which layer visits are sliding-window, in :attr:`layers` order. A
    visit's K and V widths follow this, so it also splits the pool into
    uniformly shaped leaves."""

    @property
    def kinds(self) -> tuple[bool, ...]:
        """The layer kinds present, global first for a stable leaf order."""
        return tuple(
            kind for kind in (False, True) if kind in set(self.is_local)
        )

    def layers_of_kind(self, is_local: bool) -> tuple[int, ...]:
        """Visit indices of one kind, in order."""
        return tuple(
            idx for idx, local in enumerate(self.is_local) if local == is_local
        )

    def regions(self) -> tuple[RecurrentStateRegion, ...]:
        """The pool leaves a request occupies, per device.

        One leaf per (site, layer kind): a leaf is uniformly shaped and only
        the K and V widths differ, and only between the two kinds.
        """
        return tuple(
            RecurrentStateRegion(
                leaf_id=(
                    f"conv/{'local' if is_local else 'global'}/{site.name.lower()}"
                ),
                num_layers=len(self.layers_of_kind(is_local)),
                row_shape=(self.ring_len, self.layers[first][site]),
                dtype=CONV_STATE_DTYPE,
            )
            for is_local in self.kinds
            for site in ConvSite
            for first in (self.layers_of_kind(is_local)[0],)
        )

    def row_for(self, visit_idx: int, site: ConvSite) -> tuple[int, int]:
        """Returns the leaf a visit's site lives in, and its row within it.

        The leaf is a position among :meth:`regions`, which is how a device's
        ``leaves`` tuple is addressed.
        """
        is_local = self.is_local[visit_idx]
        ordinal = self.layers_of_kind(is_local).index(visit_idx)
        leaf = self.kinds.index(is_local) * len(ConvSite) + int(site)
        return leaf, ordinal

    @classmethod
    def from_config(
        cls,
        text_config: Inkling2TextConfig,
        *,
        tp_size: int = 1,
        max_rollback: int = 0,
    ) -> Self:
        """Derives the layout from the checkpoint config.

        ``max_rollback`` is the most positions a verify step can reject. A
        verify commits every row it scores, so the ring keeps that many more
        entries than the kernel reads: the rejected rows then land in slots
        the accepted history does not occupy.
        """
        is_local = tuple(
            text_config.is_local_attention(layer_idx)
            for layer_idx in text_config.visit_layer_ids()
        )
        # The residual sites are not channel-sharded: each rank convolves a
        # full-width partial sum and the ranks are summed afterwards.
        layers = tuple(
            (
                text_config.kv_conv_dim(local) // tp_size,
                text_config.kv_conv_dim(local) // tp_size,
                text_config.hidden_size,
                text_config.hidden_size,
            )
            for local in is_local
        )
        return cls(
            ring_len=text_config.sconv_kernel_size - 1 + max_rollback,
            layers=layers,
            is_local=is_local,
        )
