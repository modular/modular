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
"""Per-request slot pool for Inkling's short-convolution state: one pool per
conv site per layer per device. The K and V sites are channel-sharded with
the qkvr projection; the two residual sites are full width on every rank.

A slot is a ring of conv inputs indexed by position; see
:mod:`max.nn.state_space.short_conv_ring`. It holds
``kernel_size - 1 + num_draft_tokens`` entries, enough history for a chunk
that starts after rolling back every draft token.

Slots are never cleared. Taps before position zero read as zero, so a fresh
request never sees the previous tenant's inputs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Final

from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import BufferType, BufferValue, DeviceRef, Value
from max.nn.kv_cache import RecurrentStateRegion
from typing_extensions import Self

if TYPE_CHECKING:
    # `model_config` imports this module for the regions a cache declares,
    # and every use here is an annotation.
    from .model_config import InklingTextConfig

# The conv kernel accumulates in float32 regardless of the model dtype.
CONV_STATE_DTYPE: Final = DType.float32


class ConvSite(IntEnum):
    """The four convolution sites of a layer, in pool order."""

    K = 0
    V = 1
    ATTN_OUT = 2
    MLP_OUT = 3


def _leaf_id(site: ConvSite, is_local: bool, prefix: str = "") -> str:
    """Names a pool leaf by the site it serves and the layers it spans."""
    kind = "local" if is_local else "global"
    return f"{prefix}conv/{kind}/{site.name.lower()}"


def conv_state_regions(
    text_config: InklingTextConfig,
    *,
    tp_size: int = 1,
    num_draft_tokens: int = 0,
) -> tuple[RecurrentStateRegion, ...]:
    """Returns the conv leaves a request occupies, per device.

    Backbone only; the MTP draft uses scratch pools, see
    :class:`InklingConvScratchPools`.
    """
    return InklingConvStateLayout.from_config(
        text_config, tp_size=tp_size, num_draft_tokens=num_draft_tokens
    ).regions()


@dataclass(frozen=True)
class InklingConvStateLayout:
    """Per-device channel widths of each layer's four sites, in
    :class:`ConvSite` order."""

    ring_len: int
    """Entries per slot: ``kernel_size - 1`` plus the speculative width."""

    layers: tuple[tuple[int, int, int, int], ...]
    is_local: tuple[bool, ...] = ()
    """Which layers are sliding-window, in :attr:`layers` order.

    A layer's K and V widths follow this, so it is also what splits the pool
    into uniformly-shaped leaves.
    """

    leaf_prefix: str = ""
    """Namespaces this layout's leaves apart from another's.

    The MTP draft keeps its own conv state beside the backbone's, in the same
    row, so the two sets of leaves share a cache child and must not share
    names.
    """

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def kinds(self) -> tuple[bool, ...]:
        """The layer kinds present, widest group first for stable leaf order."""
        return tuple(
            kind for kind in (False, True) if kind in set(self.is_local)
        )

    def layers_of_kind(self, is_local: bool) -> tuple[int, ...]:
        """Layer indices of one kind, in order."""
        return tuple(
            idx for idx, local in enumerate(self.is_local) if local == is_local
        )

    def regions(self) -> tuple[RecurrentStateRegion, ...]:
        """The pool leaves this layout occupies, per device.

        One leaf per (site, layer kind), because a leaf is uniformly shaped
        and only the K and V widths vary, and only between the two kinds. The
        two residual sites split by kind as well, so a layer's row index
        within a leaf is its ordinal among its own kind at every site.
        """
        return tuple(
            RecurrentStateRegion(
                leaf_id=_leaf_id(site, is_local, self.leaf_prefix),
                num_layers=len(self.layers_of_kind(is_local)),
                row_shape=(self.ring_len, self.layers[first][site]),
                dtype=CONV_STATE_DTYPE,
            )
            for is_local in self.kinds()
            for site in ConvSite
            for first in (self.layers_of_kind(is_local)[0],)
        )

    def row_for(self, layer_idx: int, site: ConvSite) -> tuple[int, int]:
        """Returns the leaf a layer's site lives in, and its row within it.

        The leaf is a position among :meth:`regions`, which is how a device's
        ``leaves`` tuple is addressed.
        """
        is_local = self.is_local[layer_idx]
        ordinal = self.layers_of_kind(is_local).index(layer_idx)
        leaf = self.kinds().index(is_local) * len(ConvSite) + int(site)
        return leaf, ordinal

    def take_pools(
        self, inputs: Iterator[Value[Any]], num_devices: int
    ) -> list[list[BufferValue]]:
        """Pulls this layout's pools off a graph-input iterator, one list per rank."""
        per_device = self.num_layers * len(ConvSite)
        return [
            [next(inputs).buffer for _ in range(per_device)]
            for _ in range(num_devices)
        ]

    def buffer_types(self, devices: Sequence[DeviceRef]) -> list[BufferType]:
        """Graph input types of the pools, in device then :attr:`layers` order."""
        return [
            BufferType(
                CONV_STATE_DTYPE,
                shape=["max_conv_slots", self.ring_len, channels],
                device=device,
            )
            for device in devices
            for widths in self.layers
            for channels in widths
        ]

    @classmethod
    def from_config(
        cls,
        text_config: InklingTextConfig,
        *,
        tp_size: int = 1,
        num_draft_tokens: int = 0,
        leaf_prefix: str = "",
    ) -> Self:
        """Derives the layout from the checkpoint config."""
        return cls.from_local_flags(
            text_config,
            [
                text_config.is_local_attention(i)
                for i in range(text_config.num_hidden_layers)
            ],
            tp_size=tp_size,
            num_draft_tokens=num_draft_tokens,
            leaf_prefix=leaf_prefix,
        )

    @classmethod
    def from_local_flags(
        cls,
        text_config: InklingTextConfig,
        is_local: Sequence[bool],
        *,
        tp_size: int = 1,
        num_draft_tokens: int = 0,
        leaf_prefix: str = "",
    ) -> Self:
        """Layout for decoder blocks with an explicit local/global mix."""
        # Not channel-sharded: each rank convolves a full-width partial sum
        # and the ranks are summed afterwards.
        residual_width = text_config.hidden_size
        layers = []
        for local in is_local:
            kv_width = text_config.kv_conv_dim(local) // tp_size
            layers.append((kv_width, kv_width, residual_width, residual_width))
        return cls(
            ring_len=text_config.sconv_kernel_size - 1 + num_draft_tokens,
            layers=tuple(layers),
            is_local=tuple(is_local),
            leaf_prefix=leaf_prefix,
        )


class InklingConvScratchPools:
    """The MTP draft's conv pools: one zero slot that every batch item
    reads. The draft never commits, so its taps before a chunk read as
    zero."""

    def __init__(
        self, layout: InklingConvStateLayout, devices: Sequence[Device]
    ) -> None:
        self._pools: list[list[Buffer]] = [
            [
                Buffer.zeros(
                    [1, layout.ring_len, channels], CONV_STATE_DTYPE, device
                )
                for widths in layout.layers
                for channels in widths
            ]
            for device in devices
        ]

    def pools(self, device_idx: int) -> list[Buffer]:
        """Per-site pools of one rank, in depth then :class:`ConvSite` order."""
        return self._pools[device_idx]
