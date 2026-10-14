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
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef, TensorType

ATTN_DISPATCH_METADATA = "attn_dispatch_metadata"
"""Graph input holding the resolved decode-attention dispatch shape."""

DRAFT_ATTN_DISPATCH_METADATA = "draft_attn_dispatch_metadata"
"""The same, for the draft key under speculative decoding."""


@runtime_checkable
class GraphInputStagingInterface(Protocol):
    """One forward's staging: somewhere to write a graph input.

    Structural on purpose: what builds the dispatch metadata lives here, one
    layer below the staging that carries it, so this names the one method it
    needs rather than importing the implementation.
    """

    def get(
        self, name: str, shape: tuple[int, ...]
    ) -> tuple[Buffer, tuple[Buffer, ...]]:
        """Returns host staging for ``name`` and the buffers it is sent to."""
        ...


@dataclass(frozen=True)
class DispatchMetadataSpec:
    """The shape of a kernel's dispatch-metadata input, and where it lives.

    One source for both halves of the input: the ``TensorType`` the graph
    declares, and the buffer a forward fills. They used to be written out
    separately and kept in step by comment.
    """

    shape: tuple[int, ...]
    dtype: DType
    on_device: bool
    """Whether the kernel reads it on the accelerator rather than the host.

    A host-resident one is never transferred, so there is nothing to stage
    for it; a device-resident one is staged like any other graph input.
    """

    def tensor_type(self, device: DeviceRef) -> TensorType:
        """The graph input this metadata is declared as."""
        return TensorType(
            self.dtype,
            shape=list(self.shape),
            device=device if self.on_device else DeviceRef.CPU(),
        )


@dataclass(frozen=True)
class AttnKeyInterface:
    """Common base for resolved attention keys."""

    @classmethod
    def dispatch_metadata_spec(cls) -> DispatchMetadataSpec:
        """Returns the shape, dtype and residence of this kernel's metadata."""
        raise NotImplementedError

    def pack_into(self, into: np.ndarray, max_cache_valid_length: int) -> None:
        """Writes this dispatch shape's values into ``into``.

        ``into`` is shaped by :meth:`dispatch_metadata_spec`. Writing rather
        than allocating is what lets a caller hand over staging it already
        owns -- see ``GraphInputStager``.

        ``max_cache_valid_length`` is the runtime cache length; it is supplied
        here rather than stored so the identity is independent of it.
        """
        raise NotImplementedError

    def pack_into_buffer(
        self, device: Device, max_cache_valid_length: int
    ) -> Buffer:
        """Packs this into a freshly allocated dispatch-metadata buffer.

        For callers with nowhere to stage it. A host-resident spec stays on
        the host; a device-resident one is copied over, which is the copy
        staging exists to fold into the rest of a forward's.
        """
        spec = self.dispatch_metadata_spec()
        values = np.zeros(spec.shape, dtype=spec.dtype.to_numpy())
        self.pack_into(values, max_cache_valid_length)
        host = Buffer.from_numpy(values)
        return host.to(device) if spec.on_device else host

    def dispatch_metadata_buffers(
        self,
        staging: GraphInputStagingInterface | None,
        *,
        name: str,
        devices: Sequence[Device],
        max_cache_valid_length: int,
    ) -> tuple[Buffer, ...]:
        """One buffer per tensor-parallel shard, staged when it can be.

        A device-resident key with somewhere to stage is one host write fanned
        out to every shard, so it travels in the forward's own transfer.
        Otherwise each shard gets a buffer of its own -- for a host-resident
        key there is no transfer to fold in anyway.
        """
        spec = self.dispatch_metadata_spec()
        if staging is None or not spec.on_device:
            return tuple(
                self.pack_into_buffer(device, max_cache_valid_length)
                for device in devices
            )
        host, destinations = staging.get(name, spec.shape)
        self.pack_into(host.to_numpy(), max_cache_valid_length)
        return destinations


@dataclass(frozen=True)
class AttnKey(AttnKeyInterface):
    """A resolved decode-attention dispatch shape.

    The resolved ``num_partitions`` (the kernel grid) plus the batch and prompt
    dimensions. The runtime ``max_cache_valid_length`` is supplied to
    :meth:`~AttnKeyInterface.pack_into` rather than stored, so dispatches that
    differ only in cache length share one identity. Concrete subclasses
    (:class:`MHAAttnKey`, :class:`MLAAttnKey`) declare the kernel-specific
    layout and write it.
    """

    batch_size: int
    max_prompt_length: int
    num_partitions: int


@dataclass(frozen=True)
class MHAAttnKey(AttnKey):
    """Decode dispatch metadata for multi-head attention (MHA)."""

    @classmethod
    def dispatch_metadata_spec(cls) -> DispatchMetadataSpec:
        """MHA decode kernels read four ints, on the host."""
        return DispatchMetadataSpec(
            shape=(4,), dtype=DType.int64, on_device=False
        )

    def pack_into(self, into: np.ndarray, max_cache_valid_length: int) -> None:
        """Writes batch size, prompt width, partition count, cache length."""
        into[:] = (
            self.batch_size,
            self.max_prompt_length,
            self.num_partitions,
            max_cache_valid_length,
        )


@dataclass(frozen=True)
class MLAAttnKey(AttnKey):
    """Decode dispatch metadata for multi-latent attention (MLA)."""

    @classmethod
    def dispatch_metadata_spec(cls) -> DispatchMetadataSpec:
        """MLA decode kernels read three ints, on the accelerator."""
        return DispatchMetadataSpec(
            shape=(3,), dtype=DType.int64, on_device=True
        )

    def pack_into(self, into: np.ndarray, max_cache_valid_length: int) -> None:
        """Writes batch size, prompt width and partition count.

        The cache length is not part of the MLA dispatch buffer; it is carried
        separately in ``max_cache_length``.
        """
        into[:] = (
            self.batch_size,
            self.max_prompt_length,
            self.num_partitions,
        )


@dataclass(frozen=True)
class MSAAttnKey(AttnKeyInterface):
    """Decode dispatch metadata for multi-step attention (MSA)."""

    @classmethod
    def dispatch_metadata_spec(cls) -> DispatchMetadataSpec:
        """A single host int, since MSA kernels read no dispatch metadata."""
        return DispatchMetadataSpec(
            shape=(1,), dtype=DType.int64, on_device=False
        )

    def pack_into(self, into: np.ndarray, max_cache_valid_length: int) -> None:
        """Writes the sentinel the placeholder input carries."""
        into[:] = 42


@dataclass(frozen=True)
class MultiAttnKey(AttnKeyInterface):
    """A tree of resolved dispatch metadata mirroring a ``MultiKVCacheParams``
    tree.

    ``children`` is a tuple of ``(name, key)`` pairs (rather than a dict) so it
    stays a frozen, hashable identity for the graph-capture key map.
    """

    children: tuple[tuple[str, AttnKeyInterface], ...]

    @classmethod
    def from_dict(cls, children: dict[str, AttnKeyInterface]) -> MultiAttnKey:
        """Builds a :class:`MultiAttnKey` from a name -> key mapping."""
        return cls(children=tuple(children.items()))


#: Padding added to every LUT inner dim (columns per batch row). The SIMD
#: ``populate`` in ``PagedKVCache`` reads up to 16 consecutive ``uint32``
#: entries past ``base_kv_row / page_size``; this buffer keeps those reads
#: in-bounds of the allocation for partial-tile tails. The value is also
#: a multiple of 8 so the inner-dim stride stays 32-byte aligned for the
#: ``ld.global.v{N}.u32`` vector loads.
_LUT_TAIL_PAD = 16


def padded_lut_cols(cols: int) -> int:
    """Rounds a page lookup-table inner dim up to a kernel-safe width.

    Kept in lockstep with the invariant asserted in
    ``max/kernels/src/kv_cache/types.mojo`` (``PagedKVCache.populate``):
    ``lookup_table.dim[1]`` is a multiple of 8 and is at least
    ``logical_cols + 15`` so a 16-wide SIMD lookup load from any valid
    ``first_lut_idx`` stays in-bounds.

    Args:
        cols: The number of logical page columns per batch row.

    Returns:
        The allocated inner dim to use for the lookup table.
    """
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def build_max_lengths_tensors(
    max_prompt_length: int, max_cache_length: int
) -> tuple[Buffer, Buffer]:
    """Builds two ``[1]`` uint32 scalar buffers of maximum lengths.

    Args:
        max_prompt_length: The maximum prompt (query) length.
        max_cache_length: The maximum cache length.

    Returns:
        A tuple ``(max_prompt_length, max_cache_length)`` of
        :class:`~max.driver.Buffer`, each of shape ``[1]`` and dtype
        ``uint32``.
    """
    max_prompt_length_np = np.array([max_prompt_length], np.uint32)
    max_cache_length_np = np.array([max_cache_length], np.uint32)
    return (
        Buffer.from_numpy(max_prompt_length_np),
        Buffer.from_numpy(max_cache_length_np),
    )
