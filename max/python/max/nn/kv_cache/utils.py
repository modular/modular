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

import numpy as np
from max._kv_cache_ops import (
    mha_decode_num_partitions,
    mla_dispatch_args_scalar,
)
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef, TensorType


@dataclass(frozen=True)
class AttnKeyInterface:
    """Common base for resolved attention keys."""

    def pack_into_buffer(
        self, device: Device, max_cache_valid_length: int
    ) -> Buffer:
        """Packs this into a kernel dispatch-metadata buffer.

        ``max_cache_valid_length`` is the runtime cache length; it is supplied
        here rather than stored so the identity is independent of it.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class AttnKey(AttnKeyInterface):
    """A resolved decode-attention dispatch shape.

    The resolved ``num_partitions`` (the kernel grid) plus the batch and prompt
    dimensions. The runtime ``max_cache_valid_length`` is supplied to
    :meth:`pack_into_buffer` rather than stored, so dispatches that differ only
    in cache length share one identity. Concrete subclasses
    (:class:`MHAAttnKey`, :class:`MLAAttnKey`)
    implement the kernel-specific buffer layout.
    """

    batch_size: int
    max_prompt_length: int
    num_partitions: int


@dataclass(frozen=True)
class MHAAttnKey(AttnKey):
    """Decode dispatch metadata for multi-head attention (MHA)."""

    def pack_into_buffer(
        self, device: Device, max_cache_valid_length: int
    ) -> Buffer:
        # MHA decode kernels read a 4-int dispatch buffer on the host (CPU).
        # ``device`` is intentionally ignored: the MHA dispatch-metadata graph
        # input is declared CPU-resident.
        del device
        return Buffer.from_numpy(
            np.array(
                [
                    self.batch_size,
                    self.max_prompt_length,
                    self.num_partitions,
                    max_cache_valid_length,
                ],
                dtype=np.int64,
            )
        )


@dataclass(frozen=True)
class MLAAttnKey(AttnKey):
    """Decode dispatch metadata for multi-latent attention (MLA)."""

    def pack_into_buffer(
        self, device: Device, max_cache_valid_length: int
    ) -> Buffer:
        # MLA decode kernels read a 3-int dispatch buffer on the accelerator.
        # ``max_cache_valid_length`` is not part of the MLA dispatch buffer (it
        # is carried separately in ``max_lengths``), so it is ignored here.
        del max_cache_valid_length
        metadata = Buffer.from_numpy(
            np.array(
                [
                    self.batch_size,
                    self.max_prompt_length,
                    self.num_partitions,
                ],
                dtype=np.int64,
            )
        )
        return metadata.to(device)


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


class AttentionDispatchResolverInterface:
    """Interface for attention dispatch metadata resolvers."""

    def __init__(
        self,
        devices: Sequence[DeviceRef],
        is_mla: bool,
        n_kv_heads_per_device: int,
        num_q_heads_per_device: int | None = None,
        is_fp8_kv: bool = False,
    ) -> None:
        raise NotImplementedError

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Returns the resolved decode dispatch key for the given shape."""
        raise NotImplementedError

    def get_symbolic_metadata_input(self, device: DeviceRef) -> TensorType:
        """Returns the symbolic input for this attention key."""
        raise NotImplementedError

    def probe_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns cache lengths to probe for distinct num_partitions."""
        raise NotImplementedError


class AttentionDispatchResolver(AttentionDispatchResolverInterface):
    """Resolves packed attention decode metadata via kernel custom ops.

    Supports both MHA (``mo.mha.decode.get_num_partitions``) and MLA
    (``mo.mla.compute_dispatch_args.scalar``) decode kernels, selected from the
    ``is_mla`` flag.
    """

    def __init__(
        self,
        devices: Sequence[DeviceRef],
        is_mla: bool,
        n_kv_heads_per_device: int,
        num_q_heads_per_device: int | None = None,
        is_fp8_kv: bool = False,
    ) -> None:
        if not devices:
            raise ValueError("devices must not be empty")
        self.device_ref = devices[0]
        self._is_mla = is_mla
        self._n_kv_heads_per_device = n_kv_heads_per_device
        self._num_q_heads = num_q_heads_per_device
        self._is_fp8_kv = is_fp8_kv
        self._key_cls: type[AttnKey] = MLAAttnKey if is_mla else MHAAttnKey

        if self._is_mla:
            assert num_q_heads_per_device is not None

        # Built lazily so :meth:`get_symbolic_metadata_input()` does not require
        # a device context for a GPU DeviceRef on a CPU-only host.
        self._device: None | Device = None

    @property
    def device(self) -> None | Device:
        # The decode dispatch kernels are GPU custom ops needing a concrete
        # ``Device``. A CPU-only resolver has no device; ``resolve_attn_key``
        # returns the sentinel key (num_partitions=1) without invoking them.
        if self._device is None and not self.device_ref.is_cpu():
            self._device = self.device_ref.to_device()
        return self._device

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Returns the resolved decode dispatch key for the given shape.

        Empty / degenerate replicas (``batch_size <= 0`` or a CPU-only
        resolver) return a sentinel key (``num_partitions=1``) without invoking
        the dispatch kernels.
        """
        if batch_size <= 0 or self.device is None:
            # Sentinel for empty / degenerate replicas; skip the kernels.
            num_partitions = 1
        elif self._is_mla:
            assert self._num_q_heads is not None
            # mla_dispatch_args_scalar returns (batch_size, max_prompt_length,
            # num_partitions), possibly adjusted from the inputs.
            batch_size, max_prompt_length, num_partitions = (
                mla_dispatch_args_scalar(
                    batch_size,
                    max_cache_valid_length,
                    max_prompt_length,
                    self._num_q_heads,
                    self._is_fp8_kv,
                    self.device,
                )
            )
        else:
            num_partitions = mha_decode_num_partitions(
                batch_size,
                max_cache_valid_length,
                self._n_kv_heads_per_device,
                self.device,
            )

        return self._key_cls(
            batch_size=int(batch_size),
            max_prompt_length=int(max_prompt_length),
            num_partitions=int(num_partitions),
        )

    def get_symbolic_metadata_input(self, device: DeviceRef) -> TensorType:
        """Returns the symbolic input for this attention key."""
        if self._is_mla:
            return TensorType(
                DType.int64,
                shape=[3],
                device=device,
            )
        else:
            return TensorType(
                DType.int64,
                shape=[4],
                device=DeviceRef.CPU(),
            )

    def probe_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns cache lengths to probe for distinct num_partitions.

        These are the cache lengths warmed up during graph capture. MHA probes
        at 256-token granularity; MLA probes at a finer 64-token granularity
        (and, under speculative decoding, adds extra probes to hit more
        ``(num_partitions, draft_num_partitions)`` pairs). The selected
        granularity follows ``is_mla``.
        """
        granularity = 64 if self._is_mla else 256
        probe_lengths = (
            [1]
            + list(range(granularity, max_cache_length, granularity))
            + [max_cache_length]
        )
        if self._is_mla and q_max_seq_len > 1:
            # With spec decoding, probe a few more entries to hit all viable
            # (num_partition, draft_num_partition) pairs. Determined
            # experimentally; brute-forcing all pairs would inflate capture
            # time significantly.
            probe_lengths.extend(
                range(granularity - 1, max_cache_length, granularity)
            )
        return probe_lengths


def build_max_lengths_tensor(
    max_seq_length: int, max_cache_length: int
) -> Buffer:
    """Builds a ``[1, 2]`` uint32 buffer of maximum lengths for a single decode step.

    Args:
        max_seq_length: The maximum sequence length.
        max_cache_length: The maximum cache length.

    Returns:
        A :class:`~max.driver.Buffer` of shape ``[1, 2]`` and dtype
        ``uint32`` containing ``(max_seq_length, max_cache_length)``.
    """
    max_lengths_np = np.empty((1, 2), np.uint32)
    max_lengths_np[0, 0] = max_seq_length
    max_lengths_np[0, 1] = max_cache_length
    return Buffer.from_numpy(max_lengths_np)
