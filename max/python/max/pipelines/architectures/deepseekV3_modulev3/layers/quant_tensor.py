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

"""Quantized tensor wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeGuard

from max.dtype import DType
from max.experimental.nn import Module
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Sharded,
)
from max.experimental.tensor import Tensor


def _ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


class QTensor(Module[[], None]):
    """Base class for quantized tensor wrappers.

    QTensors are :class:`~max.experimental.nn.Module` subclasses so that
    parameter discovery finds the inner tensors. They are data wrappers, not
    callable layers — pass them to a quantized kernel op
    (e.g. ``quant_ops.matmul``).
    """

    def forward(self) -> None:
        raise NotImplementedError("QTensor is not a callable layer")


class FP8BlockTensor(QTensor):
    """FP8 block-scaled quantized tensor.
    Args:
        data: Packed ``float8_e4m3fn`` data of shape ``[rows, cols]``.
        scale_inv: Per-block inverse ``float32`` scales of shape
            ``[ceil(rows / block_m), ceil(cols / block_k)]``.
        block_size: Per-block size as ``(block_m, block_k)``.
    """

    data: Tensor
    scale_inv: Tensor

    def __init__(
        self,
        *,
        data: Tensor,
        scale_inv: Tensor,
        block_size: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.data = data
        self.scale_inv = scale_inv
        self._block_size = block_size

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        *,
        block_size: tuple[int, int] = (128, 128),
    ) -> FP8BlockTensor:
        """Builds a :class:`FP8BlockTensor` of the given shape, with all zeros."""
        rows, cols = int(shape[0]), int(shape[1])
        block_m, block_k = block_size
        return cls(
            data=Tensor.zeros((rows, cols), dtype=DType.float8_e4m3fn),
            scale_inv=Tensor.zeros(
                (_ceildiv(rows, block_m), _ceildiv(cols, block_k)),
                dtype=DType.float32,
            ),
            block_size=block_size,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        return self._block_size

    def shard(self, axis: int, mesh: DeviceMesh) -> FP8BlockTensor:
        """Co-shard both leaves along ``axis`` onto ``mesh``.

        Args:
            axis: Tensor axis to shard along (same index into both
                ``data.shape`` and ``scale_inv.shape``).
            mesh: 1-D :class:`DeviceMesh` to scatter onto.

        Returns:
            A new :class:`FP8BlockTensor` whose ``data`` and ``scale_inv``
            leaves are each distributed ``Sharded(axis=axis)`` across
            ``mesh``.
        """
        mapping = PlacementMapping(mesh, (Sharded(axis=axis),))
        return FP8BlockTensor(
            data=self.data.to(mapping),
            scale_inv=self.scale_inv.to(mapping),
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[FP8BlockTensor, ...]:
        return tuple(
            FP8BlockTensor(
                data=self.data.local_shards[i],
                scale_inv=self.scale_inv.local_shards[i],
                block_size=self.block_size,
            )
            for i in range(self.data.num_shards)
        )

    @property
    def mesh(self) -> DeviceMesh:
        return self.data.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.data.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.data._mapping = mapping
        self.scale_inv._mapping = mapping


def all_fp8_block(
    weights: Sequence[Tensor | FP8BlockTensor],
) -> TypeGuard[Sequence[FP8BlockTensor]]:
    """Narrow a sequence of mixed weights to a homogeneous FP8 sequence."""
    return all(isinstance(w, FP8BlockTensor) for w in weights)
