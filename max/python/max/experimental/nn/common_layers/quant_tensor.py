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
from functools import cached_property
from typing import Any, TypeAlias, TypeGuard

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.module import module_dataclass
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import Shape, TensorValue
from max.nn.kernels import quantize_dynamic_block_scaled
from typing_extensions import Self

MAX_E2M1 = 6.0
MAX_E4M3 = 448.0


def _ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


class QTensor(Module[[], None]):
    """Base class for quantized tensor wrappers.

    QTensors are :class:`~max.experimental.nn.Module` subclasses so that
    parameter discovery finds the inner tensors. They are data wrappers, not
    callable layers — pass them to a quantized kernel op
    (such as ``quant_ops.matmul``).
    """

    def forward(self) -> None:
        """Raises, since a QTensor is a data wrapper rather than a layer."""
        raise NotImplementedError("QTensor is not a callable layer")

    @property
    def local_shards(self) -> tuple[Self, ...]:
        """Per-device shards of this tensor; subclasses implement it."""
        raise NotImplementedError("QTensor does not support local_shards")

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh the tensor is placed on; subclasses implement it."""
        raise NotImplementedError("mesh is not implemented for QTensor")

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Returns a copy placed on ``target``; subclasses implement it."""
        raise NotImplementedError("to() is not implemented for QTensor")


class FP8BlockTensor(QTensor):
    """FP8 block-scaled quantized tensor.

    Args:
        data: Packed ``float8_e4m3fn`` data of shape ``[rows, cols]``.
        weight_scale_inv: Per-block inverse ``float32`` scales of shape
            ``[ceil(rows / block_m), ceil(cols / block_k)]``.
        block_size: Per-block size as ``(block_m, block_k)``.
    """

    data: Tensor
    weight_scale_inv: Tensor

    def __init__(
        self,
        *,
        data: Tensor,
        weight_scale_inv: Tensor,
        block_size: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.data = data
        self.weight_scale_inv = weight_scale_inv
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
            weight_scale_inv=Tensor.zeros(
                (_ceildiv(rows, block_m), _ceildiv(cols, block_k)),
                dtype=DType.float32,
            ),
            block_size=block_size,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        """Quantization block shape as ``(rows, cols)``."""
        return self._block_size

    def shard(self, axis: int, mesh: DeviceMesh) -> FP8BlockTensor:
        """Co-shard both leaves along ``axis`` onto ``mesh``.

        Args:
            axis: Tensor axis to shard along (same index into both
                ``data.shape`` and ``weight_scale_inv.shape``).
            mesh: 1-D :class:`DeviceMesh` to scatter onto.

        Returns:
            A new :class:`FP8BlockTensor` whose ``data`` and ``weight_scale_inv``
            leaves are each distributed ``Sharded(axis=axis)`` across
            ``mesh``.
        """
        mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
        return FP8BlockTensor(
            data=self.data.to(mapping),
            weight_scale_inv=self.weight_scale_inv.to(mapping),
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[FP8BlockTensor, ...]:
        """Per-device shards, pairing each ``data`` shard with its scales."""
        return tuple(
            FP8BlockTensor(
                data=self.data.local_shards[i],
                weight_scale_inv=self.weight_scale_inv.local_shards[i],
                block_size=self.block_size,
            )
            for i in range(self.data.num_shards)
        )

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh of the packed ``data`` tensor."""
        return self.data.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.data.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.data._mapping = mapping
        self.weight_scale_inv._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> FP8BlockTensor:
        """Returns a copy with ``data`` and its scales placed on ``target``."""
        return FP8BlockTensor(
            data=self.data.to(target),
            weight_scale_inv=self.weight_scale_inv.to(target),
            block_size=self.block_size,
        )


class NVFP4Tensor(QTensor):
    """NVFP4 (modelopt) block-scaled quantized tensor.

    NVFP4 packs two ``float4_e2m1fn`` values per ``uint8`` and applies two
    scale levels: a per-16-element block scale (``float8_e4m3fn``) and a
    per-tensor global scale (``weight_scale_2``, ``float32``). The activation
    scale (``input_scale``, ``float32``) is *static* — loaded from the
    checkpoint — and used to normalize activations before dynamic block
    quantization (see :func:`~.quant_ops._matmul_nvfp4`).

    Args:
        data: Packed ``uint8`` weight data of shape ``[rows, cols // 2]``.
        weight_scale: Per-block ``float8_e4m3fn`` scales of shape
            ``[ceil(rows / block_m), ceil(cols / block_k)]``.
        weight_scale_2: Per-tensor ``float32`` global scale, scalar ``()``
            (or ``[num_experts]`` when stacked for grouped MoE).
        input_scale: Static per-tensor ``float32`` activation scale, scalar
            ``()`` (or ``[num_experts]`` when stacked).
        block_size: Per-block size as ``(block_m, block_k)``; ``(1, 16)`` for
            NVFP4.
    """

    data: Tensor
    weight_scale: Tensor
    weight_scale_2: Tensor
    input_scale: Tensor

    def __init__(
        self,
        *,
        data: Tensor,
        weight_scale: Tensor,
        weight_scale_2: Tensor,
        input_scale: Tensor,
        block_size: tuple[int, int] = (1, 16),
    ) -> None:
        super().__init__()
        self.data = data
        self.weight_scale = weight_scale
        self.weight_scale_2 = weight_scale_2
        self.input_scale = input_scale
        self._block_size = block_size

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        *,
        block_size: tuple[int, int] = (1, 16),
    ) -> NVFP4Tensor:
        """Builds a zero-filled :class:`NVFP4Tensor` of logical ``shape``.

        ``shape`` is the *unpacked* Linear-convention ``[out_dim, in_dim]``.
        The packed ``data`` leaf has half the trailing dimension.
        """
        rows, cols = int(shape[0]), int(shape[1])
        block_m, block_k = block_size
        return cls(
            data=Tensor.zeros((rows, cols // 2), dtype=DType.uint8),
            weight_scale=Tensor.zeros(
                (_ceildiv(rows, block_m), _ceildiv(cols, block_k)),
                dtype=DType.float8_e4m3fn,
            ),
            weight_scale_2=Tensor.zeros((), dtype=DType.float32, device=CPU()),
            input_scale=Tensor.zeros((), dtype=DType.float32, device=CPU()),
            block_size=block_size,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        """Quantization block shape as ``(rows, cols)``."""
        return self._block_size

    def shard(self, axis: int, mesh: DeviceMesh) -> NVFP4Tensor:
        """Co-shard ``data``/``weight_scale`` along ``axis`` onto ``mesh``."""
        mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
        return NVFP4Tensor(
            data=self.data.to(mapping),
            weight_scale=self.weight_scale.to(mapping),
            weight_scale_2=self.weight_scale_2,
            input_scale=self.input_scale,
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[NVFP4Tensor, ...]:
        """Per-device shards, pairing each ``data`` shard with its scales."""
        n = self.data.num_shards
        return tuple(
            NVFP4Tensor(
                data=self.data.local_shards[i],
                weight_scale=self.weight_scale.local_shards[i],
                weight_scale_2=self.weight_scale_2,
                input_scale=self.input_scale,
                block_size=self.block_size,
            )
            for i in range(n)
        )

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh of the packed ``data`` tensor."""
        return self.data.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.data.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.data._mapping = mapping
        self.weight_scale._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> NVFP4Tensor:
        """Returns a copy with ``data`` and ``weight_scale`` placed on ``target``."""
        return NVFP4Tensor(
            data=self.data.to(target),
            weight_scale=self.weight_scale.to(target),
            # weight_scale_2 and input_scale stay on CPU
            weight_scale_2=self.weight_scale_2,
            input_scale=self.input_scale,
            block_size=self.block_size,
        )


class NVFP4TensorQuantizer(NVFP4Tensor):
    """FakeQuant-style on-the-fly quantizable weight tensor.

    By contrast to NVFP4Tensor, which serves as a container
    for loading pre-quantized weights and scales from the checkpoint, LazyNVFP4Tensor wraps the native precision tensor,
    while exposing scales tensors via properties that, on first use, trigger dynamic quantization via a kernel.
    Subclassing from NVFP4Tensor allows for quant ops dispatch, like a normal NVFP4 weight tensor.
    """

    _tensor: Tensor

    def __init__(
        self,
        tensor: Tensor,
        block_size: tuple[int, int] = (1, 16),
    ):
        if int(tensor.shape[-1]) % block_size[-1]:
            raise ValueError(
                f"NVFP4 needs a K that is a multiple of {block_size[-1]}, got "
                f"{tensor.shape[-1]}"
            )
        self._tensor = tensor
        self._block_size = block_size

    @property
    def shape(self) -> Shape:
        """Delegates shape to underlying tensor."""
        return self._tensor.shape

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        # "" adds no segment, so the leaf path stays "...gate_proj.weight".
        return {"": self._tensor, "_block_size": self._block_size}, None

    def __tree_setattr__(self, key: Any, value: Any) -> None:
        if key == "":
            key = "_tensor"
        super().__tree_setattr__(key, value)

    def to(
        self, target: Device | DeviceMesh | DeviceMapping
    ) -> NVFP4TensorQuantizer:
        """Delegates placement to underlying tensor."""
        self._tensor = self._tensor.to(target)
        return self

    @property
    def block_size(self) -> tuple[int, int]:
        """Quantization block shape as ``(rows, cols)``."""
        return self._block_size

    def _quantize(self) -> NVFP4Tensor:
        amax = (
            F.abs(self._tensor)
            .max(axis=None)
            .reshape([])
            .cast(DType.float32)
            .to(CPU())
        )
        encode_scale = (MAX_E2M1 * MAX_E4M3) / amax
        decode_scale = 1.0 / encode_scale

        packed_value, scales_value = quantize_dynamic_block_scaled(
            TensorValue(self._tensor),
            tensor_sf=TensorValue(encode_scale),
            sf_vector_size=self.block_size[1],
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,
        )
        packed = Tensor.from_graph_value(packed_value)
        scales = Tensor.from_graph_value(scales_value)
        rows = self._tensor.shape[0]
        scale_cols = self._tensor.shape[1] // self.block_size[1]
        weight_scale = scales.permute([0, 3, 2, 1, 4]).reshape(
            [rows, scale_cols]
        )
        input_scale = Tensor.ones((), dtype=DType.float32, device=CPU())

        return NVFP4Tensor(
            data=packed,
            weight_scale=weight_scale,
            weight_scale_2=decode_scale,
            input_scale=input_scale,
        )

    @cached_property
    def quant_tensor(self) -> NVFP4Tensor:
        """Holds a cached quantized tensor."""
        return self._quantize()

    # The base class stores these as attributes. These stay read-only views
    # of the cached quantized tensor so dispatch can read them the same way.
    @property
    def data(self) -> Tensor:  # type: ignore[override]
        """Packed NVFP4 weights."""
        return self.quant_tensor.data

    @property
    def weight_scale(self) -> Tensor:  # type: ignore[override]
        """FP8 block scales."""
        return self.quant_tensor.weight_scale

    @property
    def weight_scale_2(self) -> Tensor:  # type: ignore[override]
        """Global weight scaling factor."""
        return self.quant_tensor.weight_scale_2

    @property
    def input_scale(self) -> Tensor:  # type: ignore[override]
        """Global activations scaling factor."""
        return self.quant_tensor.input_scale


@module_dataclass(frozen=True)
class NVFP4Activation(QTensor):
    """NVFP4-quantized activation tensor.

    Args:
        data: Packed ``uint8`` FP4 tokens of shape ``[total_tokens, K / 2]``.
        scales: Block scales in the padded SF-atom layout.
        input_scale: The ``float32`` scale these tokens were quantized with.
        scales_offset: Per-expert offset into the padded ``scales``.
    """

    data: Tensor
    scales: Tensor
    input_scale: Tensor
    scales_offset: Tensor


QuantAwareTensor: TypeAlias = (
    Tensor
    | FP8BlockTensor
    | NVFP4Tensor
    | NVFP4TensorQuantizer
    | NVFP4Activation
)


def all_fp8_block(
    weights: Sequence[QuantAwareTensor],
) -> TypeGuard[Sequence[FP8BlockTensor]]:
    """Narrow a sequence of mixed weights to a homogeneous FP8 sequence."""
    return all(isinstance(w, FP8BlockTensor) for w in weights)


def all_nvfp4(
    weights: Sequence[QuantAwareTensor],
) -> TypeGuard[Sequence[NVFP4Tensor]]:
    """Narrow a sequence of mixed weights to a homogeneous NVFP4 sequence."""
    return all(isinstance(w, NVFP4Tensor) for w in weights)
