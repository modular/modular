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
from typing import Any, ClassVar, TypeAlias, TypeGuard

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.quant_kernels import (
    quantize_dynamic_block_scaled,
    quantize_dynamic_block_scaled_mxfp4,
)
from max.experimental.nn.module import module_dataclass
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import Shape
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


# --------------------------------------------------------------------------- #
# Weight containers
#
# Each holds a checkpoint's pre-quantized weight, with every companion tensor
# it needs listed as its own field, so the layout of a format reads off its
# class.
# --------------------------------------------------------------------------- #


class _QuantizedWeight(QTensor):
    """A quantized weight whose fields carry the checkpoint's names.

    A checkpoint stores a quantized projection's packed data as its
    ``weight``, beside the scales, and the fields follow suit. Held as
    ``proj``, the weight loads ``proj.weight`` and ``proj.weight_scale``;
    held by a :class:`~.quant_linear.QuantizedLinear`, which adds no path
    segment for it, it loads the same names.
    """

    weight: Tensor


class FP8BlockTensor(_QuantizedWeight):
    """FP8 block-scaled quantized tensor.

    Args:
        weight: Packed ``float8_e4m3fn`` data of shape ``[rows, cols]``.
        weight_scale_inv: Per-block inverse ``float32`` scales of shape
            ``[ceil(rows / block_m), ceil(cols / block_k)]``.
        block_size: Per-block size as ``(block_m, block_k)``.
    """

    weight_scale_inv: Tensor

    def __init__(
        self,
        *,
        weight: Tensor,
        weight_scale_inv: Tensor,
        block_size: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.weight = weight
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
            weight=Tensor.zeros((rows, cols), dtype=DType.float8_e4m3fn),
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
                ``weight.shape`` and ``weight_scale_inv.shape``).
            mesh: 1-D :class:`DeviceMesh` to scatter onto.

        Returns:
            A new :class:`FP8BlockTensor` whose ``weight`` and
            ``weight_scale_inv`` leaves are each distributed
            ``Sharded(axis=axis)`` across ``mesh``.
        """
        mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
        return FP8BlockTensor(
            weight=self.weight.to(mapping),
            weight_scale_inv=self.weight_scale_inv.to(mapping),
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[FP8BlockTensor, ...]:
        """Per-device shards, pairing each ``weight`` shard with its scales."""
        return tuple(
            FP8BlockTensor(
                weight=self.weight.local_shards[i],
                weight_scale_inv=self.weight_scale_inv.local_shards[i],
                block_size=self.block_size,
            )
            for i in range(self.weight.num_shards)
        )

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh of the packed ``weight`` tensor."""
        return self.weight.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.weight.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.weight._mapping = mapping
        self.weight_scale_inv._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> FP8BlockTensor:
        """Returns a copy with ``weight`` and its scales placed on ``target``."""
        return FP8BlockTensor(
            weight=self.weight.to(target),
            weight_scale_inv=self.weight_scale_inv.to(target),
            block_size=self.block_size,
        )


class NVFP4Tensor(_QuantizedWeight):
    """NVFP4 (modelopt) block-scaled quantized tensor.

    NVFP4 packs two ``float4_e2m1fn`` values per ``uint8`` and applies two
    scale levels: a per-16-element block scale (``float8_e4m3fn``) and a
    per-tensor global scale (``weight_scale_2``, ``float32``). The activation
    scale (``input_scale``, ``float32``) is *static* — loaded from the
    checkpoint — and used to normalize activations before dynamic block
    quantization (see :func:`~.quant_ops._matmul_nvfp4`).

    Args:
        weight: Packed ``uint8`` weight data of shape ``[rows, cols // 2]``.
        weight_scale: Per-block ``float8_e4m3fn`` scales of shape
            ``[ceil(rows / block_m), ceil(cols / block_k)]``.
        weight_scale_2: Per-tensor ``float32`` global scale, scalar ``()``
            (or ``[num_experts]`` when stacked for grouped MoE).
        input_scale: Static per-tensor ``float32`` activation scale, scalar
            ``()`` (or ``[num_experts]`` when stacked).
        block_size: Per-block size as ``(block_m, block_k)``; ``(1, 16)`` for
            NVFP4.
    """

    weight_scale: Tensor
    weight_scale_2: Tensor
    input_scale: Tensor

    def __init__(
        self,
        *,
        weight: Tensor,
        weight_scale: Tensor,
        weight_scale_2: Tensor,
        input_scale: Tensor,
        block_size: tuple[int, int] = (1, 16),
    ) -> None:
        super().__init__()
        self.weight = weight
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
        The packed ``weight`` leaf has half the trailing dimension.
        """
        rows, cols = int(shape[0]), int(shape[1])
        block_m, block_k = block_size
        return cls(
            weight=Tensor.zeros((rows, cols // 2), dtype=DType.uint8),
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
        """Co-shard ``weight``/``weight_scale`` along ``axis`` onto ``mesh``."""
        mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
        return NVFP4Tensor(
            weight=self.weight.to(mapping),
            weight_scale=self.weight_scale.to(mapping),
            weight_scale_2=self.weight_scale_2,
            input_scale=self.input_scale,
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[NVFP4Tensor, ...]:
        """Per-device shards, pairing each ``weight`` shard with its scales."""
        n = self.weight.num_shards
        return tuple(
            NVFP4Tensor(
                weight=self.weight.local_shards[i],
                weight_scale=self.weight_scale.local_shards[i],
                weight_scale_2=self.weight_scale_2,
                input_scale=self.input_scale,
                block_size=self.block_size,
            )
            for i in range(n)
        )

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh of the packed ``weight`` tensor."""
        return self.weight.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.weight.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.weight._mapping = mapping
        self.weight_scale._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> NVFP4Tensor:
        """Returns a copy with ``weight`` and ``weight_scale`` on ``target``."""
        return NVFP4Tensor(
            weight=self.weight.to(target),
            weight_scale=self.weight_scale.to(target),
            # weight_scale_2 and input_scale stay on CPU
            weight_scale_2=self.weight_scale_2,
            input_scale=self.input_scale,
            block_size=self.block_size,
        )


class MXFP8Tensor(_QuantizedWeight):
    """MXFP8 (OCP microscaling) block-scaled quantized tensor.

    One unpacked ``float8_e4m3fn`` value per byte, with one power-of-two
    ``float8_e8m0fnu`` scale per 32-element block along K. MX is pure block
    scaling: unlike :class:`NVFP4Tensor` there is no per-tensor second level
    and no static activation scale.

    Args:
        weight: ``float8_e4m3fn`` weight data of shape ``[.., rows, cols]``.
        weight_scale: Per-block ``float8_e8m0fnu`` scales of shape
            ``[.., ceil(rows / block_m), ceil(cols / block_k)]``, in
            checkpoint order; the matmuls interleave them for the kernel
            (see :func:`~.quant_ops.matmul`).
        block_size: Per-block size as ``(block_m, block_k)``; ``(1, 32)`` for
            MXFP8.
    """

    #: Dtype of the stored ``weight``.
    storage_dtype: ClassVar[DType] = DType.float8_e4m3fn
    #: Logical values held per stored ``weight`` element.
    values_per_byte: ClassVar[int] = 1
    #: Dtype of ``weight_scale``.
    scale_dtype: ClassVar[DType] = DType.float8_e8m0fnu

    weight_scale: Tensor

    def __init__(
        self,
        *,
        weight: Tensor,
        weight_scale: Tensor,
        block_size: tuple[int, int] = (1, 32),
    ) -> None:
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self._block_size = block_size

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        *,
        block_size: tuple[int, int] = (1, 32),
    ) -> MXFP8Tensor:
        """Builds a zero-filled :class:`MXFP8Tensor` of shape ``[.., rows, cols]``.

        Leading dimensions (such as stacked experts) are kept on both leaves.
        """
        *lead, rows, cols = (int(d) for d in shape)
        block_m, block_k = block_size
        return cls(
            weight=Tensor.zeros(
                (*lead, rows, cols // cls.values_per_byte),
                dtype=cls.storage_dtype,
            ),
            weight_scale=Tensor.zeros(
                (*lead, _ceildiv(rows, block_m), _ceildiv(cols, block_k)),
                dtype=cls.scale_dtype,
            ),
            block_size=block_size,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        """Quantization block shape as ``(rows, cols)``."""
        return self._block_size

    def shard(self, axis: int, mesh: DeviceMesh) -> MXFP8Tensor:
        """Co-shard ``weight``/``weight_scale`` along ``axis`` onto ``mesh``."""
        mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
        return MXFP8Tensor(
            weight=self.weight.to(mapping),
            weight_scale=self.weight_scale.to(mapping),
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[MXFP8Tensor, ...]:
        """Per-device shards, pairing each ``weight`` shard with its scales."""
        return tuple(
            MXFP8Tensor(
                weight=self.weight.local_shards[i],
                weight_scale=self.weight_scale.local_shards[i],
                block_size=self.block_size,
            )
            for i in range(self.weight.num_shards)
        )

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh of the ``weight`` tensor."""
        return self.weight.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.weight.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.weight._mapping = mapping
        self.weight_scale._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> MXFP8Tensor:
        """Returns a copy with ``weight`` and ``weight_scale`` on ``target``."""
        return MXFP8Tensor(
            weight=self.weight.to(target),
            weight_scale=self.weight_scale.to(target),
            block_size=self.block_size,
        )


class MXFP4Tensor(_QuantizedWeight):
    """MXFP4 (OCP microscaling) block-scaled quantized tensor.

    Two ``e2m1`` codes packed per ``uint8``, with one power-of-two
    ``float8_e8m0fnu`` scale per 32-element block along K. Same two leaves
    as :class:`MXFP8Tensor`; only the stored dtype and packing differ.

    Args:
        weight: Packed ``uint8`` weight data of shape ``[.., rows, cols // 2]``.
        weight_scale: Per-block ``float8_e8m0fnu`` scales of shape
            ``[.., ceil(rows / block_m), ceil(cols / block_k)]``. The AMD
            matmul consumes this rank-2 layout directly.
        block_size: Per-block size as ``(block_m, block_k)``; ``(1, 32)`` for
            MXFP4.
    """

    #: Dtype of the stored ``weight``.
    storage_dtype: ClassVar[DType] = DType.uint8
    #: Logical values held per stored ``weight`` element.
    values_per_byte: ClassVar[int] = 2
    #: Dtype of ``weight_scale``.
    scale_dtype: ClassVar[DType] = DType.float8_e8m0fnu

    weight_scale: Tensor

    def __init__(
        self,
        *,
        weight: Tensor,
        weight_scale: Tensor,
        block_size: tuple[int, int] = (1, 32),
    ) -> None:
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self._block_size = block_size

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        *,
        block_size: tuple[int, int] = (1, 32),
    ) -> MXFP4Tensor:
        """Builds a zero-filled :class:`MXFP4Tensor` of logical ``[.., rows, cols]``.

        ``shape`` is *unpacked*; the packed ``weight`` leaf has half the
        trailing dimension. Leading dimensions are kept on both leaves.
        """
        *lead, rows, cols = (int(d) for d in shape)
        block_m, block_k = block_size
        return cls(
            weight=Tensor.zeros(
                (*lead, rows, cols // cls.values_per_byte),
                dtype=cls.storage_dtype,
            ),
            weight_scale=Tensor.zeros(
                (*lead, _ceildiv(rows, block_m), _ceildiv(cols, block_k)),
                dtype=cls.scale_dtype,
            ),
            block_size=block_size,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        """Quantization block shape as ``(rows, cols)``."""
        return self._block_size

    def shard(self, axis: int, mesh: DeviceMesh) -> MXFP4Tensor:
        """Co-shard ``weight``/``weight_scale`` along ``axis`` onto ``mesh``."""
        mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
        return MXFP4Tensor(
            weight=self.weight.to(mapping),
            weight_scale=self.weight_scale.to(mapping),
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[MXFP4Tensor, ...]:
        """Per-device shards, pairing each ``weight`` shard with its scales."""
        return tuple(
            MXFP4Tensor(
                weight=self.weight.local_shards[i],
                weight_scale=self.weight_scale.local_shards[i],
                block_size=self.block_size,
            )
            for i in range(self.weight.num_shards)
        )

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh of the packed ``weight`` tensor."""
        return self.weight.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.weight.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.weight._mapping = mapping
        self.weight_scale._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> MXFP4Tensor:
        """Returns a copy with ``weight`` and ``weight_scale`` on ``target``."""
        return MXFP4Tensor(
            weight=self.weight.to(target),
            weight_scale=self.weight_scale.to(target),
            block_size=self.block_size,
        )


# --------------------------------------------------------------------------- #
# Activation containers
# --------------------------------------------------------------------------- #


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


@module_dataclass(frozen=True)
class FP8BlockActivation(QTensor):
    """FP8 block-quantized activation tensor.

    Args:
        data: ``float8_e4m3fn`` tokens of shape ``[total_tokens, K]``.
        scales: ``float32`` scales, one per ``(1, block_k)`` block.
    """

    data: Tensor
    scales: Tensor


# --------------------------------------------------------------------------- #
# On-the-fly quantizers
#
# Each holds a native-precision weight and is also its format's container,
# reading the container's fields from the quantization it builds on first use.
# They are written against ``Tensor`` and the ``F.functional`` kernel
# wrappers, so they trace the same way eagerly and under ``F.lazy()`` with no
# ``TensorValue`` round trip.
# --------------------------------------------------------------------------- #


def _scales_as_rows(scales: Tensor, rows: int, k_blocks: int) -> Tensor:
    """Returns block scales in the rank-2 ``[rows, K // block]`` layout.

    The block-scaled quantize kernel hands back the rank-5 SF-atom layout on
    NVIDIA SM100/SM120 and rank-2 elsewhere. Containers hold rank-2 -- the
    layout a checkpoint stores -- so dispatch sees one shape whatever
    produced the weight, and the matmul re-interleaves where it needs to.

    The rank-5 layout ``[ceil(rows / 128), ceil(k_blocks / 4), 32, 4, 4]``
    pads rows to 128 and scale columns to 4, so the flattened tensor is
    sliced back to the logical extent rather than reshaped onto it.
    """
    if len(scales.shape) == 2:
        return scales
    m_tiles, k_tiles = int(scales.shape[0]), int(scales.shape[1])
    padded = scales.permute([0, 3, 2, 1, 4]).reshape(
        [m_tiles * 128, k_tiles * 4]
    )
    return padded[:rows, :k_blocks]


class _Quantizer(QTensor):
    """Wraps a native-precision weight, quantizing it on first use.

    Unlike the containers above, which hold a checkpoint's pre-quantized
    weight and scales, this holds the checkpoint's native-precision
    ``weight``, as ``_tensor``. A subclass also derives from its format's
    container and exposes the container's fields as read-only properties,
    so quant ops dispatch on it as on the container. The properties read
    :attr:`quant_tensor`, which quantizes on first use and caches the
    container, so repeated matmuls -- eager or traced -- quantize once. A
    subclass wanting to re-quantize under different parameters invalidates
    ``quant_tensor`` instead.

    ``_tensor`` is the only state: the scales are derived from it. Placement
    therefore moves ``_tensor`` and drops any cached quantization, which is
    rebuilt on the new placement at next use (see :meth:`to`).

    Everything here is format-independent; a subclass supplies only
    ``_quantize`` and its container's fields.
    """

    _tensor: Tensor
    _block_size: tuple[int, int]

    def __init__(self, tensor: Tensor, block_size: tuple[int, int]):
        if int(tensor.shape[-1]) % block_size[-1]:
            raise ValueError(
                f"{type(self).__name__} needs a K that is a multiple of "
                f"{block_size[-1]}, got {tensor.shape[-1]}"
            )
        self._tensor = tensor
        self._block_size = block_size

    def _quantize(self) -> QTensor:
        """Produces the quantized container. The only format-specific part."""
        raise NotImplementedError

    @cached_property
    def quant_tensor(self) -> Any:
        """Holds a cached quantized tensor."""
        return self._quantize()

    @property
    def shape(self) -> Shape:
        """Delegates shape to underlying tensor."""
        return self._tensor.shape

    @property
    def block_size(self) -> tuple[int, int]:
        """Quantization block shape as ``(rows, cols)``."""
        return self._block_size

    def _scale_grid(self) -> tuple[int, int]:
        """The ``(rows, K // block)`` extent of the rank-2 block scales."""
        return (
            int(self._tensor.shape[0]),
            int(self._tensor.shape[1]) // self.block_size[1],
        )

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        # Not vars(self): that also holds the cached quant_tensor once it is
        # computed, which is no weight to load. The tensor is keyed by its
        # checkpoint name, though the attribute `weight` is the packed field.
        return {"weight": self._tensor, "_block_size": self._block_size}, None

    def __tree_setattr__(self, key: Any, value: Any) -> None:
        super().__tree_setattr__("_tensor" if key == "weight" else key, value)

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Places the wrapped tensor on ``target``, in place.

        In place because :meth:`~max.experimental.nn.Module.to` discards a
        child's return value. The cached quantization, if any, was built on
        the old placement, so it is dropped rather than moved: its scales are
        a function of the wrapped tensor and are rebuilt where it now lives.
        """
        self._tensor = self._tensor.to(target)
        self.__dict__.pop("quant_tensor", None)
        return self


class NVFP4TensorQuantizer(_Quantizer, NVFP4Tensor):
    """On-the-fly NVFP4 quantization of a bf16 weight."""

    def __init__(
        self,
        tensor: Tensor,
        block_size: tuple[int, int] = (1, 16),
    ):
        super().__init__(tensor, block_size)

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

        packed, scales = quantize_dynamic_block_scaled(
            self._tensor,
            tensor_sf=encode_scale,
            sf_vector_size=self.block_size[1],
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,
        )
        input_scale = Tensor.ones((), dtype=DType.float32, device=CPU())

        return NVFP4Tensor(
            weight=packed,
            weight_scale=_scales_as_rows(scales, *self._scale_grid()),
            weight_scale_2=decode_scale,
            input_scale=input_scale,
        )

    @cached_property
    def quant_tensor(self) -> NVFP4Tensor:
        """The NVFP4 quantization of the wrapped tensor, built on first use."""
        return self._quantize()

    # The base class stores these as attributes. These stay read-only views
    # of the cached quantized tensor so dispatch can read them the same way.
    @property
    def weight(self) -> Tensor:  # type: ignore[override]
        """Packed NVFP4 weights."""
        return self.quant_tensor.weight

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


class MXFP8Quantizer(_Quantizer, MXFP8Tensor):
    """On-the-fly MXFP8 quantization of a bf16 weight.

    MXFP8 is pure block scaling, so unlike :class:`NVFP4TensorQuantizer`
    there is no amax pass and no per-tensor scale: ``tensor_sf`` stays at its
    identity default.
    """

    def __init__(
        self,
        tensor: Tensor,
        block_size: tuple[int, int] = (1, 32),
    ):
        super().__init__(tensor, block_size)

    def _quantize(self) -> MXFP8Tensor:
        data, scales = quantize_dynamic_block_scaled(
            self._tensor,
            sf_vector_size=self.block_size[1],
            scales_type=MXFP8Tensor.scale_dtype,
            out_type=MXFP8Tensor.storage_dtype,
        )
        return MXFP8Tensor(
            weight=data,
            weight_scale=_scales_as_rows(scales, *self._scale_grid()),
            block_size=self.block_size,
        )

    @cached_property
    def quant_tensor(self) -> MXFP8Tensor:
        """The MXFP8 quantization of the wrapped tensor, built on first use."""
        return self._quantize()

    # Read-only views of the cached quantized tensor, as for NVFP4.
    @property
    def weight(self) -> Tensor:  # type: ignore[override]
        """Quantized ``float8_e4m3fn`` weights."""
        return self.quant_tensor.weight

    @property
    def weight_scale(self) -> Tensor:  # type: ignore[override]
        """E8M0 block scales."""
        return self.quant_tensor.weight_scale


class MXFP4Quantizer(_Quantizer, MXFP4Tensor):
    """On-the-fly MXFP4 quantization of a bf16 weight.

    Uses the dedicated MXFP4 op rather than the shared block-scaled one. It
    returns rank-2 ``[N, K // 32]`` scales on every target, which is both the
    layout the container holds and the layout the AMD matmul consumes.
    """

    def __init__(
        self,
        tensor: Tensor,
        block_size: tuple[int, int] = (1, 32),
    ):
        super().__init__(tensor, block_size)

    def _quantize(self) -> MXFP4Tensor:
        data, scales = quantize_dynamic_block_scaled_mxfp4(
            self._tensor,
            scales_type=MXFP4Tensor.scale_dtype,
            out_type=MXFP4Tensor.storage_dtype,
        )
        return MXFP4Tensor(
            weight=data, weight_scale=scales, block_size=self.block_size
        )

    @cached_property
    def quant_tensor(self) -> MXFP4Tensor:
        """The MXFP4 quantization of the wrapped tensor, built on first use."""
        return self._quantize()

    # Read-only views of the cached quantized tensor, as for NVFP4.
    @property
    def weight(self) -> Tensor:  # type: ignore[override]
        """Packed MXFP4 weights, two ``e2m1`` codes per byte."""
        return self.quant_tensor.weight

    @property
    def weight_scale(self) -> Tensor:  # type: ignore[override]
        """E8M0 block scales."""
        return self.quant_tensor.weight_scale


# --------------------------------------------------------------------------- #
# Type aliases and guards
# --------------------------------------------------------------------------- #


QuantAwareTensor: TypeAlias = (
    Tensor
    | FP8BlockTensor
    | NVFP4Tensor
    | MXFP8Tensor
    | MXFP4Tensor
    | NVFP4TensorQuantizer
    | MXFP8Quantizer
    | MXFP4Quantizer
    | NVFP4Activation
    | FP8BlockActivation
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


def all_mxfp8(
    weights: Sequence[QuantAwareTensor],
) -> TypeGuard[Sequence[MXFP8Tensor]]:
    """Narrow a sequence of mixed weights to a homogeneous MXFP8 sequence."""
    return all(isinstance(w, MXFP8Tensor) for w in weights)


def all_mxfp4(
    weights: Sequence[QuantAwareTensor],
) -> TypeGuard[Sequence[MXFP4Tensor]]:
    """Narrow a sequence of mixed weights to a homogeneous MXFP4 sequence."""
    return all(isinstance(w, MXFP4Tensor) for w in weights)


__all__ = [
    "FP8BlockActivation",
    "FP8BlockTensor",
    "MXFP4Quantizer",
    "MXFP4Tensor",
    "MXFP8Quantizer",
    "MXFP8Tensor",
    "NVFP4Activation",
    "NVFP4Tensor",
    "NVFP4TensorQuantizer",
    "QTensor",
    "QuantAwareTensor",
    "all_fp8_block",
    "all_mxfp4",
    "all_mxfp8",
    "all_nvfp4",
]
