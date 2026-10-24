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

"""Quantization-aware kernel dispatch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from max.driver import CPU, accelerator_api, accelerator_architecture_name
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.functional_kernels import (
    fused_silu,
    fused_silu_quantized,
    grouped_matmul_ragged,
)
from max.experimental.sharding import DeviceMapping, DeviceMesh
from max.experimental.sharding.placements import Sharded
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.activation import GatedActivation
from max.nn.comm.ep import EPConfig
from max.nn.kernels import _is_amd_gpu
from max.nn.kernels import (
    grouped_matmul_blocked_swiglu as _grouped_matmul_blocked_swiglu,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from .quant_kernels import (
    block_scales_interleave,
    dynamic_block_scaled_matmul,
    dynamic_block_scaled_matmul_amd,
    dynamic_scaled_matmul,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled,
    quantize_dynamic_block_scaled,
    quantize_dynamic_block_scaled_mxfp4,
    quantize_dynamic_scaled_float8,
    silu_quantize_fp8,
)
from .quant_tensor import (
    FP8BlockActivation,
    FP8BlockTensor,
    MXFP4Quantizer,
    MXFP4Tensor,
    MXFP8Quantizer,
    MXFP8Tensor,
    NVFP4Activation,
    NVFP4Tensor,
    NVFP4TensorQuantizer,
    QuantAwareTensor,
    all_fp8_block,
    all_mxfp8,
    all_nvfp4,
)

# NVFP4 packs two float4-e2m1 values per byte and uses one float8_e4m3fn scale
# per 16-element block along K.
_NVFP4_SF_VECTOR_SIZE = 16
# Every OCP MX format uses one float8_e8m0fnu scale per 32-element block.
_MX_SF_VECTOR_SIZE = 32


def is_fp8_block_quantized(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if ``quant_config`` selects FP8 block-scaled weights."""
    return (
        quant_config is not None
        and quant_config.format == QuantFormat.BLOCKSCALED_FP8
    )


def is_nvfp4_quantized(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if ``quant_config`` selects NVFP4 block-scaled weights."""
    return quant_config is not None and quant_config.is_nvfp4


def is_mxfp8_quantized(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if ``quant_config`` selects MXFP8 block-scaled weights."""
    return quant_config is not None and quant_config.is_mxfp8


@dataclass(frozen=True)
class EPDispatchPayload:
    """Named, per-device view of the fused EP dispatch outputs.

    - bf16 dispatch: ``tokens``, ``expert_start``, ``expert_ids``,
      ``usage_stats``.
    - FP8 dispatch: adds ``scales``.
    - NVFP4 dispatch: adds ``scales`` and ``scales_offset``; ``usage_stats`` is
      absent (the NVFP4 grouped matmul synthesizes its own host stats).
    """

    tokens: list[Tensor]
    expert_start: list[Tensor]
    expert_ids: list[Tensor]
    usage_stats: list[Tensor] | None = None
    scales: list[Tensor] | None = None
    scales_offset: list[Tensor] | None = None

    @classmethod
    def from_dispatch(
        cls,
        dispatch_results: list[tuple[Tensor, ...]],
        quant_config: QuantConfig | None,
        ep_config: EPConfig,
    ) -> EPDispatchPayload:
        """Parse the per-device dispatch tuples into named per-column fields.

        Args:
            dispatch_results: Per-device dispatch output tuples (the return of
                :meth:`~max.nn.comm.ep.EPBatchManager.ep_dispatch`).
            quant_config: Routed-expert quant config selecting the format.
            ep_config: EP batch-manager config; its ``dispatch_dtype`` decides
                whether a ``scales`` column is present (a separate axis from the
                weight quant format).
        """
        columns = [
            list(column) for column in zip(*dispatch_results, strict=True)
        ]

        if _dispatch_format(quant_config) == QuantFormat.NVFP4:
            tokens, scales, expert_start, scales_offset, expert_ids, _ = columns
            return cls(
                tokens=tokens,
                expert_start=expert_start,
                expert_ids=expert_ids,
                scales=scales,
                scales_offset=scales_offset,
            )
        else:
            fp8_dispatch = (
                ep_config.dispatch_dtype.is_float8()
                and ep_config.dispatch_quant_config is not None
            )
            expert_start, expert_ids, usage_stats = columns[-3:]
            for stat in usage_stats:
                assert stat.device.is_host
            return cls(
                tokens=columns[0],
                expert_start=expert_start,
                expert_ids=expert_ids,
                usage_stats=usage_stats,
                scales=columns[1] if fp8_dispatch else None,
            )

    def per_device_tokens(
        self,
        *,
        nvfp4_global_scale: Tensor | None = None,
    ) -> Sequence[QuantAwareTensor]:
        """Wrap the raw per-device token columns into activations.

        Args:
            nvfp4_global_scale: Uniform NVFP4 activation scale the dispatch
                quantized with (the max static ``input_scale`` across experts);
                required when the payload carries NVFP4 tokens.
        """
        if self.scales_offset is not None:
            assert self.scales is not None and nvfp4_global_scale is not None, (
                "NVFP4 tokens require dispatch scales and the uniform scale"
            )
            return [
                NVFP4Activation(
                    data=d,
                    scales=s,
                    input_scale=nvfp4_global_scale,
                    scales_offset=o,
                )
                for d, s, o in zip(
                    self.tokens, self.scales, self.scales_offset, strict=True
                )
            ]
        if self.scales is not None:
            return [
                FP8BlockActivation(data=d, scales=s)
                for d, s in zip(self.tokens, self.scales, strict=True)
            ]
        return self.tokens


def _dispatch_format(quant_config: QuantConfig | None) -> QuantFormat | None:
    """The weight format a dispatch kernel will see, or ``None`` if unquantized.

    Routed-expert weights are stacked tensors rather than ``Linear``
    submodules, and on-the-fly quantization only replaces ``Linear``, so it
    cannot produce one through :func:`insert_quantized_linears`. Nothing
    stops a QUANTIZE-sourced config reaching a MoE layer by another route,
    though, and that case has not been reasoned about: whether such a
    dispatch should carry scales is a real question, since its activations
    still need quantizing for an FP4 matmul.

    Answering it here would decide it by default, and the answer sets how
    many columns :meth:`EPDispatchPayload.from_dispatch` unpacks -- so a
    wrong one surfaces as a tuple-arity error well below the cause. Refuse
    instead, at the point the assumption breaks.

    Raises:
        NotImplementedError: if the config quantizes weights on load.
    """
    if quant_config is None:
        return None
    if quant_config.quantizes_on_load:
        raise NotImplementedError(
            "Dispatch does not support weights quantized on load; got "
            f"source={quant_config.source.value} for "
            f"{quant_config.format.value}. Load a pre-quantized checkpoint "
            "for routed-expert weights."
        )
    return quant_config.format


def ep_requires_dispatch_scales(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if the EP dispatch kernel needs input scales.

    Takes a routed-expert config. The answer decides how many columns
    :meth:`EPDispatchPayload.from_dispatch` unpacks, so it describes the
    kernel's output shape rather than a preference.

    Refuses a quantize-on-load config rather than answering -- see
    :func:`_dispatch_format`.
    """
    return _dispatch_format(quant_config) == QuantFormat.NVFP4


def moe_requires_scales_offsets(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if the MoE dispatch kernel needs scales offsets.

    The block-scaled grouped matmuls (NVFP4 and MXFP8) read the activation
    scales at per-expert offsets. Refuses a quantize-on-load config rather
    than answering -- see :func:`_dispatch_format`.
    """
    return _dispatch_format(quant_config) in (
        QuantFormat.NVFP4,
        QuantFormat.MXFP8,
    )


#: The container a pre-quantized checkpoint loads into, per format. Adding a
#: format is an entry here, not a branch in :func:`quantized_weight`.
class _ContainerFactory(Protocol):
    """Builds a zero-filled container for a pre-quantized checkpoint.

    The registries hold classes, so this describes the class object rather
    than the instance. What a format must supply is the ability to build an
    unloaded parameter -- not a common base class, which is why this is a
    protocol and not ``type[QTensor]``.
    """

    def zeros(
        self, shape: tuple[int, int], *, block_size: tuple[int, int]
    ) -> QuantAwareTensor: ...


class _QuantizerFactory(Protocol):
    """Wraps a native-precision tensor, quantizing it on first use."""

    def __call__(
        self, tensor: Tensor, block_size: tuple[int, int]
    ) -> QuantAwareTensor: ...


_CONTAINERS: dict[QuantFormat, _ContainerFactory] = {
    QuantFormat.BLOCKSCALED_FP8: FP8BlockTensor,
    QuantFormat.NVFP4: NVFP4Tensor,
    QuantFormat.MXFP8: MXFP8Tensor,
    QuantFormat.MXFP4: MXFP4Tensor,
}

#: The wrapper that quantizes a native-precision weight on first use. Only
#: formats that can be produced on the fly appear here, which is why this is
#: a separate mapping rather than a field on the container.
_QUANTIZERS: dict[QuantFormat, _QuantizerFactory] = {
    QuantFormat.NVFP4: NVFP4TensorQuantizer,
    QuantFormat.MXFP8: MXFP8Quantizer,
    QuantFormat.MXFP4: MXFP4Quantizer,
}

#: Dtype of the weight as the CHECKPOINT holds it, per format.
_STORAGE_DTYPES: dict[QuantFormat, DType] = {
    QuantFormat.BLOCKSCALED_FP8: DType.float8_e4m3fn,
    QuantFormat.NVFP4: DType.uint8,
    QuantFormat.MXFP8: MXFP8Tensor.storage_dtype,
    QuantFormat.MXFP4: MXFP4Tensor.storage_dtype,
}


def routed_weight_dtype(quant_config: QuantConfig | None) -> DType:
    """The storage dtype :func:`quantized_weight` picks for ``quant_config``.

    Weights quantized on the fly are bf16 on disk whatever they will become.
    """
    if quant_config is None or quant_config.quantizes_on_load:
        return DType.bfloat16
    return _STORAGE_DTYPES.get(quant_config.format, DType.bfloat16)


def quantized_weight(
    out_dim: int,
    in_dim: int,
    quant_config: QuantConfig | None,
) -> QuantAwareTensor:
    """Build a Linear-shaped ``[out_dim, in_dim]`` weight parameter.

    The result is the *unloaded* parameter the weight loader binds into, so
    its dtype has to match what the checkpoint holds: a quantized container
    when the bytes are pre-quantized, a float tensor wrapped in a quantizer
    when they are not.

    There are three cases to consider: no config, a pre-quantized checkpoint,
    and quantize-on-load.
    """
    shape = (int(out_dim), int(in_dim))
    if quant_config is None:
        return Tensor.zeros(shape)

    registry = _QUANTIZERS if quant_config.quantizes_on_load else _CONTAINERS
    if quant_config.format not in registry:
        known = ", ".join(sorted(f.value for f in registry))
        raise ValueError(
            f"Quant type {quant_config.format} is not yet supported"
            + (
                " for on-the-fly quantization"
                if quant_config.quantizes_on_load
                else ""
            )
            + f"; known: {known}."
        )

    block_size = quant_config.weight_scale.block_size
    assert block_size is not None
    if quant_config.quantizes_on_load:
        # Pinned, not left to the ambient default. `Tensor.zeros` picks
        # float32 on CPU and bfloat16 on an accelerator, which would make a
        # parameter's dtype depend on the hardware present when it is built
        # -- and disagree with what `routed_weight_dtype` tells callers the
        # checkpoint holds.
        return _QUANTIZERS[quant_config.format](
            Tensor.zeros(shape, dtype=routed_weight_dtype(quant_config)),
            block_size=block_size,
        )
    return _CONTAINERS[quant_config.format].zeros(shape, block_size=block_size)


def stack(items: list[QuantAwareTensor], axis: int = 0) -> QuantAwareTensor:
    """Stack a homogeneous bundle along ``axis``, dispatching on quant type.

    For FP8 items, both leaves (``weight`` and ``weight_scale_inv``) are stacked and
    rewrapped in an :class:`FP8BlockTensor`; for plain tensors the list is
    stacked directly. Companion to :func:`concat_weights`.

    Args:
        items: Homogeneous list of :class:`QuantAwareTensor`s (all plain
            tensors, or all :class:`FP8BlockTensor`s).
        axis: Axis to stack along (a new dimension is inserted here).

    Returns:
        A single stacked :class:`QuantAwareTensor` of the same kind as
        ``items``.
    """
    first = items[0]
    if isinstance(first, FP8BlockTensor):
        assert all_fp8_block(items)
        return FP8BlockTensor(
            weight=F.stack([w.weight for w in items], axis=axis),
            weight_scale_inv=F.stack(
                [w.weight_scale_inv for w in items], axis=axis
            ),
            block_size=first.block_size,
        )
    if isinstance(first, NVFP4Tensor):
        assert all_nvfp4(items)
        return NVFP4Tensor(
            weight=F.stack([w.weight for w in items], axis=axis),
            weight_scale=F.stack([w.weight_scale for w in items], axis=axis),
            weight_scale_2=F.stack(
                [w.weight_scale_2 for w in items], axis=axis
            ),
            input_scale=F.stack([w.input_scale for w in items], axis=axis),
            block_size=first.block_size,
        )
    if isinstance(first, MXFP8Tensor):
        assert all_mxfp8(items)
        return MXFP8Tensor(
            weight=F.stack([w.weight for w in items], axis=axis),
            weight_scale=F.stack([w.weight_scale for w in items], axis=axis),
            block_size=first.block_size,
        )
    assert all(isinstance(w, Tensor) for w in items)
    return F.stack(list(items), axis=axis)


def combine_quant_per_device(
    items: list[QuantAwareTensor],
    combine: Callable[[list[Tensor]], list[Tensor]],
    *,
    global_scale_items: list[QuantAwareTensor] | None = None,
) -> list[QuantAwareTensor]:
    """Map a per-device leaf transform over a homogeneous bundle.

    ``combine`` merges all items' tensors for one leaf into a per-device list
    (one tensor per mesh device) — for example, a TP shard-and-stack.  For FP8 input,
    ``combine`` is applied to the ``weight`` and ``weight_scale_inv`` leaves
    independently and the per-device leaves are zipped back into one
    :class:`FP8BlockTensor` per device, so the FP8 invariant (``weight`` and
    ``weight_scale_inv`` are each a single :class:`~max.experimental.tensor.Tensor`)
    is preserved without the caller transposing a struct-of-lists into a
    list-of-structs.  For plain tensors the per-device list is returned as-is.

    ``combine`` must be leaf-agnostic — read any per-leaf difference (such as the
    block-scale leaf's smaller trailing dim) off the leaf tensors' own shapes
    rather than branching on which leaf it is.

    Args:
        items: Homogeneous list of :class:`QuantAwareTensor`s (all plain
            tensors, or all :class:`FP8BlockTensor`s).
        combine: Callable that merges a list of leaf tensors into a per-device
            list of tensors.
        global_scale_items: Items to read the NVFP4 per-tensor scales from, when
            ``items`` holds more than one entry per expert. The TP gate/up
            bundle interleaves each expert's gate and up weights, which share
            one global scale (see :func:`concat_nvfp4`), so stacking over
            ``items`` would give two entries per expert. Defaults to ``items``.

    Returns:
        One :class:`QuantAwareTensor` per device. For FP8 input, each is an
        :class:`FP8BlockTensor` whose ``weight``/``weight_scale_inv`` are that device's
        leaves; for plain tensors, the per-device list is returned directly.
    """
    first = items[0]
    if isinstance(first, FP8BlockTensor):
        assert all_fp8_block(items)
        data = combine([w.weight for w in items])
        weight_scale_inv = combine([w.weight_scale_inv for w in items])
        return [
            FP8BlockTensor(
                weight=d, weight_scale_inv=s, block_size=first.block_size
            )
            for d, s in zip(data, weight_scale_inv, strict=True)
        ]
    if isinstance(first, NVFP4Tensor):
        assert all_nvfp4(items)
        scale_items = (
            items if global_scale_items is None else global_scale_items
        )
        assert all_nvfp4(scale_items)
        data = combine([w.weight for w in items])
        weight_scale = combine([w.weight_scale for w in items])
        weight_scale_2 = F.stack(
            [w.weight_scale_2 for w in scale_items], axis=0
        )
        input_scale = F.stack([w.input_scale for w in scale_items], axis=0)
        return [
            NVFP4Tensor(
                weight=d,
                weight_scale=s,
                weight_scale_2=weight_scale_2,
                input_scale=input_scale,
                block_size=first.block_size,
            )
            for d, s in zip(data, weight_scale, strict=True)
        ]
    if isinstance(first, MXFP8Tensor):
        assert all_mxfp8(items)
        data = combine([w.weight for w in items])
        weight_scale = combine([w.weight_scale for w in items])
        return [
            MXFP8Tensor(weight=d, weight_scale=s, block_size=first.block_size)
            for d, s in zip(data, weight_scale, strict=True)
        ]
    plain = [w for w in items if isinstance(w, Tensor)]
    result: list[QuantAwareTensor] = [*combine(plain)]
    return result


def stack_device_shards(
    shards: Sequence[QuantAwareTensor], axis: int, mesh: DeviceMesh
) -> QuantAwareTensor:
    """Reassembles a per-device weight-shard bundle into one ``Sharded`` tensor."""
    if len(shards) == 1:
        return shards[0]
    mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
    first = shards[0]
    if isinstance(first, FP8BlockTensor):
        assert all_fp8_block(shards)
        return FP8BlockTensor(
            weight=Tensor.from_shard_values(
                [TensorValue(s.weight) for s in shards], mapping
            ),
            weight_scale_inv=Tensor.from_shard_values(
                [TensorValue(s.weight_scale_inv) for s in shards], mapping
            ),
            block_size=first.block_size,
        )
    if isinstance(first, NVFP4Tensor):
        assert all_nvfp4(shards)
        return NVFP4Tensor(
            weight=Tensor.from_shard_values(
                [TensorValue(s.weight) for s in shards], mapping
            ),
            weight_scale=Tensor.from_shard_values(
                [TensorValue(s.weight_scale) for s in shards], mapping
            ),
            weight_scale_2=first.weight_scale_2,
            input_scale=first.input_scale,
            block_size=first.block_size,
        )
    if isinstance(first, MXFP8Tensor):
        assert all_mxfp8(shards)
        return MXFP8Tensor(
            weight=Tensor.from_shard_values(
                [TensorValue(s.weight) for s in shards], mapping
            ),
            weight_scale=Tensor.from_shard_values(
                [TensorValue(s.weight_scale) for s in shards], mapping
            ),
            block_size=first.block_size,
        )
    return Tensor.from_shard_values(
        [TensorValue(s) for s in shards if isinstance(s, Tensor)], mapping
    )


def concat_weights(
    *weights: QuantAwareTensor, axis: int = 0
) -> QuantAwareTensor:
    """Concatenate weights along ``axis``, dispatching on the weight type."""
    if not weights:
        raise ValueError("concat_weights requires at least one tensor")
    if isinstance(weights[0], FP8BlockTensor):
        assert all_fp8_block(weights), (
            "concat_weights requires all weights to be FP8BlockTensor when "
            "the first is"
        )
        return concat_fp8_block(*weights, axis=axis)
    if isinstance(weights[0], NVFP4Tensor):
        assert all_nvfp4(weights), (
            "concat_weights requires all weights to be NVFP4Tensor when the "
            "first is"
        )
        return concat_nvfp4(*weights, axis=axis)
    if isinstance(weights[0], MXFP8Tensor):
        assert all_mxfp8(weights), (
            "concat_weights requires all weights to be MXFP8Tensor when the "
            "first is"
        )
        return concat_mxfp8(*weights, axis=axis)
    assert all(isinstance(w, Tensor) for w in weights)
    return F.concat(list(weights), axis=axis)


def _fp8_block_specs(
    weight_block: tuple[int, int],
    *,
    input_block: tuple[int, int] = (1, 128),
) -> tuple[InputScaleSpec, WeightScaleSpec]:
    """Standard FP8 block-scale specs for matmul/grouped-matmul kernels."""
    return (
        InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=input_block,
        ),
        WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=weight_block,
        ),
    )


def matmul(x: Tensor, weight: QuantAwareTensor) -> Tensor:
    """Matmul ``x @ weight.T`` dispatching on the weight type.

    ``weight`` follows the Linear convention: shape ``[out_dim, in_dim]``.

    - ``Tensor`` weight: regular bf16/float matmul.
    - :class:`FP8BlockTensor` weight: quantizes ``x`` to FP8 with
      ``(1, block_k)`` activation blocks, then runs the block-scaled FP8
      matmul kernel and returns bf16.
    - :class:`NVFP4Tensor` weight: normalizes ``x`` by the static
      ``input_scale``, dynamically block-quantizes it to FP4, then runs the
      SM100 block-scaled FP4 matmul and returns bf16.
    - :class:`MXFP8Tensor` weight: dynamically block-quantizes ``x`` to
      MXFP8, then runs the SM100 block-scaled matmul and returns bf16. No
      input scale: MX is pure block scaling.
    - :class:`MXFP4Tensor` weight: dynamically block-quantizes ``x`` to
      MXFP4, then runs the AMD block-scaled matmul and returns bf16. Raises
      off AMD, where there is no MXFP4 matmul.
    - Any on-the-fly quantizer (:class:`NVFP4TensorQuantizer`,
      :class:`MXFP8Quantizer`, :class:`MXFP4Quantizer`): each is its format's
      container, so it takes that container's path and reads the fields of
      its cached quantization.
    """
    if isinstance(weight, FP8BlockTensor):
        return _matmul_fp8_block(x, weight)
    if isinstance(weight, NVFP4Tensor):
        return _matmul_nvfp4(x, weight)
    if isinstance(weight, MXFP8Tensor):
        return _matmul_mxfp8(x, weight)
    if isinstance(weight, MXFP4Tensor):
        return _matmul_mxfp4(x, weight)
    assert isinstance(weight, Tensor)
    return x @ weight.T


def _matmul_mxfp8(x: Tensor, weight: MXFP8Tensor) -> Tensor:
    """Block-scaled MXFP8 matmul ``x @ weight.T`` with dynamic activation quant.

    MXFP8 is pure block scaling, so no ``tensor_sf`` is threaded through:
    both operands carry their own E8M0 scales and nothing rescales the
    output. Shares the SM100 kernel with NVFP4, which selects the format
    from the operand dtype.
    """
    x_fp8, x_scales = quantize_dynamic_block_scaled(
        x,
        sf_vector_size=_MX_SF_VECTOR_SIZE,
        scales_type=MXFP8Tensor.scale_dtype,
        out_type=MXFP8Tensor.storage_dtype,
    )
    # Lift the rank-2 [N, K // 32] scales the container holds into the
    # rank-5 SF-atom layout the MMA consumes.
    weight_scale = block_scales_interleave(
        weight.weight_scale, sf_vector_size=_MX_SF_VECTOR_SIZE
    )
    return dynamic_block_scaled_matmul(
        x_fp8,
        weight.weight,
        x_scales,
        weight_scale,
        sf_vector_size=_MX_SF_VECTOR_SIZE,
        out_type=DType.bfloat16,
    )


def _matmul_mxfp4(x: Tensor, weight: MXFP4Tensor) -> Tensor:
    """Block-scaled MXFP4 matmul ``x @ weight.T``. AMD CDNA only.

    MXFP4 has no NVIDIA path: the SM100 block-scaled op takes ``uint8``
    operands only with ``float8_e4m3fn`` scales, so MXFP4's E8M0 scales are
    rejected. Checked up front so the failure names the hardware rather than
    surfacing as a scales-dtype mismatch inside the kernel.

    Both operands carry rank-2 ``[N, K // 32]`` scales, which the AMD kernel
    consumes directly -- no SF-atom interleave, unlike
    :func:`_matmul_mxfp8`.
    """
    if not _is_amd_gpu():
        raise NotImplementedError(
            "MXFP4 matmul requires an AMD CDNA GPU. The SM100 block-scaled "
            "kernel supports only NVFP4 or MXFP8; use MXFP8 on NVIDIA."
        )
    x_fp4, x_scales = quantize_dynamic_block_scaled_mxfp4(
        x,
        scales_type=MXFP4Tensor.scale_dtype,
        out_type=MXFP4Tensor.storage_dtype,
    )
    return dynamic_block_scaled_matmul_amd(
        x_fp4,
        weight.weight,
        x_scales,
        weight.weight_scale,
        out_type=DType.bfloat16,
    )


def _matmul_nvfp4(x: Tensor, weight: NVFP4Tensor) -> Tensor:
    """Block-scaled NVFP4 matmul ``x @ weight.T`` with dynamic activation quant."""
    x_fp4, x_scales = quantize_dynamic_block_scaled(
        x,
        tensor_sf=1.0 / weight.input_scale,
        sf_vector_size=_NVFP4_SF_VECTOR_SIZE,
        scales_type=DType.float8_e4m3fn,
        out_type=DType.uint8,
    )
    weight_scale = block_scales_interleave(
        weight.weight_scale, sf_vector_size=_NVFP4_SF_VECTOR_SIZE
    )
    return dynamic_block_scaled_matmul(
        x_fp4,
        weight.weight,
        x_scales,
        weight_scale,
        tensor_sf=weight.weight_scale_2 * weight.input_scale,
        sf_vector_size=_NVFP4_SF_VECTOR_SIZE,
        out_type=DType.bfloat16,
    )


def _matmul_fp8_block(x: Tensor, weight: FP8BlockTensor) -> Tensor:
    """Block-scaled FP8 matmul ``x @ weight.T``.

    Activations are quantized dynamically.
    The activation block is ``(1, block_k)`` and the weight block is
    ``(block_m, block_k) = weight.block_size``. The kernel returns bf16.
    """
    block_m, block_k = weight.block_size
    input_spec, weight_spec = _fp8_block_specs(
        (block_m, block_k), input_block=(1, block_k)
    )

    x_fp8, x_scales = quantize_dynamic_scaled_float8(
        x,
        input_spec,
        weight_spec,
        group_size_or_per_token=block_k,
        scales_type=DType.float32,
        out_type=DType.float8_e4m3fn,
    )

    return dynamic_scaled_matmul(
        x_fp8,
        weight.weight,
        x_scales,
        weight.weight_scale_inv,
        input_spec,
        weight_spec,
        out_type=DType.bfloat16,
    )


def grouped_matmul(
    x: QuantAwareTensor,
    weight: QuantAwareTensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_usage_stats: Tensor | None,
    *,
    scales_offset: Tensor | None = None,
    out_type: DType = DType.bfloat16,
    estimated_total_m: Tensor | None = None,
    decode_grid_m_cap: int = 0,
) -> Tensor:
    """Grouped (MoE) matmul dispatching on the stacked-weight type.

    For a plain ``Tensor`` weight of shape ``[num_experts, N, K]``, this
    falls back to the standard ragged grouped matmul. For an
    :class:`FP8BlockTensor` weight, it runs the block-scaled FP8 grouped
    matmul kernel. For an :class:`NVFP4Tensor` weight, it runs the SM100
    block-scaled FP4 grouped matmul (``scales_offset`` required); likewise
    for an :class:`MXFP8Tensor` weight with the block-scaled FP8 kernel.

    Args:
        x: Ragged activations of shape ``[total_tokens, K]``.
        weight: Stacked expert weights, ``[num_experts, N, K]``.
        expert_start_indices: Ragged group offsets, ``uint32``.
        expert_ids: Per-group expert id, ``int32``.
        expert_usage_stats: ``[max_tokens_per_expert, num_active_experts]``
            device tensor. The bf16 fallback requires it on-device (the SM100
            kernel reads ``num_active_experts`` there); the FP8 branch copies it
            to CPU itself. The NVFP4 branch synthesizes its own host stats and
            ignores this (so it may be ``None``).
        scales_offset: Per-expert block-scale offsets from
            :func:`moe_create_indices` with ``needs_scales_offset=True``;
            required for the NVFP4 branch, ignored otherwise.
        out_type: Output dtype for the FP8/NVFP4 branch (bf16 by default).
        estimated_total_m: Host scalar estimate of the non-padded token-expert
            row count. The SM100 NVFP4 dispatch picks its tile regime from it,
            and the MI355 FP8 kernel bounds its grid.y with it; ignored by the
            bf16 branch.
        decode_grid_m_cap: Capture-safe decode grid.y cap for the MI355 FP8
            kernel; 0 disables. Ignored elsewhere.
    """
    if isinstance(weight, NVFP4Tensor):
        if isinstance(x, NVFP4Activation):
            x_fp4, x_scales = x.data, x.scales
            offsets, act_scale = x.scales_offset, x.input_scale
        else:
            assert isinstance(x, Tensor), (
                "NVFP4 grouped matmul expects bf16 activations (quantized here)"
            )
            assert scales_offset is not None, (
                "NVFP4 grouped matmul requires scales_offset from "
                "moe_create_indices(needs_scales_offset=True)"
            )
            offsets = scales_offset
            x_fp4, x_scales = grouped_quantize_dynamic_block_scaled(
                x,
                row_offsets=expert_start_indices,
                scales_offsets=scales_offset,
                expert_ids=expert_ids,
                # NVFP4Tensor keeps its scalars on the host.
                sf_tensor=(1.0 / weight.input_scale).to(x.mesh),
                sf_vector_size=_NVFP4_SF_VECTOR_SIZE,
                scales_type=DType.float8_e4m3fn,
                out_type=DType.uint8,
            )
            act_scale = weight.input_scale
        expert_scales = (weight.weight_scale_2 * act_scale).to(x_fp4.mesh)
        return _block_scaled_grouped_matmul(
            x_fp4,
            weight,
            x_scales,
            offsets,
            expert_start_indices,
            expert_ids,
            expert_scales,
            estimated_total_m=estimated_total_m,
            out_type=out_type,
        )
    if isinstance(weight, MXFP8Tensor):
        assert isinstance(x, Tensor), (
            "MXFP8 grouped matmul expects bf16 activations (quantized here)"
        )
        assert scales_offset is not None, (
            "MXFP8 grouped matmul requires scales_offset from "
            "moe_create_indices(needs_scales_offset=True)"
        )
        # MXFP8 has no per-tensor scale; the kernels take unit ones.
        unit_scales = F.broadcast_to(
            F.constant(1.0, DType.float32, device=x.device), expert_ids.shape
        )
        x_fp8, x_scales = grouped_quantize_dynamic_block_scaled(
            x,
            row_offsets=expert_start_indices,
            scales_offsets=scales_offset,
            expert_ids=expert_ids,
            sf_tensor=unit_scales,
            sf_vector_size=_MX_SF_VECTOR_SIZE,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        return _block_scaled_grouped_matmul(
            x_fp8,
            weight,
            x_scales,
            scales_offset,
            expert_start_indices,
            expert_ids,
            unit_scales,
            estimated_total_m=estimated_total_m,
            out_type=out_type,
        )
    if isinstance(weight, FP8BlockTensor):
        if isinstance(x, FP8BlockActivation):
            # Activations already block-quantized (e.g. by the EP FP8 dispatch
            # or a quantized SiLU); pass them through directly.
            x_fp8, x_scales = x.data, x.scales
        else:
            assert isinstance(x, Tensor)
            x_fp8, x_scales = _quantize_activation_fp8(x, weight.block_size)
        assert expert_usage_stats is not None
        return _grouped_fp8_matmul(
            x_fp8,
            x_scales,
            weight,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            out_type=out_type,
            estimated_total_m=estimated_total_m,
            decode_grid_m_cap=decode_grid_m_cap,
        )
    assert isinstance(x, Tensor) and isinstance(weight, Tensor)
    assert scales_offset is None
    assert expert_usage_stats is not None
    return grouped_matmul_ragged(
        x, weight, expert_start_indices, expert_ids, expert_usage_stats
    )


def interleave_grouped_scales(
    scales: Tensor, sf_vector_size: int = _NVFP4_SF_VECTOR_SIZE
) -> Tensor:
    """Interleaves each expert's block scales into the TCGEN layout.

    Args:
        scales: The ``[experts, M, K // sf_vector_size]`` block scales, maybe
            sharded on ``M`` or ``K``, in which case each device interleaves
            its share.
        sf_vector_size: Elements per block: 16 for NVFP4, 32 for MXFP8.

    Returns:
        The ``[experts, ceildiv(M, 128), ceildiv(K // sf_vector_size, 4), 32,
        4, 4]`` scales.
    """
    num_experts = int(scales.shape[0])
    m = scales.shape[1]
    k = scales.shape[2]
    per_expert = F.split(scales, [1] * num_experts, axis=0)
    return F.stack(
        [
            block_scales_interleave(
                s.reshape([m, k]), sf_vector_size=sf_vector_size
            )
            for s in per_expert
        ],
        axis=0,
    )


def _block_scaled_grouped_matmul(
    x_quantized: Tensor,
    weight: NVFP4Tensor | MXFP8Tensor,
    x_scales: Tensor,
    scales_offset: Tensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_scales: Tensor,
    *,
    estimated_total_m: Tensor | None = None,
    out_type: DType = DType.bfloat16,
) -> Tensor:
    """Block-scaled NVFP4 or MXFP8 grouped matmul on quantized activations."""
    b_scales = interleave_grouped_scales(
        weight.weight_scale, sf_vector_size=weight.block_size[1]
    )
    # With no estimate, the dispatcher picks its tile regime from the row
    # count here, so it must be the routed rows, not a prefill-sized bound.
    usage_stats_host = F.shape_to_tensor(
        [x_quantized.shape[0], expert_ids.shape[0]]
    ).cast(DType.uint32)
    return grouped_matmul_block_scaled(
        x_quantized,
        weight.weight,
        x_scales,
        b_scales,
        expert_start_indices,
        scales_offset,
        expert_ids,
        expert_scales,
        usage_stats_host,
        out_type=out_type,
        estimated_total_m=TensorValue(estimated_total_m)
        if estimated_total_m is not None
        else None,
    )


def sigma_permute_gate_up_nvfp4(weight: NVFP4Tensor) -> NVFP4Tensor:
    """Sigma-permutation when fused SwiGLU+NVFP4 is enabled.

    The stacked [2E, scale_m, scale_k] tensor splits to
    [E, 2, scale_m, scale_k], then permute axes 1,2 → collapse to
    [E, 2*scale_m, scale_k] with rows row-interleaved (g_0, u_0, ...).
    This sits before NvMxf4f8Strategy.prepare_weight_scales lifts to the
    5D tcgen05 layout the kernel expects.
    """

    def _perm(t: Tensor) -> Tensor:
        e = int(t.shape[0])
        two_d = int(t.shape[1])
        c = int(t.shape[-1])
        return (
            t.reshape([e, 2, two_d // 2, c])
            .permute([0, 2, 1, 3])
            .reshape([e, two_d, c])
        )

    return NVFP4Tensor(
        weight=_perm(weight.weight),
        weight_scale=_perm(weight.weight_scale),
        weight_scale_2=weight.weight_scale_2,
        input_scale=weight.input_scale,
        block_size=weight.block_size,
    )


def grouped_matmul_swiglu_nvfp4_ep(
    tokens: Tensor,
    token_scales: Tensor,
    weight: NVFP4Tensor,
    expert_start_indices: Tensor,
    scales_offset: Tensor,
    expert_ids: Tensor,
    expert_scales: Tensor,
    down_input_scale: Tensor,
    *,
    estimated_total_m: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused NVFP4 gate/up grouped matmul + SwiGLU + re-quantize (EP path)."""
    b_scales = interleave_grouped_scales(weight.weight_scale)
    num_active = int(expert_ids.shape[0])
    usage_stats_host = F.constant(
        [8192, num_active], dtype=DType.uint32, device=CPU()
    )
    # The fused kernel consumes the inverted down input scale internally;
    # invert here to match the fused_silu_quantized path, which passes the raw
    # scale (and inverts it itself).
    c_input_scales = 1.0 / down_input_scale
    c_packed, c_scales = _grouped_matmul_blocked_swiglu(
        TensorValue(tokens),
        TensorValue(weight.weight),
        TensorValue(token_scales),
        TensorValue(b_scales),
        TensorValue(expert_start_indices),
        TensorValue(scales_offset),
        TensorValue(expert_ids),
        TensorValue(usage_stats_host),
        expert_scales=TensorValue(expert_scales.to(tokens.device)),
        c_input_scales=TensorValue(c_input_scales.to(tokens.device)),
        estimated_total_m=TensorValue(estimated_total_m)
        if estimated_total_m is not None
        else None,
    )
    return Tensor.from_graph_value(c_packed), Tensor.from_graph_value(c_scales)


def grouped_matmul_silu(
    tokens: QuantAwareTensor,
    gate_up: QuantAwareTensor,
    down: QuantAwareTensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_usage_stats: Tensor | None,
    quant_config: QuantConfig | None,
    scales_offset: Tensor | None = None,
    estimated_total_m: Tensor | None = None,
    accum_dtype: DType | None = None,
    decode_grid_m_cap: int = 0,
    swiglu_limit: float = 0.0,
) -> QuantAwareTensor:
    """Gate/up grouped matmul + SwiGLU, returning the down-projection input.

    ``accum_dtype`` upcasts the gate/up output before the SwiGLU, which
    DeepSeek-V3.2 does in float32. A non-zero ``swiglu_limit`` clamps the
    activation's inputs, ``silu(min(gate, limit)) * clamp(up, -limit,
    limit)``, which GLM-5.3-Flash needs.
    """
    # Pre-quantized EP activations carry their own per-expert scale offset;
    # single-device tokens rely on the caller-supplied one.
    offset = (
        tokens.scales_offset
        if isinstance(tokens, NVFP4Activation)
        else scales_offset
    )
    if (
        isinstance(tokens, NVFP4Activation)
        and quant_config is not None
        and quant_config.can_use_fused_swiglu
        # The fused kernel computes the activation internally, so it cannot
        # carry a float32 gate/up accumulation. ``ExpertParallelMoE``'s weight
        # layout is gated on the same condition.
        and accum_dtype is None
        and not swiglu_limit
    ):
        assert isinstance(gate_up, NVFP4Tensor) and isinstance(
            down, NVFP4Tensor
        )
        assert offset is not None
        # Gate/up epilogue uses the uniform scale the dispatch quantized with;
        # the down input is re-quantized with the per-expert down.input_scale.
        gate_up_scale = gate_up.weight_scale_2 * tokens.input_scale
        c_packed, c_scales = grouped_matmul_swiglu_nvfp4_ep(
            tokens.data,
            tokens.scales,
            gate_up,
            expert_start_indices,
            offset,
            expert_ids,
            gate_up_scale,
            down.input_scale,
            estimated_total_m=estimated_total_m,
        )
        return NVFP4Activation(
            data=c_packed,
            scales=c_scales,
            input_scale=down.input_scale,
            scales_offset=offset,
        )

    gate_up_out = grouped_matmul(
        tokens,
        gate_up,
        expert_start_indices,
        expert_ids,
        expert_usage_stats,
        scales_offset=scales_offset,
        estimated_total_m=estimated_total_m,
        decode_grid_m_cap=decode_grid_m_cap,
    )
    # Fuse the down-proj re-quant into the SiLU only when the activation was
    # pre-quantized upstream (EP dispatch); otherwise emit bf16 and let the
    # down matmul quantize.
    prequantized = isinstance(tokens, (FP8BlockActivation, NVFP4Activation))
    # The down matmul quantizes from bf16 only, so a float32 SwiGLU output has
    # to be packed by the fused quantized SiLU. A bf16 output does not need it
    # and must not take it. That kernel gives each expert group a share of the
    # GPU's SMs, so it cannot be built for a model with more experts than the
    # GPU has SMs.
    requant_weight = (
        down
        if prequantized
        or (isinstance(down, NVFP4Tensor) and accum_dtype is not None)
        else None
    )
    if accum_dtype is not None:
        gate_up_out = gate_up_out.cast(accum_dtype)
    return grouped_silu(
        gate_up_out,
        expert_start_indices,
        requant_weight,
        quant_config,
        scales_offset=offset,
        swiglu_limit=swiglu_limit,
    )


def _quantize_activation_fp8(
    x: Tensor, weight_block: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Per-token FP8 block quantization of ``x`` matching ``weight_block``."""
    block_m, block_k = weight_block
    input_spec, weight_spec = _fp8_block_specs(
        (block_m, block_k), input_block=(1, block_k)
    )
    return quantize_dynamic_scaled_float8(
        x,
        input_spec,
        weight_spec,
        scales_type=DType.float32,
        out_type=DType.float8_e4m3fn,
    )


def _grouped_fp8_matmul(
    x_fp8: Tensor,
    x_scales: Tensor,
    weight: FP8BlockTensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_usage_stats: Tensor,
    *,
    out_type: DType = DType.bfloat16,
    estimated_total_m: Tensor | None = None,
    decode_grid_m_cap: int = 0,
) -> Tensor:
    """Block-scaled FP8 grouped matmul on already-quantized activations."""
    block_m, block_k = weight.block_size
    input_spec, weight_spec = _fp8_block_specs(
        (block_m, block_k), input_block=(1, block_k)
    )
    # This kernel reads the usage stats host-side, so copy to CPU here rather
    # than at the call site (the bf16 path needs them on-device).
    usage_stats = expert_usage_stats.to(CPU())
    grid_kwargs: dict[str, Any] = {}
    if _amd_tiled_grouped_fp8(x_scales, weight):
        # MI355 replays decode from a captured device graph, so the grid must
        # not come from runtime reads: grid.z uses the static expert count and
        # grid.y is bounded by this step's routed rows.
        grid_kwargs = {
            "static_grid_z": True,
            "decode_grid_m_cap": decode_grid_m_cap,
        }
        if estimated_total_m is not None:
            usage_stats = _bound_grid_y(usage_stats, estimated_total_m)
    return grouped_dynamic_scaled_fp8_matmul(
        x_fp8,
        weight.weight,
        x_scales,
        weight.weight_scale_inv,
        expert_start_indices,
        expert_ids,
        usage_stats,
        input_spec,
        weight_spec,
        out_type=out_type,
        **grid_kwargs,
    )


def _amd_tiled_grouped_fp8(x_scales: Tensor, weight: FP8BlockTensor) -> bool:
    """Returns whether the FP8 grouped matmul takes the MI355 tiled kernel.

    Mirrors the kernel's ``amd_tiled`` predicate: only that kernel accepts
    the capture-safe grid controls, and the other arms reject them.
    """
    n, k = int(weight.weight.shape[1]), int(weight.weight.shape[2])
    return (
        accelerator_api() == "hip"
        and accelerator_architecture_name().startswith("gfx950")
        and x_scales.dtype == DType.float32
        and weight.weight_scale_inv.dtype == DType.float32
        and n % 128 == 0
        and k % 128 == 0
    )


def _bound_grid_y(usage_stats: Tensor, estimated_total_m: Tensor) -> Tensor:
    """Bounds the usage stats' grid.y hint by the step's routed rows.

    ``usage_stats[0]`` is the prefill-sized dispatch bound; at decode it
    launches mostly empty CTAs. The tiled kernel strides over M tiles, so a
    smaller grid.y only packs the same tiles into fewer CTAs. The floor of 1
    keeps a ``top_k < n_gpus`` estimate that floors to 0 from skipping the
    kernel entirely.
    """
    device = usage_stats.device
    bounded = F.max(
        F.min(usage_stats[0], estimated_total_m.to(device)),
        F.constant(1, DType.uint32, device=device),
    )
    return F.stack([bounded, usage_stats[1]])


def grouped_silu(
    x: Tensor,
    expert_start_indices: Tensor,
    out_weight: QuantAwareTensor | None = None,
    quant_config: QuantConfig | None = None,
    *,
    scales_offset: Tensor | None = None,
    swiglu_limit: float = 0.0,
) -> QuantAwareTensor:
    """SiLU-gate a grouped gate/up output.

    A non-zero ``swiglu_limit`` clamps the gate from above and the up
    projection on both sides before the activation.
    """
    activation = (
        GatedActivation("silu", limit=swiglu_limit) if swiglu_limit else None
    )
    if isinstance(out_weight, NVFP4Tensor):
        assert quant_config is not None and scales_offset is not None
        data, scales = fused_silu_quantized(
            x,
            expert_start_indices,
            quant_config,
            DType.uint8,
            input_scales=out_weight.input_scale,
            scales_offsets=scales_offset,
            activation=activation,
        )
        return NVFP4Activation(
            data=data.rebind_mapping(x.mapping),
            scales=scales.rebind_mapping(x.mapping),
            input_scale=out_weight.input_scale,
            scales_offset=scales_offset,
        )
    if isinstance(out_weight, FP8BlockTensor):
        block_m, block_k = out_weight.block_size
        input_spec, weight_spec = _fp8_block_specs(
            (block_m, block_k), input_block=(1, block_k)
        )
        data, scales = silu_quantize_fp8(
            x,
            expert_start_indices,
            input_spec,
            weight_spec,
            block_k,
            swiglu_limit,
        )
        return FP8BlockActivation(
            data=data.rebind_mapping(x.mapping),
            scales=scales.rebind_mapping(x.mapping),
        )
    # Exhaustive: a new quantized down-weight must not silently skip its
    # re-quant by taking the plain bf16 SiLU.
    assert out_weight is None or isinstance(out_weight, Tensor)
    return fused_silu(x, expert_start_indices, activation)


def concat_fp8_block(*tensors: FP8BlockTensor, axis: int = 0) -> FP8BlockTensor:
    """Concatenate two or more :class:`FP8BlockTensor`s along ``axis``."""
    if not tensors:
        raise ValueError("concat_fp8_block requires at least one tensor")
    if axis != 0:
        raise ValueError(
            "FP8BlockTensor concat currently only supports axis=0 (row axis)"
        )
    block_size = tensors[0].block_size
    for q in tensors[1:]:
        if q.block_size != block_size:
            raise ValueError(
                "All FP8BlockTensors must have the same block_size to "
                f"concat; got {block_size} and {q.block_size}"
            )

    data = F.concat([q.weight for q in tensors], axis=0)
    weight_scale_inv = F.concat([q.weight_scale_inv for q in tensors], axis=0)
    return FP8BlockTensor(
        weight=data, weight_scale_inv=weight_scale_inv, block_size=block_size
    )


def concat_nvfp4(*tensors: NVFP4Tensor, axis: int = 0) -> NVFP4Tensor:
    """Concatenate two or more :class:`NVFP4Tensor`s along the row axis.

    Concatenates the packed ``weight`` and per-block ``weight_scale`` along
    ``axis`` (the ``N``/row axis). The per-tensor ``weight_scale_2`` and static
    ``input_scale`` are taken from the first tensor: modelopt emits a shared
    global scale for the fused ``gate``/``up`` projections.
    """
    if not tensors:
        raise ValueError("concat_nvfp4 requires at least one tensor")
    if axis != 0:
        raise ValueError(
            "NVFP4Tensor concat currently only supports axis=0 (row axis)"
        )
    block_size = tensors[0].block_size
    for q in tensors[1:]:
        if q.block_size != block_size:
            raise ValueError(
                "All NVFP4Tensors must have the same block_size to concat; "
                f"got {block_size} and {q.block_size}"
            )
    return NVFP4Tensor(
        weight=F.concat([q.weight for q in tensors], axis=0),
        weight_scale=F.concat([q.weight_scale for q in tensors], axis=0),
        weight_scale_2=tensors[0].weight_scale_2,
        input_scale=tensors[0].input_scale,
        block_size=block_size,
    )


def concat_mxfp8(*tensors: MXFP8Tensor, axis: int = 0) -> MXFP8Tensor:
    """Concatenate two or more :class:`MXFP8Tensor`s along the row axis."""
    if not tensors:
        raise ValueError("concat_mxfp8 requires at least one tensor")
    if axis != 0:
        raise ValueError(
            "MXFP8Tensor concat currently only supports axis=0 (row axis)"
        )
    block_size = tensors[0].block_size
    for q in tensors[1:]:
        if q.block_size != block_size:
            raise ValueError(
                "All MXFP8Tensors must have the same block_size to concat; "
                f"got {block_size} and {q.block_size}"
            )
    return MXFP8Tensor(
        weight=F.concat([q.weight for q in tensors], axis=0),
        weight_scale=F.concat([q.weight_scale for q in tensors], axis=0),
        block_size=block_size,
    )
