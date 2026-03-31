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

"""Scaled quantization configuration data structures for models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from max.dtype import DType
from max.graph import DeviceRef, Dim, DimLike, Shape, TensorType


class ScaleGranularity(Enum):
    """Specifies the granularity of the quantization scale factor.

    Determines whether a scale factor applies per-tensor, per-row (often for
    weights), per-column, or per-block within a tensor.
    """

    TENSOR = "tensor"
    """Per-tensor scaling."""

    ROWWISE = "rowwise"
    """Per-row scaling."""

    COLWISE = "colwise"
    """Per-column scaling."""

    BLOCK = "block"
    """Per-block scaling."""

    def __str__(self):
        return self.value


class ScaleOrigin(Enum):
    """Specifies whether the quantization scale is determined statically or dynamically."""

    STATIC = "static"
    """Scales are pre-computed and loaded with the model weights."""

    DYNAMIC = "dynamic"
    """Scales are computed at runtime based on the input data."""


class QuantFormat(Enum):
    """Identifies the quantization format of a model checkpoint."""

    COMPRESSED_TENSORS_FP8 = "compressed-tensors-fp8"
    FBGEMM_FP8 = "fbgemm-fp8"
    BLOCKSCALED_FP8 = "blockscaled-fp8"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"


@dataclass
class WeightScaleSpec:
    """Specifies how weights are scaled for scaled quantization."""

    granularity: ScaleGranularity
    """The :class:`~max.nn.quant_config.ScaleGranularity` of the weight scale factor application."""

    dtype: DType
    """The :class:`~max.dtype.DType` of the weight scale factor(s)."""

    block_size: tuple[int, int] | None = None
    """The :obj:`tuple[int, int]` of the block size for block-wise scaling."""

    def __post_init__(self):
        if self.granularity == ScaleGranularity.BLOCK:
            if self.block_size is None:
                raise ValueError(
                    "block_size must be specified for block-wise scaling"
                )
            if len(self.block_size) != 2:
                raise ValueError("block_size must be a tuple of two integers")

    @property
    def is_tensor(self) -> bool:
        """Whether the weight scale granularity is per-tensor."""
        return self.granularity == ScaleGranularity.TENSOR

    @property
    def is_rowwise(self) -> bool:
        """Whether the weight scale granularity is row-wise."""
        return self.granularity == ScaleGranularity.ROWWISE

    @property
    def is_colwise(self) -> bool:
        """Whether the weight scale granularity is column-wise."""
        return self.granularity == ScaleGranularity.COLWISE

    @property
    def is_block(self) -> bool:
        """Whether the weight scale granularity is block-wise."""
        return self.granularity == ScaleGranularity.BLOCK


@dataclass
class InputScaleSpec:
    """Specifies how input activations are scaled for scaled quantization."""

    granularity: ScaleGranularity
    """The :class:`~max.nn.quant_config.ScaleGranularity` of the input scale factor application."""

    origin: ScaleOrigin
    """The :class:`~max.nn.quant_config.ScaleOrigin` (static or dynamic) of the input scale factor."""

    dtype: DType
    """The :class:`~max.dtype.DType` of the input scale factor(s)."""

    activation_scale_ub: float | None = None
    """An optional upper bound for dynamic activation scaling."""

    block_size: tuple[int, int] | None = None
    """The :obj:`tuple[int, int]` of the block size for block-wise scaling."""

    def __post_init__(self):
        if self.granularity == ScaleGranularity.BLOCK:
            if self.block_size is None:
                raise ValueError(
                    "block_size must be specified for block-wise scaling"
                )
            if len(self.block_size) != 2:
                raise ValueError("block_size must be a tuple of two integers")

    @property
    def is_tensor(self) -> bool:
        """Whether the input scale granularity is per-tensor."""
        return self.granularity == ScaleGranularity.TENSOR

    @property
    def is_rowwise(self) -> bool:
        """Whether the input scale granularity is row-wise."""
        return self.granularity == ScaleGranularity.ROWWISE

    @property
    def is_colwise(self) -> bool:
        """Whether the input scale granularity is column-wise."""
        return self.granularity == ScaleGranularity.COLWISE

    @property
    def is_block(self) -> bool:
        """Whether the input scale granularity is block-wise."""
        return self.granularity == ScaleGranularity.BLOCK


@dataclass
class QuantConfig:
    """Configures scaled quantization settings for a layer or model section."""

    input_scale: InputScaleSpec
    """:class:`~max.nn.quant_config.InputScaleSpec` for input activation scaling."""

    weight_scale: WeightScaleSpec
    """:class:`~max.nn.quant_config.WeightScaleSpec` for weight scaling."""

    mlp_quantized_layers: set[int]
    """Set of layer indices with quantized MLPs.

    MLPs are considered to be either "all quantized" or all not quantized per
    layer.
    So either all of gate proj, down proj, and up proj are quantized, or all bfloat16.
    """

    attn_quantized_layers: set[int]
    """Set of layer indices with quantized attention QKV projections.

    QKV projections are considered to be either "all quantized" or all not
    quantized per layer.
    So either all of {q,k,v,o}_proj are quantized, or all bfloat16.
    """

    format: QuantFormat
    """The :class:`~max.nn.quant_config.QuantFormat` identifying the quantization format."""

    embedding_output_dtype: DType | None = None
    """The :class:`~max.dtype.DType` of the output from the embedding layer."""

    bias_dtype: DType | None = None
    """The :class:`~max.dtype.DType` of bias weights."""

    can_use_fused_mlp: bool = False
    """Whether the quantization scales can be used with fused MLP operations."""

    @property
    def scales_granularity_mnk(self) -> tuple[int, int, int]:
        """The weight and input scale granularities on the M, N, and K axes."""
        m_input_granularity: int
        k_input_granularity: int
        if self.input_scale.is_block:
            input_block_size = self.input_scale.block_size
            assert input_block_size is not None
            m_input_granularity = input_block_size[0]
            k_input_granularity = input_block_size[1]
        elif self.input_scale.is_colwise:
            m_input_granularity = 1
            k_input_granularity = -1  # one scale shared by one token
        elif self.input_scale.is_tensor:
            m_input_granularity = -1
            k_input_granularity = -1
        else:
            raise ValueError("unsupported input scale granularity")

        n_weight_granularity: int
        k_weight_granularity: int
        if self.weight_scale.is_block:
            weight_block_size = self.weight_scale.block_size
            assert weight_block_size is not None
            n_weight_granularity = weight_block_size[0]
            k_weight_granularity = weight_block_size[1]
        elif self.weight_scale.is_rowwise:
            n_weight_granularity = 1
            k_weight_granularity = -1  # one scale shared by one row
        elif self.weight_scale.is_tensor:
            n_weight_granularity = -1
            k_weight_granularity = -1
        else:
            raise ValueError("unsupported weight scale granularity")

        assert k_input_granularity == k_weight_granularity, (
            "k_input_granularity and k_weight_granularity must be the same"
        )

        return (m_input_granularity, n_weight_granularity, k_input_granularity)

    @property
    def is_static(self) -> bool:
        """``True`` if this input scale is static."""
        return self.input_scale.origin == ScaleOrigin.STATIC

    @property
    def is_dynamic(self) -> bool:
        """``True`` if this input scale is dynamic."""
        return self.input_scale.origin == ScaleOrigin.DYNAMIC

    @property
    def is_nvfp4(self) -> bool:
        """``True`` if this config represents modelopt NVFP4."""
        return self.format == QuantFormat.NVFP4

    @property
    def is_mxfp4(self) -> bool:
        """Returns ``True`` if this config represents MXFP4 quantization."""
        return self.format == QuantFormat.MXFP4

    @property
    def is_fp4(self) -> bool:
        """``True`` if this config represents any FP4 variant (NVFP4 or MXFP4)."""
        return self.is_nvfp4 or self.is_mxfp4

    def quantized_scales_type(
        self, quantized_shape: Shape, device_ref: DeviceRef
    ) -> TensorType:
        """The :class:`~max.graph.TensorType` of the scales tensor after dynamic quantization."""
        if self.is_nvfp4:
            return _nvfp4_scales_type(quantized_shape, device_ref)
        elif (
            self.input_scale.block_size is not None
            and self.input_scale.block_size == (1, 128)
        ):
            return _blockwise_fp8_scales_type(quantized_shape, device_ref)
        else:
            raise ValueError("Can not determine the quantized scales type")


def fp4_packed_k(in_dim: int, quant_config: QuantConfig | None) -> int:
    """Returns packed K dimension for FP4 weights, else returns in_dim."""
    return (
        in_dim // 2
        if quant_config is not None and quant_config.is_fp4
        else in_dim
    )


def ceildiv(n: DimLike, d: DimLike) -> Dim:
    """Returns the ceiling division of ``n`` by ``d``.

    Args:
        n: The numerator as a ``DimLike`` value.
        d: The denominator as a ``DimLike`` value.

    Returns:
        A :class:`~max.graph.Dim` equal to ceil(n / d).
    """
    return (Dim(n) + Dim(d) - Dim(1)) // Dim(d)


def _blockwise_fp8_scales_type(
    quantized_shape: Shape, device_ref: DeviceRef
) -> TensorType:
    """Returns the TensorType of the blockwise FP8 scales tensor."""
    # Blockwise FP8 quantization uses a transposed layout for the scales tensor.
    return TensorType(
        dtype=DType.float32,
        shape=(ceildiv(quantized_shape[1], 128), quantized_shape[0]),
        device=device_ref,
    )


def _nvfp4_scales_type(
    quantized_shape: Shape, device_ref: DeviceRef
) -> TensorType:
    """Returns the TensorType of the NVFP4 scales tensor."""
    # Nvidia NVFP4 format requires the scales tensor to be in a 128x4 tiled
    # layout. The follow constant needs to be in sync with those defined in
    # `max/kernels/src/linalg/fp4_utils.mojo`.
    #
    # References:
    # - https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
    NVFP4_SF_VECTOR_SIZE = 16

    scales_dim_0 = ceildiv(quantized_shape[0], SF_MN_GROUP_SIZE)
    scales_dim_1 = ceildiv(quantized_shape[1], NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    return TensorType(
        dtype=DType.float8_e4m3fn,
        shape=(
            scales_dim_0,
            scales_dim_1,
            SF_ATOM_M[0],
            SF_ATOM_M[1],
            SF_ATOM_K,
        ),
        device=device_ref,
    )
