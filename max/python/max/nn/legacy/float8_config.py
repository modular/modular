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

"""Float8 configuration data structures for models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from max.dtype import DType


class Float8ScaleGranularity(Enum):
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


class Float8ScaleOrigin(Enum):
    """Specifies whether the quantization scale is determined statically or dynamically."""

    STATIC = "static"
    """Scales are pre-computed and loaded with the model weights."""

    DYNAMIC = "dynamic"
    """Scales are computed at runtime based on the input data."""


@dataclass
class Float8WeightScaleSpec:
    """Specifies how weights are scaled for float8 quantization."""

    granularity: Float8ScaleGranularity
    """The :obj:`Float8ScaleGranularity` of the weight scale factor application."""

    dtype: DType
    """The :obj:`DType` of the weight scale factor(s)."""

    block_size: tuple[int, int] | None = None
    """The :obj:`tuple[int, int]` of the block size for block-wise scaling."""

    def __post_init__(self):
        if self.granularity == Float8ScaleGranularity.BLOCK:
            if self.block_size is None:
                raise ValueError(
                    "block_size must be specified for block-wise scaling"
                )
            if len(self.block_size) != 2:
                raise ValueError("block_size must be a tuple of two integers")

    @property
    def is_tensor(self) -> bool:
        """Whether the weight scale granularity is per-tensor."""
        return self.granularity == Float8ScaleGranularity.TENSOR

    @property
    def is_rowwise(self) -> bool:
        """Whether the weight scale granularity is row-wise."""
        return self.granularity == Float8ScaleGranularity.ROWWISE

    @property
    def is_colwise(self) -> bool:
        """Whether the weight scale granularity is column-wise."""
        return self.granularity == Float8ScaleGranularity.COLWISE

    @property
    def is_block(self) -> bool:
        """Whether the weight scale granularity is block-wise."""
        return self.granularity == Float8ScaleGranularity.BLOCK


@dataclass
class Float8InputScaleSpec:
    """Specifies how input activations are scaled for float8 quantization."""

    granularity: Float8ScaleGranularity
    """The :obj:`Float8ScaleGranularity` of the input scale factor application."""

    origin: Float8ScaleOrigin
    """The :obj:`Float8ScaleOrigin` (static or dynamic) of the input scale factor."""

    dtype: DType
    """The :obj:`DType` of the input scale factor(s)."""

    activation_scale_ub: float | None = None
    """An optional upper bound for dynamic activation scaling."""

    block_size: tuple[int, int] | None = None
    """The :obj:`tuple[int, int]` of the block size for block-wise scaling."""

    def __post_init__(self):
        if self.granularity == Float8ScaleGranularity.BLOCK:
            if self.block_size is None:
                raise ValueError(
                    "block_size must be specified for block-wise scaling"
                )
            if len(self.block_size) != 2:
                raise ValueError("block_size must be a tuple of two integers")

    @property
    def is_tensor(self) -> bool:
        """Whether the input scale granularity is per-tensor."""
        return self.granularity == Float8ScaleGranularity.TENSOR

    @property
    def is_rowwise(self) -> bool:
        """Whether the input scale granularity is row-wise."""
        return self.granularity == Float8ScaleGranularity.ROWWISE

    @property
    def is_colwise(self) -> bool:
        """Whether the input scale granularity is column-wise."""
        return self.granularity == Float8ScaleGranularity.COLWISE

    @property
    def is_block(self) -> bool:
        """Whether the input scale granularity is block-wise."""
        return self.granularity == Float8ScaleGranularity.BLOCK


@dataclass
class Float8Config:
    """Configures float8 quantization settings for a layer or model section."""

    input_scale: Float8InputScaleSpec
    """:obj:`Float8InputScaleSpec` for input activation scaling."""

    weight_scale: Float8WeightScaleSpec
    """:obj:`Float8WeightScaleSpec` for weight scaling."""

    mlp_in_float8: set[int]
    """Set of layer indices with MLPs in float8.

    MLPs are considered to be either "all quantized" or all not quantized per
    layer.
    So either all of gate proj, down proj, and up proj are float8, or all bfloat16.
    """

    attn_qkv_in_float8: set[int]
    """Set of layer indices with attention QKV projections in float8.

    QKV projections are considered to be either "all quantized" or all not
    quantized per layer.
    So either all of {q,k,v,o}_proj are float8, or all bfloat16.
    """

    embedding_output_dtype: DType | None = None
    """The :obj:`DType` of the output from the embedding layer."""

    bias_dtype: DType | None = None
    """The :obj:`DType` of bias weights."""

    quant_method: str | None = None
    """The quantization method used (e.g., "fbgemm_fp8")."""

    quant_algo: str | None = None
    """Additional differentiator within same quant_method e.g. modelopt NVFP4 vs FP8"""

    @property
    def scales_granularity_mnk(self) -> tuple[int, int, int]:
        """Returns the weight and input scale granularities on M, N and K axis."""
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
        """Returns ``True`` if this input scale is static."""
        return self.input_scale.origin == Float8ScaleOrigin.STATIC

    @property
    def is_dynamic(self) -> bool:
        """Returns ``True`` if this input scale is dynamic."""
        return self.input_scale.origin == Float8ScaleOrigin.DYNAMIC

    @property
    def is_nvfp4(self) -> bool:
        """Returns ``True`` if this config represents modelopt NVFP4."""
        return self.quant_method == "modelopt" and self.quant_algo == "NVFP4"


def nvfp4_packed_k(in_dim: int, float8_config: Float8Config | None) -> int:
    """Returns packed K dimension for NVFP4 weights, else returns in_dim."""
    return (
        in_dim // 2
        if float8_config is not None and float8_config.is_nvfp4
        else in_dim
    )
