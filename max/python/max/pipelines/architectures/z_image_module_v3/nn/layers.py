# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import max.experimental.functional as F
import max.nn.module_v3 as nn
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph.type import ConvInputLayout, FilterLayout
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu
from max.nn.module_v3.norm.rms_norm import rms_norm
from max.nn.module_v3.sequential import ModuleList
from typing_extensions import TypeVar

flash_attention_gpu = F.functional(_flash_attention_gpu)

# ===----------------------------------------------------------------------=== #
# Output Dataclasses
# ===----------------------------------------------------------------------=== #


@dataclass(eq=False)
class Transformer2DModelOutput:
    """Output class for Transformer2DModel.

    Attributes:
        sample: The output tensor from the transformer.
    """

    sample: Tensor


@dataclass(eq=False)
class AutoencoderKLOutput:
    """Output class for AutoencoderKL encode method.

    Attributes:
        latent_dist: The diagonal gaussian distribution of the latent space.
    """

    latent_dist: "DiagonalGaussianDistribution"


# ===----------------------------------------------------------------------=== #
# Utility Functions
# ===----------------------------------------------------------------------=== #


def reduce_mean(x: Tensor, axis: int | Sequence[int]) -> Tensor:
    """Compute mean along one or more axes, keeping dimensions."""
    if isinstance(axis, int):
        return F.mean(x, axis=axis)

    for ax in sorted(axis, reverse=True):
        x = F.mean(x, axis=ax)
    return x


def reduce_var(x: Tensor, axis: int | Sequence[int]) -> Tensor:
    """Compute variance along one or more axes, keeping dimensions."""
    mean_sq = reduce_mean(x**2, axis)
    sq_mean = reduce_mean(x, axis) ** 2
    return mean_sq - sq_mean


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Device | None = None,
        dtype: DType | None = None,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        ph, pw = padding
        self.padding = (ph, ph, pw, pw)
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = Tensor.zeros(
            [out_channels, in_channels // groups, *kernel_size]
        )

        if bias:
            self.bias = Tensor.zeros([out_channels])
        else:
            self.bias = None

    def __call__(self, hidden_states: Tensor) -> Tensor:
        return F.conv2d(
            hidden_states,
            self.weight,
            self.stride,
            self.dilation,
            self.padding,
            self.groups,
            self.bias,
            input_layout=ConvInputLayout.NCHW,
            filter_layout=FilterLayout.FCRS,
        )


class RMSNorm(nn.Module):
    r"""
    RMS Norm as introduced in https://huggingface.co/papers/1910.07467 by Zhang et al.

    Args:
        dim (`int`): Number of dimensions to use for `weights`. Only effective when `elementwise_affine` is True.
        eps (`float`): Small value to use when calculating the reciprocal of the square-root.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        bias (`bool`, defaults to False): If also training the `bias` param.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = False,
    ):
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = Tensor.ones(dim)
            if bias:
                self.bias = Tensor.zeros(dim)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        if not self.elementwise_affine:
            # Non-affine case: manual implementation
            input_dtype = hidden_states.dtype
            variance = reduce_mean(
                F.pow(hidden_states.cast(DType.float32), 2), axis=-1
            )
            return (hidden_states * F.rsqrt(variance + self.eps)).cast(
                input_dtype
            )

        # Affine case: use fused kernel
        result = rms_norm(hidden_states, self.weight, self.eps)
        if self.bias is not None:
            result = result + self.bias
        return result


class GroupNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Tensor.ones([num_channels])
            self.bias = Tensor.zeros([num_channels])
        else:
            self.weight = None
            self.bias = None

    def __call__(self, hidden_states: Tensor) -> Tensor:
        if self.affine:
            gamma = self.weight.cast(hidden_states.dtype)
            beta = self.bias.cast(hidden_states.dtype)
        else:
            gamma = Tensor.ones(
                [self.num_channels],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            beta = Tensor.zeros(
                [self.num_channels],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        return F.custom(
            "group_norm",
            hidden_states.device,
            [
                hidden_states,
                gamma,
                beta,
                Tensor.constant(
                    [self.eps], dtype=hidden_states.dtype, device=CPU()
                ),
                Tensor.constant(
                    [self.num_groups], dtype=DType.int32, device=CPU()
                ),
            ],
            [hidden_states.type],
        )[0]


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Tensor.ones([normalized_shape])
            self.bias = Tensor.zeros([normalized_shape])
        else:
            self.weight = None
            self.bias = None

    def __call__(self, hidden_states: Tensor) -> Tensor:
        if self.elementwise_affine:
            return F.layer_norm(hidden_states, self.weight, self.bias, self.eps)

        # Manual fallback for non-affine case
        mean = reduce_mean(hidden_states, axis=-1)
        var = reduce_var(hidden_states, axis=-1)
        return (hidden_states - mean) / F.sqrt(var + self.eps)


class SiLU(nn.Module):
    def __call__(self, hidden_states: Tensor) -> Tensor:
        return F.silu(hidden_states)


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, hidden_states: Tensor) -> Tensor:
        # TODO: Implement dropout when training support is needed
        return hidden_states


class Attention(nn.Module):
    """
    A self-attention layer for VAE models.

    Ported from diffusers.models.attention_processor.Attention with simplifications
    for the VAE use case (no cross-attention, no complex processors).

    Args:
        query_dim: Number of channels in the query (and input).
        heads: Number of attention heads.
        dim_head: Dimension of each attention head.
        rescale_output_factor: Factor to rescale the output by dividing.
        eps: Epsilon for normalization layers.
        norm_num_groups: Number of groups for GroupNorm. If None, no GroupNorm is applied.
        spatial_norm_dim: If provided, applies SpatialNorm instead of GroupNorm.
        residual_connection: If True, adds input to the output (residual connection).
        bias: Whether to use bias in linear projections.
        upcast_softmax: If True, upcast softmax computation to float32.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
        norm_num_groups: int | None = None,
        spatial_norm_dim: int | None = None,
        residual_connection: bool = False,
        bias: bool = True,
        upcast_softmax: bool = True,
    ):
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.scale = dim_head**-0.5
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.upcast_softmax = upcast_softmax

        # Normalization
        if spatial_norm_dim is not None:
            raise NotImplementedError("SpatialNorm not implemented yet")
            # self.group_norm = SpatialNorm(
            #     f_channels=query_dim, zq_channels=spatial_norm_dim
            # )
        elif norm_num_groups is not None:
            self.group_norm = GroupNorm(
                num_groups=norm_num_groups,
                num_channels=query_dim,
                eps=eps,
                affine=True,
            )
        else:
            self.group_norm = None

        # Projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)
        # Use ModuleList to match checkpoint's indexing (from nn.Sequential in Diffusers)
        # Parameter names will be to_out.0.weight, to_out.0.bias, etc.
        self.to_out = ModuleList(
            [
                nn.Linear(self.inner_dim, query_dim, bias=bias),
                Dropout(p=0.0),
            ]
        )

    def __call__(
        self,
        hidden_states: Tensor,
        temb: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for self-attention.

        Args:
            hidden_states: Input tensor of shape (batch, channels, height, width).
            temb: Optional temporal/conditioning embedding for SpatialNorm.

        Returns:
            Output tensor of the same shape as input.
        """
        residual = hidden_states

        # Get dimensions
        # Input: (batch, channels, height, width)
        batch, channels, height, width = [int(d) for d in hidden_states.shape]

        # Apply normalization
        if self.group_norm is not None:
            if isinstance(self.group_norm, GroupNorm):
                hidden_states = self.group_norm(hidden_states)
            else:
                hidden_states = self.group_norm(hidden_states, temb)

        # Reshape from (B, C, H, W) to (B, H*W, C) for attention
        hidden_states = hidden_states.permute([0, 2, 3, 1])  # (B, H, W, C)
        hidden_states = hidden_states.reshape([batch, height * width, channels])

        # Project to Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape for multi-head attention: (B, seq_len, heads, head_dim)
        seq_len = height * width
        query = query.reshape([batch, seq_len, self.heads, self.head_dim])
        key = key.reshape([batch, seq_len, self.heads, self.head_dim])
        value = value.reshape([batch, seq_len, self.heads, self.head_dim])

        # Transpose to (B, heads, seq_len, head_dim)
        # query = query.permute([0, 2, 1, 3])
        # key = key.permute([0, 2, 1, 3])
        # value = value.permute([0, 2, 1, 3])

        # Compute attention scores
        # (B, heads, seq_len, head_dim) @ (B, heads, head_dim, seq_len) -> (B, heads, seq_len, seq_len)
        # attn_scores = query @ key.permute([0, 1, 3, 2])
        # attn_scores = attn_scores * self.scale

        # Softmax with optional upcasting
        # if self.upcast_softmax:
        #     attn_scores = attn_scores.cast(DType.float32)
        # attn_probs = F.softmax(attn_scores, axis=-1)
        # attn_probs = attn_probs.cast(value.dtype)

        # Apply attention to values
        # (B, heads, seq_len, seq_len) @ (B, heads, seq_len, head_dim) -> (B, heads, seq_len, head_dim)
        # hidden_states = attn_probs @ value

        attn_out = flash_attention_gpu(
            query,
            key,
            value,
            MHAMaskVariant.NULL_MASK,
            self.scale,
        )

        # Reshape back: (B, heads, seq_len, head_dim) -> (B, seq_len, heads * head_dim)
        # hidden_states = hidden_states.permute(
        #     [0, 2, 1, 3]
        # )  # (B, seq_len, heads, head_dim)
        hidden_states = attn_out.reshape([batch, seq_len, self.inner_dim])

        # Output projection (Linear + Dropout from to_out list)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        # Reshape back to image format: (B, H*W, C) -> (B, C, H, W)
        hidden_states = hidden_states.reshape([batch, height, width, channels])
        hidden_states = hidden_states.permute([0, 3, 1, 2])

        # Rescale output
        if self.rescale_output_factor != 1.0:
            hidden_states = hidden_states / self.rescale_output_factor

        # Residual connection
        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


T = TypeVar("T")


class ModuleDict(dict[str, T], nn.Module):
    """A ``Module`` subclass which is locally a dict container.

    For example:

    .. code-block:: python

        from max.nn.module_v3 import Linear, ModuleDict

        model = ModuleDict({
            "layer1": Linear(5, 10),
            "layer2": Linear(10, 5),
        })
    """

    @property
    def children(self) -> Iterable[tuple[str, nn.Module]]:
        """Iterates over the direct child modules of the ``Module``.

        Yields:
            ``(name, module)`` pairs, where ``name`` is the attribute name of
            the child on the module.
        """
        yield from self.items()

    def __rich_repr__(self):
        """Omits the path for children in the repr."""
        yield from self.items()

    # C3 linearization resolves dict.__repr__ before nn.Module.__repr__.
    # This explicitly overrides and tells the class to use nn.Module.__repr__.
    __repr__ = nn.Module.__repr__


ACT2CLS = {
    "swish": SiLU,
    "silu": SiLU,
    # "mish": nn.Mish,
    # "gelu": nn.GELU,
    # "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(
            f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}"
        )
