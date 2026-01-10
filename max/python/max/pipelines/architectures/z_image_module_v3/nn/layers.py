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
from max.driver import Device
from max.dtype import DType
from max.experimental import random
from max.experimental.tensor import Tensor
from max.nn.module_v3.sequential import ModuleList
from typing_extensions import TypeVar

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

    latent_dist: "DiagonalGaussianDistribution"  # Forward reference


# ===----------------------------------------------------------------------=== #
# Utility Functions
# ===----------------------------------------------------------------------=== #


def pad_sequence(
    sequences: Sequence[Tensor],
    batch_first: bool = True,
    padding_value: float = 0.0,
) -> Tensor:
    """Pad a list of variable length tensors with a padding value.

    This is a Modular-native implementation of torch.nn.utils.rnn.pad_sequence.

    Args:
        sequences: List of tensors to pad. All tensors must have the same
            number of dimensions except for the first (sequence) dimension.
        batch_first: If True, output is (batch, seq, *). If False, output
            is (seq, batch, *).
        padding_value: Value for padded elements.

    Returns:
        Padded tensor with shape (batch, max_seq_len, *) if batch_first=True,
        otherwise (max_seq_len, batch, *).
    """
    if not sequences:
        raise ValueError("sequences cannot be empty")

    # Get the max sequence length
    max_len = max(seq.shape[0] for seq in sequences)
    batch_size = len(sequences)

    # Get the trailing dimensions (all dims except the first)
    trailing_dims = sequences[0].shape[1:]
    dtype = sequences[0].dtype

    # Create output shape
    if batch_first:
        out_shape = [batch_size, max_len] + list(trailing_dims)
    else:
        out_shape = [max_len, batch_size] + list(trailing_dims)

    # Create output tensor filled with padding value
    out = Tensor.full(out_shape, value=padding_value, dtype=dtype)

    # Copy each sequence into the output tensor
    # Note: This uses a functional approach via concatenation since
    # MAX tensors are immutable
    padded_sequences = []
    for seq in sequences:
        seq_len = seq.shape[0]
        if seq_len < max_len:
            # Create padding tensor
            pad_shape = [max_len - seq_len] + list(trailing_dims)
            padding = Tensor.full(pad_shape, value=padding_value, dtype=dtype)
            padded_seq = F.concat([seq, padding], axis=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    # Stack all padded sequences
    if batch_first:
        return F.stack(padded_sequences, axis=0)
    else:
        stacked = F.stack(padded_sequences, axis=0)
        # Transpose to (seq, batch, *)
        perm = [1, 0] + list(range(2, len(stacked.shape)))
        return F.transpose(stacked, perm)


def create_attention_mask(
    batch_size: int,
    max_seq_len: int,
    seq_lens: Sequence[int],
    dtype: DType = DType.bool,
) -> Tensor:
    """Create an attention mask for variable-length sequences.

    This function creates a batched attention mask where each row i has
    True values for positions 0 to seq_lens[i]-1 and False elsewhere.

    Uses functional construction compatible with MAX's immutable tensors.

    Args:
        batch_size: Number of sequences in the batch.
        max_seq_len: Maximum sequence length (width of the mask).
        seq_lens: List of actual sequence lengths for each batch item.
        dtype: Data type for the mask (default: bool).

    Returns:
        Tensor of shape (batch_size, max_seq_len) with True for valid positions.
    """
    # Create a range tensor [0, 1, 2, ..., max_seq_len-1]
    positions = Tensor.arange(0, max_seq_len, dtype=DType.int32)

    # Create seq_lens as a column tensor for broadcasting
    seq_lens_tensor = Tensor.constant(seq_lens, dtype=DType.int32).reshape(
        [batch_size, 1]
    )

    # Broadcast comparison: positions < seq_lens for each batch item
    # Result: (batch_size, max_seq_len) boolean mask
    mask = F.unsqueeze(positions, 0) < seq_lens_tensor

    return mask.cast(dtype)


def masked_scatter(
    tensor: Tensor,
    mask: Tensor,
    value: Tensor | float,
) -> Tensor:
    """Return a new tensor with values at masked positions replaced.

    This is a functional equivalent of tensor[mask] = value, compatible with
    MAX's immutable tensors.

    Args:
        tensor: The input tensor to update.
        mask: Boolean mask indicating positions to update.
        value: The value to place at masked positions.

    Returns:
        New tensor with masked positions updated.
    """
    if isinstance(value, (int, float)):
        value = Tensor.full(tensor.shape, value=value, dtype=tensor.dtype)
    elif value.shape != tensor.shape:
        # Broadcast value to tensor shape
        value = F.broadcast_to(value, tensor.shape)

    # Ensure mask broadcasts to tensor
    while mask.rank < tensor.rank:
        mask = F.unsqueeze(mask, -1)

    return F.where(mask, value, tensor)


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
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        # Use provided dtype or default to bfloat16 to match checkpoint weights
        weight_dtype = dtype if dtype is not None else DType.bfloat16

        # Weight shape: (out_channels, in_channels // groups, *kernel_size)
        weight_shape = [out_channels, in_channels // groups, *kernel_size]
        self.weight = random.normal(weight_shape, dtype=weight_dtype)

        if bias:
            self.bias = Tensor.zeros([out_channels], dtype=weight_dtype)
        else:
            self.bias = None

    def __call__(self, input: Tensor) -> Tensor:
        # Input is NCHW, permute to NHWC for conv2d op
        x = input.permute([0, 2, 3, 1])

        # Permute weight to RSCF (H, W, I, O) from (O, I, H, W)
        weight = self.weight.permute((2, 3, 1, 0))

        # Prepare padding (pad_h_before, pad_h_after, pad_w_before, pad_w_after)
        ph, pw = self.padding
        padding = (ph, ph, pw, pw)

        out = F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
            # Default input_layout is NHWC
        )
        # F.conv2d returns NHWC, convert back to NCHW
        return out.permute([0, 3, 1, 2])


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
        input_dtype = hidden_states.dtype
        variance = reduce_mean(
            F.pow(hidden_states.cast(DType.float32), 2), axis=-1
        )
        hidden_states = hidden_states * F.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in (DType.float16, DType.bfloat16):
                hidden_states = hidden_states.cast(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.cast(input_dtype)

        return hidden_states


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
            # Use bfloat16 to match checkpoint weights
            self.weight = Tensor.ones([num_channels], dtype=DType.bfloat16)
            self.bias = Tensor.zeros([num_channels], dtype=DType.bfloat16)
        else:
            self.weight = None
            self.bias = None

    def __call__(self, input: Tensor) -> Tensor:
        # Input: (N, C, H, W)
        N, C, H, W = [int(d) for d in input.shape]

        # Reshape to (N, G, C//G, H, W)
        x = input.reshape([N, self.num_groups, C // self.num_groups, H, W])

        # Calculate mean and var over (C//G, H, W)
        # dims to reduce: 2, 3, 4
        mean = reduce_mean(x, axis=[2, 3, 4])
        var = reduce_var(x, axis=[2, 3, 4])

        x = (x - mean) / F.rsqrt(var + self.eps)

        # Reshape back to (N, C, H, W)
        x = x.reshape([N, C, H, W])

        if self.affine:
            # weight: (C,), bias: (C,)
            # Broadcast to (1, C, 1, 1)
            w = self.weight.reshape([1, C, 1, 1])
            b = self.bias.reshape([1, C, 1, 1])
            x = x * w + b

        return x


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
            # Use bfloat16 to match checkpoint weights
            self.weight = Tensor.ones([normalized_shape], dtype=DType.bfloat16)
            self.bias = Tensor.zeros([normalized_shape], dtype=DType.bfloat16)
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        mean = reduce_mean(x, axis=-1)
        var = reduce_var(x, axis=-1)
        x = (x - mean) / F.rsqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x


class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://huggingface.co/papers/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        self.norm_layer = GroupNorm(
            num_channels=f_channels, num_groups=32, eps=1e-6, affine=True
        )
        self.conv_y = Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv_b = Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, f: Tensor, zq: Tensor) -> Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class SiLU(nn.Module):
    def __call__(self, x: Tensor) -> Tensor:
        return F.silu(x)


class Identity(nn.Module):
    def __call__(self, x: Tensor) -> Tensor:
        return x


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        # TODO: Implement dropout when training support is needed
        return x


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
            self.group_norm = SpatialNorm(
                f_channels=query_dim, zq_channels=spatial_norm_dim
            )
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
                Dropout(p=0.0),  # Dropout is typically 0 for inference
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
            if isinstance(self.group_norm, SpatialNorm):
                hidden_states = self.group_norm(hidden_states, temb)
            else:
                hidden_states = self.group_norm(hidden_states)

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
        query = query.permute([0, 2, 1, 3])
        key = key.permute([0, 2, 1, 3])
        value = value.permute([0, 2, 1, 3])

        # Compute attention scores
        # (B, heads, seq_len, head_dim) @ (B, heads, head_dim, seq_len) -> (B, heads, seq_len, seq_len)
        attn_scores = query @ key.permute([0, 1, 3, 2])
        attn_scores = attn_scores * self.scale

        # Softmax with optional upcasting
        if self.upcast_softmax:
            attn_scores = attn_scores.cast(DType.float32)
        attn_probs = F.softmax(attn_scores, axis=-1)
        attn_probs = attn_probs.cast(value.dtype)

        # Apply attention to values
        # (B, heads, seq_len, seq_len) @ (B, heads, seq_len, head_dim) -> (B, heads, seq_len, head_dim)
        hidden_states = attn_probs @ value

        # Reshape back: (B, heads, seq_len, head_dim) -> (B, seq_len, heads * head_dim)
        hidden_states = hidden_states.permute(
            [0, 2, 1, 3]
        )  # (B, seq_len, heads, head_dim)
        hidden_states = hidden_states.reshape([batch, seq_len, self.inner_dim])

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
