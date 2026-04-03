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

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.activation import activation_function_from_name
from max.nn.layer import Module
from max.nn.linear import Linear


def get_timestep_embedding(
    timesteps: TensorValue,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> TensorValue:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * ops.range(
        0, half_dim, dtype=DType.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = ops.exp(exponent)
    timesteps_expanded = ops.cast(ops.unsqueeze(timesteps, 1), DType.float32)
    emb_expanded = ops.unsqueeze(emb, 0)
    emb = scale * timesteps_expanded * emb_expanded
    emb = ops.concat([ops.sin(emb), ops.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = ops.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = ops.pad(emb, (0, 0, 0, 1))
    return emb


def apply_rotary_emb(
    x: TensorValue,
    freqs_cis: tuple[TensorValue, TensorValue],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> TensorValue:
    if not use_real:
        raise NotImplementedError("Only use_real=True is supported")

    cos, sin = freqs_cis
    if sequence_dim == 2:
        cos = ops.unsqueeze(ops.unsqueeze(cos, 0), 0)
        sin = ops.unsqueeze(ops.unsqueeze(sin, 0), 0)
    elif sequence_dim == 1:
        cos = ops.unsqueeze(ops.unsqueeze(cos, 0), 2)
        sin = ops.unsqueeze(ops.unsqueeze(sin, 0), 2)
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    input_dtype = x.dtype

    if use_real_unbind_dim == -1:
        x_shape: list[Any] = list(x.shape)
        new_shape: list[Any] = x_shape[:-1] + [x_shape[-1] // 2, 2]
        x_reshaped = ops.reshape(x, new_shape)
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]
        x_rotated_stacked = ops.stack([-x_imag, x_real], axis=-1)
        x_rotated = ops.reshape(x_rotated_stacked, x_shape)
    elif use_real_unbind_dim == -2:
        x_shape = list(x.shape)
        new_shape = x_shape[:-1] + [2, x_shape[-1] // 2]
        x_reshaped = ops.reshape(x, new_shape)
        x_real = x_reshaped[..., 0, :]
        x_imag = x_reshaped[..., 1, :]
        x_rotated = ops.concat([-x_imag, x_real], axis=-1)
    else:
        raise ValueError(
            f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
        )

    out = ops.cast(x, DType.float32) * ops.cast(cos, DType.float32) + ops.cast(
        x_rotated, DType.float32
    ) * ops.cast(sin, DType.float32)
    return ops.cast(out, input_dtype)


def _get_1d_rotary_pos_embed_np(
    dim: int,
    pos: np.ndarray,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D rotary position embeddings (numpy, for eager RoPE)."""
    freq_exponent = np.arange(0, dim, 2, dtype=np.float64) / dim
    freqs = 1.0 / (theta**freq_exponent)
    angles = np.outer(pos.astype(np.float64), freqs)
    cos_emb = np.cos(angles).astype(np.float32)
    sin_emb = np.sin(angles).astype(np.float32)
    cos_emb = np.repeat(cos_emb, 2, axis=1)
    sin_emb = np.repeat(sin_emb, 2, axis=1)
    return cos_emb, sin_emb


@lru_cache(maxsize=8)
def compute_wan_rope_cached(
    num_frames: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int],
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3D RoPE cos/sin arrays for Wan transformer (cached by resolution)."""
    p_t, p_h, p_w = patch_size
    ppf = num_frames // p_t
    pph = height // p_h
    ppw = width // p_w

    d_h = (head_dim // 3 // 2) * 2
    d_w = d_h
    d_t = head_dim - d_h - d_w

    cos_t, sin_t = _get_1d_rotary_pos_embed_np(d_t, np.arange(ppf), theta)
    cos_h, sin_h = _get_1d_rotary_pos_embed_np(d_h, np.arange(pph), theta)
    cos_w, sin_w = _get_1d_rotary_pos_embed_np(d_w, np.arange(ppw), theta)

    cos_t = np.broadcast_to(cos_t[:, None, None, :], (ppf, pph, ppw, d_t))
    sin_t = np.broadcast_to(sin_t[:, None, None, :], (ppf, pph, ppw, d_t))
    cos_h = np.broadcast_to(cos_h[None, :, None, :], (ppf, pph, ppw, d_h))
    sin_h = np.broadcast_to(sin_h[None, :, None, :], (ppf, pph, ppw, d_h))
    cos_w = np.broadcast_to(cos_w[None, None, :, :], (ppf, pph, ppw, d_w))
    sin_w = np.broadcast_to(sin_w[None, None, :, :], (ppf, pph, ppw, d_w))

    rope_cos = np.concatenate([cos_t, cos_h, cos_w], axis=-1)
    rope_sin = np.concatenate([sin_t, sin_h, sin_w], axis=-1)

    seq_len = ppf * pph * ppw
    rope_cos = np.ascontiguousarray(rope_cos.reshape(seq_len, head_dim))
    rope_sin = np.ascontiguousarray(rope_sin.reshape(seq_len, head_dim))
    return rope_cos, rope_sin


class Timesteps(Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: TensorValue) -> TensorValue:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ):
        super().__init__()
        self.linear_1 = Linear(
            in_dim=in_channels,
            out_dim=time_embed_dim,
            dtype=dtype,
            device=device,
            has_bias=sample_proj_bias,
        )
        self.cond_proj: Linear | None
        if cond_proj_dim is not None:
            self.cond_proj = Linear(
                in_dim=cond_proj_dim,
                out_dim=in_channels,
                dtype=dtype,
                device=device,
                has_bias=False,
            )
        else:
            self.cond_proj = None
        self.act = activation_function_from_name(act_fn)
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = Linear(
            in_dim=time_embed_dim,
            out_dim=time_embed_dim_out,
            dtype=dtype,
            device=device,
            has_bias=sample_proj_bias,
        )
        self.post_act: Callable[[TensorValue], TensorValue] | None
        if post_act_fn is not None:
            self.post_act = activation_function_from_name(post_act_fn)
        else:
            self.post_act = None

    def __call__(self, sample: TensorValue) -> TensorValue:
        if self.cond_proj is not None:
            sample = sample + self.cond_proj(sample)
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
