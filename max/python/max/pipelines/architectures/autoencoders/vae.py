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

from itertools import pairwise

from max.driver import accelerator_api
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, Weight, ops
from max.graph.type import FilterLayout
from max.nn.layer import LayerList, Module

from .model_config import AutoencoderKLWanConfig

CACHE_T = 2
WAN_DECODER_CACHE_SLOTS = 32
WAN_ENCODER_CHUNK_SIZE = 4  # Frames per encoder chunk (matching diffusers)


def _use_nvidia_fcrs_conv3d(device: DeviceRef | None) -> bool:
    return (
        device is not None and device.is_gpu() and accelerator_api() == "cuda"
    )


def _zero_cache_for(x: TensorValue) -> TensorValue:
    """Create a zero cache tensor shaped for a causal conv input."""
    shape: list[int | str | Dim] = [
        x.shape[0],
        x.shape[1],
        CACHE_T,
        x.shape[3],
        x.shape[4],
    ]
    return ops.constant(0.0, dtype=x.dtype, device=x.device).broadcast_to(shape)


class RMSNorm(Module):
    """RMS norm used by Wan VAE blocks."""

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = False,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.channel_first = channel_first

        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        shape = [dim, *broadcastable_dims] if channel_first else [dim]
        dev_ref = device if device is not None else DeviceRef.CPU()
        self.gamma = Weight(
            "gamma",
            dtype or DType.float32,
            shape,
            dev_ref,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        axis = 1 if self.channel_first else x.rank - 1
        rms = ops.mean(x * x, axis=axis)
        inv = ops.rsqrt(rms + 1e-12)
        gamma = ops.transfer_to(self.gamma, x.device)
        return x * inv * gamma


class CausalConv3d(Module):
    """3D causal convolution for Wan VAE.

    Temporal causality is implemented via asymmetric padding: the front
    (temporal) dimension is padded on the left only, which the conv3d
    padding parameter supports directly.

    Input is permuted from NCDHW to NDHWC before conv, and back after.
    On NVIDIA GPUs, weights stay in PyTorch FCQRS layout to use the cuDNN
    3D conv dispatch path. Other backends use MAX's native QRSCF layout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        has_bias: bool = True,
        prefer_nvidia_fcrs: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            pad_t = pad_h = pad_w = padding
        else:
            pad_t, pad_h, pad_w = padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._stride = stride
        # Causal: pad only the front of the temporal axis (left=2*pad_t, right=0).
        self._padding = (2 * pad_t, 0, pad_h, pad_h, pad_w, pad_w)

        dev_ref = device if device is not None else DeviceRef.CPU()
        dt = dtype or DType.float32
        d, h, w = kernel_size
        self._use_nvidia_fcrs = prefer_nvidia_fcrs and _use_nvidia_fcrs_conv3d(
            dev_ref
        )
        filter_shape = (
            [out_channels, in_channels, d, h, w]
            if self._use_nvidia_fcrs
            else [d, h, w, in_channels, out_channels]
        )
        self.filter = Weight("weight", dt, filter_shape, dev_ref)
        self._has_bias = has_bias
        if has_bias:
            self.bias = Weight("bias", dt, [out_channels], dev_ref)

    def __call__(self, x: TensorValue) -> TensorValue:
        # NCDHW -> NDHWC
        x_ndhwc = ops.permute(x, [0, 2, 3, 4, 1])
        out = ops.conv3d(
            x_ndhwc,
            self.filter,
            stride=self._stride,
            padding=self._padding,
            filter_layout=(
                FilterLayout.FCRS
                if self._use_nvidia_fcrs
                else FilterLayout.QRSCF
            ),
        )
        # NDHWC -> NCDHW
        out = ops.permute(out, [0, 4, 1, 2, 3])
        if self._has_bias:
            bias_5d = ops.reshape(self.bias, [1, self.out_channels, 1, 1, 1])
            out = out + bias_5d
        return out


class CausalConv3dCached(Module):
    """3D causal convolution with explicit cache tensor I/O.

    Handles temporal causal padding separately via concat/pad before
    calling the conv, while spatial padding is handled by conv3d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            pad_t = pad_h = pad_w = padding
        else:
            pad_t, pad_h, pad_w = padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._stride = stride
        # Temporal causal padding: left=2*pad_t, right=0
        self._temporal_pad_left = 2 * pad_t
        # Let conv3d handle spatial padding. Temporal padding = 0 here.
        self._padding = (0, 0, pad_h, pad_h, pad_w, pad_w)

        dev_ref = device if device is not None else DeviceRef.CPU()
        dt = dtype or DType.float32
        d, h, w = kernel_size
        self._use_nvidia_fcrs = _use_nvidia_fcrs_conv3d(dev_ref)
        filter_shape = (
            [out_channels, in_channels, d, h, w]
            if self._use_nvidia_fcrs
            else [d, h, w, in_channels, out_channels]
        )
        self.filter = Weight("weight", dt, filter_shape, dev_ref)
        self._has_bias = has_bias
        if has_bias:
            self.bias = Weight("bias", dt, [out_channels], dev_ref)

    def _apply_temporal_pad(self, x: TensorValue, pad_left: int) -> TensorValue:
        """Zero-pad the temporal dimension (axis=2) on the left only."""
        if pad_left <= 0:
            return x
        # ops.pad expects 2*rank values: [d0_before, d0_after, d1_before, d1_after, ...]
        # For 5D [B, C, T, H, W]: pad only dim 2 (T) on the left.
        pad_vals = [0, 0, 0, 0, pad_left, 0, 0, 0, 0, 0]
        return ops.pad(x, pad_vals)

    def _forward_conv(self, x: TensorValue) -> TensorValue:
        # NCDHW -> NDHWC
        x_ndhwc = ops.permute(x, [0, 2, 3, 4, 1])
        out = ops.conv3d(
            x_ndhwc,
            self.filter,
            stride=self._stride,
            padding=self._padding,
            filter_layout=(
                FilterLayout.FCRS
                if self._use_nvidia_fcrs
                else FilterLayout.QRSCF
            ),
        )
        # NDHWC -> NCDHW
        out = ops.permute(out, [0, 4, 1, 2, 3])
        if self._has_bias:
            bias_5d = ops.reshape(self.bias, [1, self.out_channels, 1, 1, 1])
            out = out + bias_5d
        return out

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self._apply_temporal_pad(x, self._temporal_pad_left)
        return self._forward_conv(x)

    def forward_cached(
        self, x: TensorValue, cache_in: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        # Rebind cache spatial dims to match x so concat sees matching dims.
        cache_in = ops.rebind(
            cache_in,
            shape=[
                cache_in.shape[0],
                cache_in.shape[1],
                cache_in.shape[2],
                x.shape[3],
                x.shape[4],
            ],
        )
        x = ops.concat([cache_in, x], axis=2)
        cache_out = x[:, :, -CACHE_T:, :, :]
        effective_pad = max(self._temporal_pad_left - CACHE_T, 0)
        x = self._apply_temporal_pad(x, effective_pad)
        return self._forward_conv(x), cache_out


class Conv2dPermuted(Module):
    """2D convolution with NCHW input and FCRS weights (permute=True equivalent).

    Input is permuted from NCHW to NHWC before conv, and back after.
    Weights stay in FCRS (PyTorch) layout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(stride, int):
            self._stride = (stride, stride)
        else:
            self._stride = stride
        if isinstance(padding, int):
            self._padding = (padding, padding, padding, padding)
        else:
            self._padding = padding

        dev_ref = device if device is not None else DeviceRef.CPU()
        dt = dtype or DType.float32
        self.filter = Weight(
            "weight",
            dt,
            [out_channels, in_channels, kernel_size, kernel_size],
            dev_ref,
        )
        self._has_bias = has_bias
        if has_bias:
            self.bias = Weight("bias", dt, [out_channels], dev_ref)

    def __call__(self, x: TensorValue) -> TensorValue:
        # NCHW -> NHWC
        x_nhwc = ops.permute(x, [0, 2, 3, 1])
        out = ops.conv2d(
            x_nhwc,
            self.filter,
            stride=self._stride,
            padding=self._padding,
            filter_layout=FilterLayout.FCRS,
        )
        # NHWC -> NCHW
        out = ops.permute(out, [0, 3, 1, 2])
        if self._has_bias:
            bias_4d = ops.reshape(self.bias, [1, self.out_channels, 1, 1])
            out = out + bias_4d
        return out


class Conv2d(Module):
    """2D convolution with NHWC input and RSCF weights (permute=False equivalent).

    Input is already in NHWC layout. Weights are in RSCF layout
    [H, W, in_channels, out_channels].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(stride, int):
            self._stride = (stride, stride)
        else:
            self._stride = stride
        if isinstance(padding, int):
            self._padding = (padding, padding, padding, padding)
        else:
            self._padding = padding

        dev_ref = device if device is not None else DeviceRef.CPU()
        dt = dtype or DType.float32
        self.filter = Weight(
            "weight",
            dt,
            [kernel_size, kernel_size, in_channels, out_channels],
            dev_ref,
        )
        self._has_bias = has_bias
        if has_bias:
            self.bias = Weight("bias", dt, [out_channels], dev_ref)

    def __call__(self, x: TensorValue) -> TensorValue:
        out = ops.conv2d(
            x,
            self.filter,
            stride=self._stride,
            padding=self._padding,
            filter_layout=FilterLayout.RSCF,
            bias=self.bias if self._has_bias else None,
        )
        return out


class ResidualBlock(Module):
    """Residual block used in Wan VAE decoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        prefer_nvidia_fcrs: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            in_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv1 = CausalConv3d(
            in_dim,
            out_dim,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
            prefer_nvidia_fcrs=prefer_nvidia_fcrs,
        )
        self.norm2 = RMSNorm(
            out_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv2 = CausalConv3d(
            out_dim,
            out_dim,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
            prefer_nvidia_fcrs=prefer_nvidia_fcrs,
        )
        self.conv_shortcut = (
            CausalConv3d(
                in_dim,
                out_dim,
                1,
                padding=0,
                dtype=dtype,
                device=device,
                has_bias=True,
                prefer_nvidia_fcrs=prefer_nvidia_fcrs,
            )
            if in_dim != out_dim
            else None
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        residual = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )
        x = ops.silu(self.norm1(x))
        x = self.conv1(x)
        x = ops.silu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class AttentionBlock(Module):
    """Per-frame windowed self-attention used in Wan decoder mid block.

    Uses window attention instead of full (H*W)^2 attention to avoid OOM
    at high resolutions. The spatial dimensions are partitioned into
    non-overlapping windows of size ws*ws, and attention is computed
    independently per window.

    Memory: O(b*t * num_windows * ws^2 * ws^2) instead of O(b*t * (H*W)^2).
    At 720p latent (90x160) with ws=8: ~158MB vs ~2.5GB+ per chunk.
    """

    _WINDOW_SIZE: int = 8

    def __init__(
        self,
        dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(
            dim,
            images=True,
            dtype=dtype,
            device=device,
        )
        self.to_qkv = Conv2d(
            in_channels=dim,
            out_channels=dim * 3,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.proj = Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        identity = x
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]
        c = self.dim
        ws = self._WINDOW_SIZE

        # [b, c, t, h, w] -> [b*t, c, h, w]
        x2d = ops.permute(x, [0, 2, 1, 3, 4])
        x2d = ops.reshape(x2d, [b * t, c, h, w])
        x2d = self.norm(x2d)

        x2d_nhwc = ops.permute(x2d, [0, 2, 3, 1])  # [bt, h, w, c]
        qkv = self.to_qkv(x2d_nhwc)  # [bt, h, w, 3c]

        # Pad H and W up to the next multiple of ws.
        # Use concat with zero tensors — always applied (no Python branching
        # on symbolic dims). If already aligned, pad dims are 0-sized.
        h_p = ((h + ws - 1) // ws) * ws
        w_p = ((w + ws - 1) // ws) * ws
        pad_w = w_p - w
        pad_h = h_p - h
        zero_w = ops.constant(
            0.0, dtype=qkv.dtype, device=qkv.device
        ).broadcast_to([b * t, h, pad_w, 3 * c])
        qkv = ops.concat([qkv, zero_w], axis=2)
        zero_h = ops.constant(
            0.0, dtype=qkv.dtype, device=qkv.device
        ).broadcast_to([b * t, pad_h, w_p, 3 * c])
        qkv = ops.concat([qkv, zero_h], axis=1)

        hws = h_p // ws
        wws = w_p // ws
        nwin = hws * wws
        tok = ws * ws

        q = qkv[:, :, :, :c]
        k = qkv[:, :, :, c : 2 * c]
        v = qkv[:, :, :, 2 * c : 3 * c]

        def to_windows(y: TensorValue) -> TensorValue:
            y = ops.reshape(y, [b * t, hws, ws, wws, ws, c])
            y = ops.permute(y, [0, 1, 3, 2, 4, 5])
            return ops.reshape(y, [b * t, nwin, tok, c])

        q_w = to_windows(q)
        k_w = to_windows(k)
        v_w = to_windows(v)

        attn_scores = ops.matmul(
            q_w * (float(c) ** -0.5), ops.permute(k_w, [0, 1, 3, 2])
        )
        attn = ops.softmax(attn_scores, axis=-1)
        out = ops.matmul(attn, v_w)  # [bt, nwin, tok, c]

        out = ops.reshape(out, [b * t, hws, wws, ws, ws, c])
        out = ops.permute(out, [0, 1, 3, 2, 4, 5])
        out = ops.reshape(out, [b * t, h_p, w_p, c])

        # Slice back to original spatial dims (remove padding).
        out = out[:, :h, :w, :]

        out = self.proj(out)  # [bt, h, w, c]
        out = ops.permute(out, [0, 3, 1, 2])  # [bt, c, h, w]
        out = ops.reshape(out, [b, t, c, h, w])
        out = ops.permute(out, [0, 2, 1, 3, 4])
        return out + identity


class MidBlock(Module):
    """Middle decoder block with residual-attention-residual."""

    def __init__(
        self,
        dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        prefer_nvidia_fcrs: bool = True,
    ) -> None:
        super().__init__()
        self.resnets = LayerList(
            [
                ResidualBlock(
                    dim,
                    dim,
                    dtype=dtype,
                    device=device,
                    prefer_nvidia_fcrs=prefer_nvidia_fcrs,
                ),
                ResidualBlock(
                    dim,
                    dim,
                    dtype=dtype,
                    device=device,
                    prefer_nvidia_fcrs=prefer_nvidia_fcrs,
                ),
            ]
        )
        self.attentions = LayerList(
            [AttentionBlock(dim, dtype=dtype, device=device)]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class Upsample2d(Module):
    """Nearest-neighbor 2D upsample by factor 2."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: TensorValue) -> TensorValue:
        n = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        # Nearest-neighbor 2x upsample: [N,C,H,W] → [N,C,H*2,W*2]
        x = ops.reshape(x, [n, c, h, 1, w, 1])
        x = ops.concat([x, x], axis=3)  # [N, C, H, 2, W, 1]
        x = ops.concat([x, x], axis=5)  # [N, C, H, 2, W, 2]
        return ops.reshape(x, [n, c, h * 2, w * 2])


class Resample(Module):
    """Wan decoder upsampling module."""

    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int | None = None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        self._out_c = upsample_out_dim

        self.time_conv: CausalConv3d | None = None
        self.resample = LayerList(
            [
                Upsample2d(),
                Conv2dPermuted(
                    in_channels=dim,
                    out_channels=upsample_out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dtype=dtype,
                    device=device,
                    has_bias=True,
                ),
            ]
        )

        if mode == "upsample3d":
            self.time_conv = CausalConv3d(
                in_channels=dim,
                out_channels=dim * 2,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=(1, 0, 0),
                dtype=dtype,
                device=device,
                has_bias=True,
            )
        elif mode != "upsample2d":
            raise ValueError(f"Unsupported Resample mode: {mode}")

    def __call__(self, x: TensorValue) -> TensorValue:
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]

        if self.mode == "upsample3d":
            if self.time_conv is None:
                raise ValueError("time_conv is required for upsample3d mode")
            x = self.time_conv(x)
            # x: [b, 2*dim, t, h, w] -> interleave temporal frames
            x = ops.reshape(x, [b, 2, self.dim, t, h, w])
            x = ops.permute(x, [0, 2, 3, 1, 4, 5])  # [b, dim, t, 2, h, w]
            t = t * 2
            x = ops.reshape(x, [b, self.dim, t, h, w])

        # Per-frame 2D upsample + conv
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, t, c, h, w]
        x = ops.reshape(x, [b * t, self.dim, h, w])
        x = self.resample[0](x)  # Upsample2d: [b*t, dim, h*2, w*2]
        # Conv2dPermuted handles NCHW->NHWC->conv->NCHW internally.
        x = self.resample[1](x)  # [b*t, out_c, h*2, w*2]

        x = ops.reshape(x, [b, t, self._out_c, h * 2, w * 2])
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, out_c, t, h*2, w*2]
        return x


class UpBlock(Module):
    """Wan decoder up block composed of residual blocks and optional upsample."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        resnets: list[ResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                ResidualBlock(
                    current_dim,
                    out_dim,
                    dtype=dtype,
                    device=device,
                )
            )
            current_dim = out_dim
        self.resnets = LayerList(resnets)

        self.upsamplers: LayerList | None = None
        if upsample_mode is not None:
            self.upsamplers = LayerList(
                [
                    Resample(
                        out_dim,
                        mode=upsample_mode,
                        upsample_out_dim=None,
                        dtype=dtype,
                        device=device,
                    )
                ]
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsamplers is not None:
            x = self.upsamplers[0](x)

        return x


class Decoder3d(Module):
    """Wan 3D decoder module."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_upsample: tuple[bool, ...] = (False, True, True),
        out_channels: int = 3,
        is_residual: bool = False,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        del is_residual

        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]

        self.conv_in = CausalConv3d(
            z_dim,
            dims[0],
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        self.mid_block = MidBlock(dims[0], dtype=dtype, device=device)

        up_blocks: list[UpBlock] = []
        final_out_dim = dims[-1]
        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            if i > 0:
                in_dim = in_dim // 2

            up_flag = i != len(dim_mult) - 1
            upsample_mode: str | None = None
            if up_flag and temporal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            up_blocks.append(
                UpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    upsample_mode=upsample_mode,
                    dtype=dtype,
                    device=device,
                )
            )
            final_out_dim = out_dim

        self.up_blocks = LayerList(up_blocks)

        self.norm_out = RMSNorm(
            final_out_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv_out = CausalConv3d(
            final_out_dim,
            out_channels,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.conv_in(x)
        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.norm_out(x)
        x = ops.silu(x)
        x = self.conv_out(x)
        return x


class ResidualBlockCached(Module):
    """Wan residual block with explicit cache I/O for conv1/conv2."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            in_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv1 = CausalConv3dCached(
            in_dim,
            out_dim,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.norm2 = RMSNorm(
            out_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv2 = CausalConv3dCached(
            out_dim,
            out_dim,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.conv_shortcut = (
            CausalConv3d(
                in_dim,
                out_dim,
                1,
                padding=0,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
            if in_dim != out_dim
            else None
        )

    def __call__(
        self,
        x: TensorValue,
        cache1_in: TensorValue | None = None,
        cache2_in: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        residual = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )

        x = ops.silu(self.norm1(x))
        if cache1_in is None:
            cache1_in = _zero_cache_for(x)
        x, cache1_out = self.conv1.forward_cached(x, cache1_in)

        x = ops.silu(self.norm2(x))
        if cache2_in is None:
            cache2_in = _zero_cache_for(x)
        x, cache2_out = self.conv2.forward_cached(x, cache2_in)
        return x + residual, cache1_out, cache2_out


class MidBlockCached(Module):
    """Middle decoder block with cache threading."""

    def __init__(
        self,
        dim: int,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.resnets = LayerList(
            [
                ResidualBlockCached(dim, dim, dtype=dtype, device=device),
                ResidualBlockCached(dim, dim, dtype=dtype, device=device),
            ]
        )
        self.attentions = LayerList(
            [AttentionBlock(dim, dtype=dtype, device=device)]
        )

    def __call__(
        self, x: TensorValue, *cache_inputs: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue, TensorValue]:
        if len(cache_inputs) not in (0, 4):
            raise ValueError(
                f"MidBlockCached expected 0 or 4 cache tensors, got {len(cache_inputs)}"
            )

        cache1_in = cache_inputs[0] if len(cache_inputs) == 4 else None
        cache2_in = cache_inputs[1] if len(cache_inputs) == 4 else None
        x, cache1_out, cache2_out = self.resnets[0](x, cache1_in, cache2_in)
        x = self.attentions[0](x)

        cache3_in = cache_inputs[2] if len(cache_inputs) == 4 else None
        cache4_in = cache_inputs[3] if len(cache_inputs) == 4 else None
        x, cache3_out, cache4_out = self.resnets[1](x, cache3_in, cache4_in)
        return x, cache1_out, cache2_out, cache3_out, cache4_out


class ResampleCached(Module):
    """Wan upsample3d module with explicit cache I/O."""

    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int | None = None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        if mode != "upsample3d":
            raise ValueError("ResampleCached only supports mode='upsample3d'")

        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        self._out_c = upsample_out_dim

        self.time_conv = CausalConv3dCached(
            in_channels=dim,
            out_channels=dim * 2,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.resample = LayerList(
            [
                Upsample2d(),
                Conv2dPermuted(
                    in_channels=dim,
                    out_channels=upsample_out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dtype=dtype,
                    device=device,
                    has_bias=True,
                ),
            ]
        )

    def __call__(
        self,
        x: TensorValue,
        cache_in: TensorValue | None = None,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, TensorValue]:
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]

        if cache_in is None:
            cache_in = _zero_cache_for(x)

        if first_chunk:
            cache_out = cache_in
        else:
            x, cache_out = self.time_conv.forward_cached(x, cache_in)
            x = ops.reshape(x, [b, 2, self.dim, t, h, w])
            x = ops.permute(x, [0, 2, 3, 1, 4, 5])
            t = t * 2
            x = ops.reshape(x, [b, self.dim, t, h, w])

        x = ops.permute(x, [0, 2, 1, 3, 4])
        x = ops.reshape(x, [b * t, self.dim, h, w])
        x = self.resample[0](x)
        x = self.resample[1](x)
        x = ops.reshape(x, [b, t, self._out_c, h * 2, w * 2])
        x = ops.permute(x, [0, 2, 1, 3, 4])
        return x, cache_out


class UpBlockCached(Module):
    """Wan decoder up block with explicit cache threading."""

    cache_slots: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        resnets: list[ResidualBlockCached] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                ResidualBlockCached(
                    current_dim,
                    out_dim,
                    dtype=dtype,
                    device=device,
                )
            )
            current_dim = out_dim
        self.resnets = LayerList(resnets)

        self._has_temporal_upsample = upsample_mode == "upsample3d"
        self.cache_slots = len(resnets) * 2 + (
            1 if self._has_temporal_upsample else 0
        )

        self.upsamplers: LayerList | None = None
        if upsample_mode is not None:
            if upsample_mode == "upsample3d":
                upsampler: Module = ResampleCached(
                    out_dim,
                    mode=upsample_mode,
                    upsample_out_dim=None,
                    dtype=dtype,
                    device=device,
                )
            elif upsample_mode == "upsample2d":
                upsampler = Resample(
                    out_dim,
                    mode=upsample_mode,
                    upsample_out_dim=None,
                    dtype=dtype,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Unsupported UpBlockCached upsample mode: {upsample_mode}"
                )

            self.upsamplers = LayerList([upsampler])

    def __call__(
        self,
        x: TensorValue,
        *cache_inputs: TensorValue,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, ...]:
        if len(cache_inputs) not in (0, self.cache_slots):
            raise ValueError(
                f"UpBlockCached expected 0 or {self.cache_slots} cache tensors, got {len(cache_inputs)}"
            )

        use_cache_inputs = len(cache_inputs) == self.cache_slots
        cache_outputs: list[TensorValue] = []
        cache_idx = 0

        for resnet in self.resnets:
            cache1_in = cache_inputs[cache_idx] if use_cache_inputs else None
            cache2_in = (
                cache_inputs[cache_idx + 1] if use_cache_inputs else None
            )
            x, cache1_out, cache2_out = resnet(x, cache1_in, cache2_in)
            cache_outputs.extend([cache1_out, cache2_out])
            cache_idx += 2

        if self.upsamplers is not None:
            upsampler = self.upsamplers[0]
            if self._has_temporal_upsample:
                cache_in = cache_inputs[cache_idx] if use_cache_inputs else None
                if not isinstance(upsampler, ResampleCached):
                    raise TypeError(
                        "Expected ResampleCached for temporal upsample"
                    )
                x, cache_out = upsampler(
                    x,
                    cache_in,
                    first_chunk=first_chunk,
                )
                cache_outputs.append(cache_out)
            else:
                x = upsampler(x)

        return (x, *cache_outputs)


class Decoder3dCached(Module):
    """Wan 3D decoder with explicit cache tensor I/O."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_upsample: tuple[bool, ...] = (False, True, True),
        out_channels: int = 3,
        is_residual: bool = False,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        del is_residual

        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]

        self.conv_in = CausalConv3dCached(
            z_dim,
            dims[0],
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        self.mid_block = MidBlockCached(dims[0], dtype=dtype, device=device)

        up_blocks: list[UpBlockCached] = []
        final_out_dim = dims[-1]
        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            if i > 0:
                in_dim = in_dim // 2

            up_flag = i != len(dim_mult) - 1
            upsample_mode: str | None = None
            if up_flag and temporal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            up_blocks.append(
                UpBlockCached(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    upsample_mode=upsample_mode,
                    dtype=dtype,
                    device=device,
                )
            )
            final_out_dim = out_dim

        self.up_blocks = LayerList(up_blocks)
        self.norm_out = RMSNorm(
            final_out_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv_out = CausalConv3dCached(
            final_out_dim,
            out_channels,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(
        self,
        x: TensorValue,
        *cache_inputs: TensorValue,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, ...]:
        if len(cache_inputs) not in (0, WAN_DECODER_CACHE_SLOTS):
            raise ValueError(
                "Decoder3dCached expected 0 or "
                f"{WAN_DECODER_CACHE_SLOTS} cache tensors, got {len(cache_inputs)}"
            )

        use_cache_inputs = len(cache_inputs) == WAN_DECODER_CACHE_SLOTS
        cache_outputs: list[TensorValue] = []
        cache_idx = 0

        conv_in_cache = cache_inputs[cache_idx] if use_cache_inputs else None
        if conv_in_cache is None:
            conv_in_cache = _zero_cache_for(x)
        x, cache_out = self.conv_in.forward_cached(x, conv_in_cache)
        cache_outputs.append(cache_out)
        cache_idx += 1

        mid_cache_inputs: tuple[TensorValue, ...] = (
            tuple(cache_inputs[cache_idx : cache_idx + 4])
            if use_cache_inputs
            else ()
        )
        mid_outputs = self.mid_block(x, *mid_cache_inputs)
        x = mid_outputs[0]
        cache_outputs.extend(mid_outputs[1:])
        cache_idx += 4

        for up_block in self.up_blocks:
            block_cache_inputs: tuple[TensorValue, ...] = (
                tuple(
                    cache_inputs[cache_idx : cache_idx + up_block.cache_slots]
                )
                if use_cache_inputs
                else ()
            )
            block_outputs = up_block(
                x,
                *block_cache_inputs,
                first_chunk=first_chunk,
            )
            x = block_outputs[0]
            cache_outputs.extend(block_outputs[1:])
            cache_idx += up_block.cache_slots

        x = self.norm_out(x)
        x = ops.silu(x)
        conv_out_cache = cache_inputs[cache_idx] if use_cache_inputs else None
        if conv_out_cache is None:
            conv_out_cache = _zero_cache_for(x)
        x, cache_out = self.conv_out.forward_cached(x, conv_out_cache)
        cache_outputs.append(cache_out)

        if len(cache_outputs) != WAN_DECODER_CACHE_SLOTS:
            raise ValueError(
                "Decoder3dCached produced "
                f"{len(cache_outputs)} cache tensors, expected {WAN_DECODER_CACHE_SLOTS}"
            )
        return (x, *cache_outputs)

    def cache_shapes(
        self,
        batch_size: int,
        latent_height: int,
        latent_width: int,
    ) -> list[list[int]]:
        h = latent_height
        w = latent_width
        shapes: list[list[int]] = [
            [batch_size, self.conv_in.in_channels, CACHE_T, h, w]
        ]

        for resnet in self.mid_block.resnets:
            shapes.append([batch_size, resnet.conv1.in_channels, CACHE_T, h, w])
            shapes.append([batch_size, resnet.conv2.in_channels, CACHE_T, h, w])

        for up_block in self.up_blocks:
            for resnet in up_block.resnets:
                shapes.append(
                    [batch_size, resnet.conv1.in_channels, CACHE_T, h, w]
                )
                shapes.append(
                    [batch_size, resnet.conv2.in_channels, CACHE_T, h, w]
                )

            if up_block.upsamplers is not None:
                if up_block._has_temporal_upsample:
                    upsampler = up_block.upsamplers[0]
                    if not isinstance(upsampler, ResampleCached):
                        raise TypeError(
                            "Expected ResampleCached for temporal upsample"
                        )
                    shapes.append(
                        [
                            batch_size,
                            upsampler.time_conv.in_channels,
                            CACHE_T,
                            h,
                            w,
                        ]
                    )
                h *= 2
                w *= 2

        shapes.append([batch_size, self.conv_out.in_channels, CACHE_T, h, w])
        if len(shapes) != WAN_DECODER_CACHE_SLOTS:
            raise ValueError(
                f"Expected {WAN_DECODER_CACHE_SLOTS} cache shapes, got {len(shapes)}"
            )
        return shapes


class VAEPostQuantConv(Module):
    """Standalone post-quant conv graph (k=1, frame-independent)."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self.post_quant_conv = CausalConv3d(
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            kernel_size=1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )

    def __call__(self, z: TensorValue) -> TensorValue:
        return self.post_quant_conv(z)


class VAEDecoderFirstFrameCached(Module):
    """First-frame decoder graph returning pixels + initialized caches."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self.decoder = Decoder3dCached(
            dim=config.base_dim,
            z_dim=config.z_dim,
            dim_mult=tuple(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            temporal_upsample=tuple(reversed(config.temporal_downsample)),
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            dtype=config.dtype,
            device=config.device,
        )

    def __call__(self, z: TensorValue) -> tuple[TensorValue, ...]:
        outputs = self.decoder(z, first_chunk=True)
        x = outputs[0]
        x = ops.max(x, -1.0)
        x = ops.min(x, 1.0)
        return (x, *outputs[1:])


class VAEDecoderRestFrameCached(Module):
    """Per-frame decoder graph with cache feedback for frames 1..T-1."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self.decoder = Decoder3dCached(
            dim=config.base_dim,
            z_dim=config.z_dim,
            dim_mult=tuple(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            temporal_upsample=tuple(reversed(config.temporal_downsample)),
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            dtype=config.dtype,
            device=config.device,
        )

    def __call__(
        self, z: TensorValue, *cache_inputs: TensorValue
    ) -> tuple[TensorValue, ...]:
        outputs = self.decoder(z, *cache_inputs, first_chunk=False)
        x = outputs[0]
        x = ops.max(x, -1.0)
        x = ops.min(x, 1.0)
        return (x, *outputs[1:])


class VAEDecoder(Module):
    """Wan VAE decoder graph used by AutoencoderKLWanModel."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self._config = config
        self.post_quant_conv = CausalConv3d(
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            kernel_size=1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )
        self.decoder = Decoder3d(
            dim=config.base_dim,
            z_dim=config.z_dim,
            dim_mult=tuple(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            temporal_upsample=tuple(reversed(config.temporal_downsample)),
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            dtype=config.dtype,
            device=config.device,
        )

    def __call__(self, z: TensorValue) -> TensorValue:
        x = self.post_quant_conv(z)
        x = self.decoder(x)
        x = ops.max(x, -1.0)
        x = ops.min(x, 1.0)
        return x


class VAEDecoderFirstFrame(Module):
    """Wan VAE decoder for the FIRST latent frame.

    Identical to VAEDecoder but ALL temporal upsamples are replaced
    with spatial-only upsample2d (time_conv is omitted).  This means
    T=1 in -> T=1 out, matching the diffusers feat_cache behavior where
    the first frame skips temporal upsampling.
    """

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self._config = config
        self.post_quant_conv = CausalConv3d(
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            kernel_size=1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )
        # Force all temporal upsamples to spatial-only.
        self.decoder = Decoder3d(
            dim=config.base_dim,
            z_dim=config.z_dim,
            dim_mult=tuple(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            temporal_upsample=(False,) * len(config.temporal_downsample),
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            dtype=config.dtype,
            device=config.device,
        )

    def __call__(self, z: TensorValue) -> TensorValue:
        x = self.post_quant_conv(z)
        x = self.decoder(x)
        x = ops.max(x, -1.0)
        x = ops.min(x, 1.0)
        return x


class DownResample(Module):
    """Wan encoder downsampling module.

    Matches diffusers Resample downsample modes:
    - downsample2d: ZeroPad2d + Conv2d(stride=2) per frame
    - downsample3d: same spatial + CausalConv3d(stride=(2,1,1)) temporal
    """

    def __init__(
        self,
        dim: int,
        mode: str,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        prefer_nvidia_fcrs: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # Spatial: ZeroPad2d(0,1,0,1) + Conv2d(stride=2, padding=0)
        # Asymmetric padding: right=1, bottom=1 only.
        # Use index [1] to match state_dict key "resample.1".
        self.resample = LayerList(
            [
                Upsample2d(),  # Dummy at index 0 (no weights, not called)
                Conv2dPermuted(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    stride=2,
                    padding=0,  # We do manual asymmetric pad in __call__
                    dtype=dtype,
                    device=device,
                    has_bias=True,
                ),
            ]
        )

        self.time_conv: CausalConv3d | None = None
        if mode == "downsample3d":
            self.time_conv = CausalConv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
                dtype=dtype,
                device=device,
                has_bias=True,
                # Encoder temporal downsample is the only conv pattern that
                # currently reproduces cuDNN aborts in VAE encode.
                prefer_nvidia_fcrs=False,
            )
        elif mode != "downsample2d":
            raise ValueError(f"Unsupported DownResample mode: {mode}")

    def __call__(self, x: TensorValue) -> TensorValue:
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]

        if self.mode == "downsample3d":
            if self.time_conv is None:
                raise ValueError("time_conv is required for downsample3d mode")
            # Temporal downsample via strided causal conv
            x = self.time_conv(x)
            t = x.shape[2]

        # Per-frame spatial downsample: ZeroPad2d(0,1,0,1) + Conv2d(stride=2)
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, t, c, h, w]
        x = ops.reshape(x, [b * t, self.dim, h, w])
        # ZeroPad2d(left=0, right=1, top=0, bottom=1) on NCHW
        # paddings format: [N_before, N_after, C_before, C_after, H_before, H_after, W_before, W_after]
        x = ops.pad(x, [0, 0, 0, 0, 0, 1, 0, 1])
        x = self.resample[1](x)  # Conv2d stride=2, padding=0
        new_h = (h + 1) // 2
        new_w = (w + 1) // 2
        # Rebind so the compiler sees conv output shape matches our computation.
        x = ops.rebind(x, shape=[b * t, self.dim, new_h, new_w])
        x = ops.reshape(x, [b, t, self.dim, new_h, new_w])
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, dim, t, h/2, w/2]

        return x


class DownResampleCached(Module):
    """Encoder downsample with temporal cache for chunked encoding.

    Matches diffusers' Resample cache behavior for the encoder:
    - downsample2d: spatial only, no temporal cache
    - downsample3d first chunk: spatial downsample, skip time_conv, cache last frame
    - downsample3d rest chunk: spatial downsample, prepend cached frame, apply time_conv

    Spatial downsample is done FIRST (matching diffusers order), then temporal.
    """

    cache_slots: int

    def __init__(
        self,
        dim: int,
        mode: str,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        self._has_temporal = mode == "downsample3d"
        self.cache_slots = 1 if self._has_temporal else 0

        self.resample = LayerList(
            [
                Upsample2d(),  # Dummy at index 0 (match weight naming)
                Conv2dPermuted(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    dtype=dtype,
                    device=device,
                    has_bias=True,
                ),
            ]
        )

        self.time_conv: CausalConv3d | None = None
        if self._has_temporal:
            self.time_conv = CausalConv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
                dtype=dtype,
                device=device,
                has_bias=True,
                prefer_nvidia_fcrs=False,
            )
        elif mode != "downsample2d":
            raise ValueError(f"Unsupported DownResampleCached mode: {mode}")

    def __call__(
        self,
        x: TensorValue,
        *,
        cache_in: TensorValue | None = None,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, ...]:
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]

        # Spatial downsample first (matching diffusers order)
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, t, c, h, w]
        x = ops.reshape(x, [b * t, self.dim, h, w])
        x = ops.pad(x, [0, 0, 0, 0, 0, 1, 0, 1])  # ZeroPad2d(0,1,0,1)
        x = self.resample[1](x)  # Conv2d stride=2
        new_h = (h + 1) // 2
        new_w = (w + 1) // 2
        # Rebind so the compiler sees conv output shape matches our computation.
        x = ops.rebind(x, shape=[b * t, self.dim, new_h, new_w])
        x = ops.reshape(x, [b, t, self.dim, new_h, new_w])
        x = ops.permute(x, [0, 2, 1, 3, 4])  # [b, c, t, h', w']

        if self._has_temporal:
            assert self.time_conv is not None
            cache_out = x[:, :, -1:, :, :]  # Last frame after spatial
            if first_chunk:
                # Skip time_conv, return spatial output + cache
                return x, cache_out
            else:
                assert cache_in is not None
                # Rebind cache spatial dims to match x after spatial downsample.
                cache_in = ops.rebind(
                    cache_in,
                    shape=[
                        cache_in.shape[0],
                        cache_in.shape[1],
                        cache_in.shape[2],
                        x.shape[3],
                        x.shape[4],
                    ],
                )
                # Prepend cached last frame, apply time_conv
                x_cat = ops.concat([cache_in, x], axis=2)
                x = self.time_conv(x_cat)
                return x, cache_out

        return (x,)


class DownBlock(Module):
    """Wan encoder down block (mirror of UpBlock)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        downsample_mode: str | None,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        resnets: list[ResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                ResidualBlock(
                    current_dim,
                    out_dim,
                    dtype=dtype,
                    device=device,
                )
            )
            current_dim = out_dim
        self.resnets = LayerList(resnets)

        self.downsamplers: LayerList | None = None
        if downsample_mode is not None:
            self.downsamplers = LayerList(
                [
                    DownResample(
                        out_dim,
                        mode=downsample_mode,
                        dtype=dtype,
                        device=device,
                    )
                ]
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        for resnet in self.resnets:
            x = resnet(x)

        if self.downsamplers is not None:
            x = self.downsamplers[0](x)

        return x


class Encoder3d(Module):
    """Wan 3D encoder module (mirror of Decoder3d).

    Uses a flat ModuleList for down_blocks to match the diffusers
    safetensors key naming (encoder.down_blocks.{i}.{conv1,norm1,...}).
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        in_channels: int = 3,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample: tuple[bool, ...] = (False, True, True),
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()

        dims = [dim * u for u in [1, *list(dim_mult)]]

        self.conv_in = CausalConv3d(
            in_channels,
            dims[0],
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
            prefer_nvidia_fcrs=False,
        )

        # Flat ModuleList matching diffusers weight naming:
        # down_blocks.{0,1} = ResidualBlock (first level, 2 blocks)
        # down_blocks.2 = Resample (downsample)
        # down_blocks.{3,4} = ResidualBlock (second level)
        # down_blocks.5 = Resample ...etc
        down_blocks: list[Module] = []
        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            for j in range(num_res_blocks):
                down_blocks.append(
                    ResidualBlock(
                        in_dim if j == 0 else out_dim,
                        out_dim,
                        dtype=dtype,
                        device=device,
                        prefer_nvidia_fcrs=False,
                    )
                )
            down_flag = i != len(dim_mult) - 1
            if down_flag:
                mode = (
                    "downsample3d" if temporal_downsample[i] else "downsample2d"
                )
                down_blocks.append(
                    DownResample(
                        out_dim,
                        mode=mode,
                        dtype=dtype,
                        device=device,
                        prefer_nvidia_fcrs=False,
                    )
                )

        self.down_blocks = LayerList(down_blocks)

        final_dim = dims[-1]
        self.mid_block = MidBlock(
            final_dim,
            dtype=dtype,
            device=device,
            prefer_nvidia_fcrs=False,
        )

        self.norm_out = RMSNorm(
            final_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        # Output 2*z_dim for mean + logvar
        self.conv_out = CausalConv3d(
            final_dim,
            z_dim * 2,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
            prefer_nvidia_fcrs=False,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.conv_in(x)

        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.mid_block(x)
        x = self.norm_out(x)
        x = ops.silu(x)
        x = self.conv_out(x)
        return x


class Encoder3dCached(Module):
    """Chunked encoder with explicit cache I/O for temporal context.

    Uses a flat ModuleList for down_blocks (matching Encoder3d weight naming).
    Each chunk processes either 1 frame (first) or CHUNK_SIZE frames (rest).
    Temporal context is maintained via cache tensors passed between chunks.
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        in_channels: int = 3,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample: tuple[bool, ...] = (False, True, True),
        dtype: DType | None = None,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self._dim = dim
        self._in_channels = in_channels
        self._dim_mult = dim_mult
        self._num_res_blocks = num_res_blocks
        self._temporal_downsample = temporal_downsample

        dims = [dim * u for u in [1, *list(dim_mult)]]

        self.conv_in = CausalConv3dCached(
            in_channels,
            dims[0],
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        # Flat list matching diffusers weight naming
        down_blocks: list[Module] = []
        self._block_cache_slots: list[int] = []
        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            for j in range(num_res_blocks):
                down_blocks.append(
                    ResidualBlockCached(
                        in_dim if j == 0 else out_dim,
                        out_dim,
                        dtype=dtype,
                        device=device,
                    )
                )
                self._block_cache_slots.append(2)
            down_flag = i != len(dim_mult) - 1
            if down_flag:
                mode = (
                    "downsample3d" if temporal_downsample[i] else "downsample2d"
                )
                ds = DownResampleCached(
                    out_dim,
                    mode=mode,
                    dtype=dtype,
                    device=device,
                )
                down_blocks.append(ds)
                self._block_cache_slots.append(ds.cache_slots)

        self.down_blocks = LayerList(down_blocks)

        final_dim = dims[-1]
        self.mid_block = MidBlockCached(final_dim, dtype=dtype, device=device)

        self.norm_out = RMSNorm(
            final_dim,
            images=False,
            dtype=dtype,
            device=device,
        )
        self.conv_out = CausalConv3dCached(
            final_dim,
            z_dim * 2,
            3,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    @property
    def total_cache_slots(self) -> int:
        return 1 + sum(self._block_cache_slots) + 4 + 1

    def cache_shapes(
        self,
        batch_size: int,
        height: int | None = None,
        width: int | None = None,
    ) -> list[list[int | None]]:
        """Compute cache shapes for this encoder configuration.

        If height/width are None, those dimensions are dynamic.
        """
        dims = [self._dim * u for u in [1, *list(self._dim_mult)]]
        h: int | None = height
        w: int | None = width
        shapes: list[list[int | None]] = []

        # conv_in cache
        shapes.append([batch_size, self._in_channels, CACHE_T, h, w])

        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            for j in range(self._num_res_blocks):
                block_in = in_dim if j == 0 else out_dim
                shapes.append([batch_size, block_in, CACHE_T, h, w])
                shapes.append([batch_size, out_dim, CACHE_T, h, w])

            down_flag = i != len(self._dim_mult) - 1
            if down_flag:
                new_h = (h + 1) // 2 if h is not None else None
                new_w = (w + 1) // 2 if w is not None else None
                if self._temporal_downsample[i]:
                    shapes.append([batch_size, out_dim, 1, new_h, new_w])
                h, w = new_h, new_w

        final_dim = dims[-1]
        for _ in range(4):
            shapes.append([batch_size, final_dim, CACHE_T, h, w])

        shapes.append([batch_size, final_dim, CACHE_T, h, w])

        assert len(shapes) == self.total_cache_slots, (
            f"cache_shapes produced {len(shapes)}, "
            f"expected {self.total_cache_slots}"
        )
        return shapes

    def __call__(
        self,
        x: TensorValue,
        *cache_inputs: TensorValue,
        first_chunk: bool = False,
    ) -> tuple[TensorValue, ...]:
        use_cache = len(cache_inputs) == self.total_cache_slots
        if len(cache_inputs) not in (0, self.total_cache_slots):
            raise ValueError(
                f"Encoder3dCached expected 0 or {self.total_cache_slots} "
                f"cache tensors, got {len(cache_inputs)}"
            )

        cache_outputs: list[TensorValue] = []
        idx = 0

        # conv_in
        c_in = cache_inputs[idx] if use_cache else _zero_cache_for(x)
        x, c_out = self.conv_in.forward_cached(x, c_in)
        cache_outputs.append(c_out)
        idx += 1

        # down_blocks (flat list of ResidualBlockCached and DownResampleCached)
        for block in self.down_blocks:
            if isinstance(block, ResidualBlockCached):
                c1 = cache_inputs[idx] if use_cache else None
                c2 = cache_inputs[idx + 1] if use_cache else None
                x, co1, co2 = block(x, c1, c2)
                cache_outputs.extend([co1, co2])
                idx += 2
            elif isinstance(block, DownResampleCached):
                if block._has_temporal:
                    c = cache_inputs[idx] if use_cache else None
                    x, co = block(x, cache_in=c, first_chunk=first_chunk)
                    cache_outputs.append(co)
                    idx += 1
                else:
                    (x,) = block(x)

        # mid_block
        mid_caches: tuple[TensorValue, ...] = (
            tuple(cache_inputs[idx : idx + 4]) if use_cache else ()
        )
        mid_out = self.mid_block(x, *mid_caches)
        x = mid_out[0]
        cache_outputs.extend(mid_out[1:])
        idx += 4

        # norm + silu + conv_out
        x = ops.silu(self.norm_out(x))
        c_in = cache_inputs[idx] if use_cache else _zero_cache_for(x)
        x, c_out = self.conv_out.forward_cached(x, c_in)
        cache_outputs.append(c_out)

        assert len(cache_outputs) == self.total_cache_slots, (
            f"Produced {len(cache_outputs)} caches, "
            f"expected {self.total_cache_slots}"
        )
        return (x, *cache_outputs)


class VAEEncoder(Module):
    """Wrapper for VAE encoder graph compilation.

    Includes quant_conv (1x1 conv applied after encoder output).
    """

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self.encoder = Encoder3d(
            dim=config.base_dim,
            z_dim=config.z_dim,
            in_channels=3,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            temporal_downsample=config.temporal_downsample,
            dtype=config.dtype,
            device=config.device,
        )
        z2 = config.z_dim * 2
        self.quant_conv = CausalConv3d(
            z2,
            z2,
            1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
            prefer_nvidia_fcrs=False,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        h = self.encoder(x)
        return self.quant_conv(h)


class VAEEncoderFirstChunk(Module):
    """First-chunk encoder graph: 1 frame in, mean latent + caches out."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self._z_dim = config.z_dim
        self.encoder = Encoder3dCached(
            dim=config.base_dim,
            z_dim=config.z_dim,
            in_channels=3,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            temporal_downsample=config.temporal_downsample,
            dtype=config.dtype,
            device=config.device,
        )
        z2 = config.z_dim * 2
        self.quant_conv = CausalConv3d(
            z2,
            z2,
            1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> tuple[TensorValue, ...]:
        outputs = self.encoder(x, first_chunk=True)
        moments = self.quant_conv(outputs[0])
        # Extract mean in-graph to avoid GPU->CPU transfer of full moments
        mean = moments[:, : self._z_dim, :, :, :]
        return (mean, *outputs[1:])


class VAEEncoderRestChunk(Module):
    """Rest-chunk encoder graph: CHUNK_SIZE frames + caches in, mean latent + caches out."""

    def __init__(self, config: AutoencoderKLWanConfig) -> None:
        super().__init__()
        self._z_dim = config.z_dim
        self.encoder = Encoder3dCached(
            dim=config.base_dim,
            z_dim=config.z_dim,
            in_channels=3,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            temporal_downsample=config.temporal_downsample,
            dtype=config.dtype,
            device=config.device,
        )
        z2 = config.z_dim * 2
        self.quant_conv = CausalConv3d(
            z2,
            z2,
            1,
            padding=0,
            dtype=config.dtype,
            device=config.device,
            has_bias=True,
        )

    def __call__(
        self, x: TensorValue, *cache_inputs: TensorValue
    ) -> tuple[TensorValue, ...]:
        outputs = self.encoder(x, *cache_inputs, first_chunk=False)
        moments = self.quant_conv(outputs[0])
        mean = moments[:, : self._z_dim, :, :, :]
        return (mean, *outputs[1:])
