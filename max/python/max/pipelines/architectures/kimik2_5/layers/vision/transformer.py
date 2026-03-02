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

"""Vision transformer for Kimi K2.5."""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, ShardingStrategy, TensorValue
from max.nn.comm import Allreduce
from max.nn.kernels import tpool_patch_merger
from max.nn.layer import LayerList, Module

from .encoder import Encoder
from .patch_embedding import PatchEmbedding


class Transformer(Module):
    """Full vision transformer.

    Composes patch embedding, the encoder stack, and spatial-temporal
    patch merging via ``tpool_patch_merger``.

    Auto-shards sub-layers with tensor-parallel strategy on construction.

    Args:
        patch_size: Spatial patch size for the Conv2d projection.
        in_channels: Number of input image channels (3 for RGB).
        hidden_dim: Hidden dimension throughout the vision transformer.
        num_heads: Number of attention heads.
        mlp_dim: Inner dimension of the feed-forward MLP.
        num_layers: Number of encoder layers.
        init_pos_emb_height: Height of the learnable 2D position grid.
        init_pos_emb_width: Width of the learnable 2D position grid.
        init_pos_emb_time: Number of temporal steps for 1D sincos time
            embedding.
        rope_max_height: Maximum grid height for RoPE frequencies.
        rope_max_width: Maximum grid width for RoPE frequencies.
        rope_theta: Base for the RoPE inverse-frequency exponent.
        merge_kernel_size: (kH, kW) spatial merge kernel for
            ``tpool_patch_merger``.
        dtype: Data type for all layer weights.
        devices: Devices on which to allocate and shard weights.
        has_bias: Whether linear projections include bias terms.
    """

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        init_pos_emb_height: int,
        init_pos_emb_width: int,
        init_pos_emb_time: int,
        rope_max_height: int,
        rope_max_width: int,
        rope_theta: float,
        merge_kernel_size: tuple[int, int],
        dtype: DType,
        devices: list[DeviceRef],
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.devices = devices
        self.merge_kernel_size = merge_kernel_size

        device = devices[0]
        patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_dim,
            init_pos_emb_height=init_pos_emb_height,
            init_pos_emb_width=init_pos_emb_width,
            init_pos_emb_time=init_pos_emb_time,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )
        encoder = Encoder(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            rope_max_height=rope_max_height,
            rope_max_width=rope_max_width,
            rope_theta=rope_theta,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )

        if len(devices) > 1:
            patch_embed.sharding_strategy = ShardingStrategy.replicate(
                len(devices)
            )
            encoder.sharding_strategy = ShardingStrategy.tensor_parallel(
                len(devices)
            )
            self.patch_embed_shards = LayerList(patch_embed.shard(devices))
            self.encoder_shards = LayerList(encoder.shard(devices))
        else:
            self.patch_embed_shards = LayerList([patch_embed])
            self.encoder_shards = LayerList([encoder])

        self.allreduce = (
            Allreduce(num_accelerators=len(devices))
            if len(devices) > 1
            else None
        )

    def __call__(
        self,
        pixel_values: Sequence[TensorValue],
        grid_thws: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        max_seq_len: Sequence[TensorValue],
        position_ids: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        max_h: int,
        max_w: int,
        total_output_patches: int,
    ) -> list[TensorValue]:
        """Full vision transformer forward pass.

        Args:
            pixel_values: Per-device (n_patches, in_channels, patch_size,
                patch_size) NCHW pixel patches.
            grid_thws: Per-device (n_videos, 3) temporal, height, width per
                video, dtype int64.
            input_row_offsets: Per-device cumulative sequence lengths of shape
                (batch_size + 1,), dtype uint32.
            max_seq_len: Per-device maximum sequence length, shape (1,),
                dtype uint32.
            position_ids: Per-device 1-D int tensor of flat grid indices for
                RoPE, dtype int64.
            signal_buffers: Per-device communication buffers for allreduce.
                Only required when using multiple devices; ignored on
                single device.
            max_h: Maximum grid height across all videos in the batch.
            max_w: Maximum grid width across all videos in the batch.
            total_output_patches: Total number of output patches after
                merging, i.e. ``sum(H_i * W_i)`` over all videos.

        Returns:
            Per-device allreduced merged patch tensors of shape
            ``(total_output_patches, hidden_dim)``.
        """
        hs = [
            patch_embed(pv, grid)
            for patch_embed, pv, grid in zip(
                self.patch_embed_shards, pixel_values, grid_thws, strict=True
            )
        ]
        hs = [
            encoder(h, offsets, max_len, pos_ids)
            for encoder, h, offsets, max_len, pos_ids in zip(
                self.encoder_shards,
                hs,
                input_row_offsets,
                max_seq_len,
                position_ids,
                strict=True,
            )
        ]
        kH, kW = self.merge_kernel_size
        hs = [
            tpool_patch_merger(
                h,
                grid,
                kH=kH,
                kW=kW,
                max_h=max_h,
                max_w=max_w,
                total_output_patches=total_output_patches,
            )
            for h, grid in zip(hs, grid_thws, strict=True)
        ]
        if self.allreduce is not None:
            return self.allreduce(hs, signal_buffers)
        return hs
