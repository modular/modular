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

from max.graph import BufferValue, ShardingStrategy, TensorValue, ops
from max.nn.comm import Allreduce
from max.nn.kernels import tpool_patch_merger
from max.nn.layer import Module
from max.pipelines.architectures.kimik2_5.model_config import VisionConfig

from .encoder import Encoder
from .patch_embedding import PatchEmbedding
from .patch_merger import PatchMergerMLP


class Transformer(Module):
    """Full vision transformer.

    Composes patch embedding, the encoder stack, spatial-temporal
    patch merging via ``tpool_patch_merger``, and a learned
    :obj:`PatchMergerMLP` projection.

    Auto-shards sub-layers with tensor-parallel strategy on construction.

    Args:
        config: :obj:`VisionConfig` instance from which all architecture
            parameters are derived.
    """

    def __init__(self, config: VisionConfig) -> None:
        patch_size = config.patch_size
        in_channels = config.in_channels
        hidden_dim = config.vt_hidden_size
        num_heads = config.vt_num_attention_heads
        mlp_dim = config.vt_intermediate_size
        num_layers = config.vt_num_hidden_layers
        init_pos_emb_height = config.init_pos_emb_height
        init_pos_emb_width = config.init_pos_emb_width
        init_pos_emb_time = config.init_pos_emb_time
        rope_max_height = config.rope_max_height
        rope_max_width = config.rope_max_width
        rope_theta = config.rope_theta
        kH, kW = config.merge_kernel_size
        merge_kernel_size: tuple[int, int] = (kH, kW)
        decoder_hidden_size = config.text_hidden_size
        dtype = config.dtype
        devices = config.devices
        has_bias = config.has_bias
        merger_eps = config.projector_ln_eps
        super().__init__()
        self.devices = devices
        self.merge_kernel_size = merge_kernel_size

        device = devices[0]
        self.patch_embed = PatchEmbedding(
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
        self.encoder = Encoder(
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
        self.patch_merger = PatchMergerMLP(
            dtype=dtype,
            device=device,
            mm_hidden_size=config.mm_hidden_size,
            hidden_size=decoder_hidden_size,
            merge_kernel_size=merge_kernel_size,
            eps=merger_eps,
        )

        if len(devices) > 1:
            self.patch_embed.sharding_strategy = ShardingStrategy.replicate(
                len(devices)
            )
            self.encoder.sharding_strategy = ShardingStrategy.tensor_parallel(
                len(devices)
            )
            self.patch_merger.sharding_strategy = (
                ShardingStrategy.tensor_parallel(len(devices))
            )
            self.patch_embed_shards = self.patch_embed.shard(devices)
            self.encoder_shards = self.encoder.shard(devices)
            self.patch_merger_shards = self.patch_merger.shard(devices)
        else:
            self.patch_embed_shards = [self.patch_embed]
            self.encoder_shards = [self.encoder]
            self.patch_merger_shards = [self.patch_merger]

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
        # PatchMergerMLP expects (n_spatial, kH*kW, D); kernel returns (total_output_patches, D).
        hs = [
            ops.reshape(h, (h.shape[0] // (kH * kW), kH * kW, h.shape[1]))
            for h in hs
        ]
        hs = [
            merger(h)
            for merger, h in zip(self.patch_merger_shards, hs, strict=True)
        ]
        if self.allreduce is not None:
            return self.allreduce(hs, signal_buffers)
        return hs
