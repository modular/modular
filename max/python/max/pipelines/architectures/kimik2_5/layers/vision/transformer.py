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
from max.graph import BufferValue, DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.comm import Allreduce
from max.nn.kernels import tpool_patch_merger
from max.nn.layer import Module

from ...model_config import VisionConfig
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
        super().__init__()
        self.devices = config.devices
        self.merge_kernel_size = (
            config.merge_kernel_size[0],
            config.merge_kernel_size[1],
        )

        device = config.devices[0]
        self.patch_embed = PatchEmbedding(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_size=config.vt_hidden_size,
            init_pos_emb_height=config.init_pos_emb_height,
            init_pos_emb_width=config.init_pos_emb_width,
            init_pos_emb_time=config.init_pos_emb_time,
            dtype=config.dtype,
            device=device,
            has_bias=config.has_bias,
        )
        self.encoder = Encoder(
            num_heads=config.vt_num_attention_heads,
            hidden_dim=config.vt_hidden_size,
            mlp_dim=config.vt_intermediate_size,
            num_layers=config.vt_num_hidden_layers,
            rope_max_height=config.rope_max_height,
            rope_max_width=config.rope_max_width,
            rope_theta=config.rope_theta,
            dtype=config.dtype,
            device=device,
            has_bias=config.has_bias,
        )
        self.patch_merger = PatchMergerMLP(
            dtype=config.dtype,
            device=device,
            mm_hidden_size=config.mm_hidden_size,
            hidden_size=config.text_hidden_size,
            merge_kernel_size=self.merge_kernel_size,
            eps=config.projector_ln_eps,
        )

        if len(config.devices) > 1:
            self.patch_embed.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )
            self.encoder.sharding_strategy = ShardingStrategy.tensor_parallel(
                len(config.devices)
            )
            self.patch_merger.sharding_strategy = (
                ShardingStrategy.tensor_parallel(len(config.devices))
            )
            self.patch_embed_shards = self.patch_embed.shard(config.devices)
            self.encoder_shards = self.encoder.shard(config.devices)
            self.patch_merger_shards = self.patch_merger.shard(config.devices)
        else:
            self.patch_embed_shards = [self.patch_embed]
            self.encoder_shards = [self.encoder]
            self.patch_merger_shards = [self.patch_merger]

        self.allreduce = (
            Allreduce(num_accelerators=len(config.devices))
            if len(config.devices) > 1
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
            total_output_patches: Static upper bound on ``sum(H_i * W_i)``
                across all images in a batch. Used as the compile-time output
                shape of ``tpool_patch_merger``. The Mojo kernel has no
                registered ``mogg.shape`` function so a dynamic string dim is
                not supported; callers should pass
                ``pipeline_config.runtime.max_batch_input_tokens``.

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
        # Compute max_h and max_w at runtime from grid_thws (replicated across
        # devices, so any shard gives the same values). ops.max keeps rank,
        # returning shape (1,); reshape to () (rank-0 scalar) and move to CPU
        # as the kernel requires scalar operands on the host.
        grid0 = grid_thws[0]  # (n_videos, 3), int64, on GPU
        max_h_tv = ops.reshape(
            ops.max(grid0[:, 1], axis=0).cast(DType.int32), []
        ).to(DeviceRef.CPU())
        max_w_tv = ops.reshape(
            ops.max(grid0[:, 2], axis=0).cast(DType.int32), []
        ).to(DeviceRef.CPU())
        hs = [
            tpool_patch_merger(
                h,
                grid,
                kH=kH,
                kW=kW,
                max_h=max_h_tv,
                max_w=max_w_tv,
                total_output_patches=total_output_patches,
            )
            for h, grid in zip(hs, grid_thws, strict=True)
        ]
        # PatchMergerMLP expects (n_spatial, kH*kW, D); kernel returns (total_output_patches, D).
        # total_output_patches is a dynamic dim, so rebind first to assert it is
        # divisible by kH*kW before reshaping (as suggested by the MAX error message).
        merge_k = kH * kW
        hs = [
            h.rebind([(h.shape[0] // merge_k) * merge_k, h.shape[1]]).reshape(
                [h.shape[0] // merge_k, merge_k, h.shape[1]]
            )
            for h in hs
        ]
        hs = [
            merger(h)
            for merger, h in zip(self.patch_merger_shards, hs, strict=True)
        ]
        if self.allreduce is not None:
            return self.allreduce(hs, signal_buffers)
        return hs
