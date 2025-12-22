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

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import max.experimental.functional as F
import max.nn.module_v3 as nn
from max.experimental.tensor import Tensor

from .layers import AutoencoderKLOutput, Conv2d
from .vae import (
    AutoencoderMixin,
    Decoder,
    DecoderOutput,
    DiagonalGaussianDistribution,
    Encoder,
)


class AutoencoderKL(nn.Module, AutoencoderMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without losing too much precision in which case `force_upcast`
            can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        mid_block_add_attention (`bool`, *optional*, default to `True`):
            If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
            mid_block will only have resnet blocks
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: float | None = None,
        latents_mean: tuple[float] | None = None,
        latents_std: tuple[float] | None = None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = (
            Conv2d(2 * latent_channels, 2 * latent_channels, 1)
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            Conv2d(latent_channels, latent_channels, 1)
            if use_post_quant_conv
            else None
        )

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        sample_size = (
            sample_size[0]
            if isinstance(sample_size, (list, tuple))
            else sample_size
        )
        self.tile_latent_min_size = int(
            sample_size / (2 ** (len(block_out_channels) - 1))
        )
        self.tile_overlap_factor = 0.25

    def _encode(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape

        if self.use_tiling and (
            width > self.tile_sample_min_size
            or height > self.tile_sample_min_size
        ):
            return self._tiled_encode(x)

        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)

        return enc

    def encode(
        self, x: Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        """
        Encode a batch of images into latents.

        Args:
            x (`Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = F.concat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, z: Tensor, return_dict: bool = True
    ) -> DecoderOutput | Tensor:
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
        ):
            return self.tiled_decode(z, return_dict=return_dict)

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(
        self,
        z: Tensor,
        return_dict: bool = True,
        # generator=None
    ) -> DecoderOutput | Tensor:
        """
        Decode a batch of images.

        Args:
            z (`Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [
                self._decode(z_slice).sample for z_slice in z.split(1)
            ]
            decoded = F.concat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
        """Blend two tensors vertically using slice/concat (no item assignment)."""
        a_h = int(a.shape[2])
        b_h = int(b.shape[2])
        blend_extent = min(a_h, b_h, blend_extent)
        if blend_extent == 0:
            return b

        # Build blended slices
        slices = []
        for y in range(blend_extent):
            weight = y / blend_extent
            # Convert negative index to positive: -blend_extent + y => a_h - blend_extent + y
            a_start = a_h - blend_extent + y
            a_slice = a[:, :, a_start : a_start + 1, :]
            b_slice = b[:, :, y : y + 1, :]
            blended = a_slice * (1 - weight) + b_slice * weight
            slices.append(blended)

        # Concatenate blended region with rest of b
        blended_region = F.concat(slices, axis=2)
        rest_of_b = b[:, :, blend_extent:, :]
        return F.concat([blended_region, rest_of_b], axis=2)

    def blend_h(self, a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
        """Blend two tensors horizontally using slice/concat (no item assignment)."""
        a_w = int(a.shape[3])
        b_w = int(b.shape[3])
        blend_extent = min(a_w, b_w, blend_extent)
        if blend_extent == 0:
            return b

        # Build blended slices
        slices = []
        for x in range(blend_extent):
            weight = x / blend_extent
            # Convert negative index to positive: -blend_extent + x => a_w - blend_extent + x
            a_start = a_w - blend_extent + x
            a_slice = a[:, :, :, a_start : a_start + 1]
            b_slice = b[:, :, :, x : x + 1]
            blended = a_slice * (1 - weight) + b_slice * weight
            slices.append(blended)

        # Concatenate blended region with rest of b
        blended_region = F.concat(slices, axis=3)
        rest_of_b = b[:, :, :, blend_extent:]
        return F.concat([blended_region, rest_of_b], axis=3)

    def _tiled_encode(self, x: Tensor) -> Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`Tensor`): Input batch of images.

        Returns:
            `Tensor`:
                The latent representation of the encoded videos.
        """

        overlap_size = int(
            self.tile_sample_min_size * (1 - self.tile_overlap_factor)
        )
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, int(x.shape[2]), overlap_size):
            row = []
            for j in range(0, int(x.shape[3]), overlap_size):
                end_i = min(i + self.tile_sample_min_size, int(x.shape[2]))
                end_j = min(j + self.tile_sample_min_size, int(x.shape[3]))
                tile = x[:, :, i:end_i, j:end_j]
                tile = self.encoder(tile)
                if self.quant_conv is not None:
                    tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                # Clamp slice limits to actual tile dimensions
                tile_h = min(row_limit, int(tile.shape[2]))
                tile_w = min(row_limit, int(tile.shape[3]))
                result_row.append(tile[:, :, :tile_h, :tile_w])
            result_rows.append(F.concat(result_row, axis=3))

        enc = F.concat(result_rows, axis=2)
        return enc

    def tiled_encode(
        self, x: Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """

        overlap_size = int(
            self.tile_sample_min_size * (1 - self.tile_overlap_factor)
        )
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, int(x.shape[2]), overlap_size):
            row = []
            for j in range(0, int(x.shape[3]), overlap_size):
                end_i = min(i + self.tile_sample_min_size, int(x.shape[2]))
                end_j = min(j + self.tile_sample_min_size, int(x.shape[3]))
                tile = x[:, :, i:end_i, j:end_j]
                tile = self.encoder(tile)
                if self.quant_conv is not None:
                    tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                # Clamp slice limits to actual tile dimensions
                tile_h = min(row_limit, int(tile.shape[2]))
                tile_w = min(row_limit, int(tile.shape[3]))
                result_row.append(tile[:, :, :tile_h, :tile_w])
            result_rows.append(F.concat(result_row, axis=3))

        moments = F.concat(result_rows, axis=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(
        self, z: Tensor, return_dict: bool = True
    ) -> DecoderOutput | Tensor:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(
            self.tile_latent_min_size * (1 - self.tile_overlap_factor)
        )
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, int(z.shape[2]), overlap_size):
            row = []
            for j in range(0, int(z.shape[3]), overlap_size):
                end_i = min(i + self.tile_latent_min_size, int(z.shape[2]))
                end_j = min(j + self.tile_latent_min_size, int(z.shape[3]))
                tile = z[:, :, i:end_i, j:end_j]
                if self.post_quant_conv is not None:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                # Clamp slice limits to actual tile dimensions
                tile_h = min(row_limit, int(tile.shape[2]))
                tile_w = min(row_limit, int(tile.shape[3]))
                result_row.append(tile[:, :, :tile_h, :tile_w])
            result_rows.append(F.concat(result_row, axis=3))

        dec = F.concat(result_rows, axis=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def __call__(
        self,
        sample: Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        # generator: Generator | None = None,
    ) -> DecoderOutput | Tensor:
        r"""
        Args:
            sample (`Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(
                # generator=generator
            )
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
