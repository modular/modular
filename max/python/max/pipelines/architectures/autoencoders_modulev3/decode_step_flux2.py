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

"""Fused decode-step module for the Flux2 pipeline.

Combines Flux2-specific BN denorm + unpatchify with the VAE decoder forward
pass into a single compiled graph, eliminating the inter-graph boundary that
previously existed between _postprocess_latents and vae.decode().
"""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType

from .vae import Decoder


class Flux2DecodeStep(Module[..., Tensor]):
    """Fused postprocess-and-decode: packed latents -> decoded image.

    Combines Flux2-specific BN denorm + unpatchify with the VAE decoder
    forward pass into a single compiled graph, eliminating the inter-graph
    boundary that previously existed between postprocess_latents and
    vae.decode().

    Accepts packed latents in (B, S, C) shape where S = latent_h * latent_w.
    Spatial dimensions are conveyed via two 1-D shape-carrier tensors whose
    *lengths* encode latent_h and latent_w as symbolic graph Dims, so a single
    compiled graph handles any spatial size without recompilation.
    """

    def __init__(self, decoder: Decoder, batch_norm_eps: float) -> None:
        """Initialize Flux2DecodeStep.

        Args:
            decoder: Raw (uncompiled) Decoder sub-module.
            batch_norm_eps: Epsilon value for BatchNorm denormalization.
        """
        super().__init__()
        self.decoder = decoder
        self.batch_norm_eps = batch_norm_eps

    def input_types(self) -> tuple[TensorType, ...]:
        """Return input TensorTypes for compilation.

        Returns:
            Tuple of TensorType objects corresponding to the forward() signature:
            (latents_bsc, h_carrier, w_carrier, bn_mean, bn_var).
        """
        num_channels = self.decoder.in_channels * 4  # e.g. 32*4 = 128
        dtype = self.decoder.dtype
        device = self.decoder.device
        assert dtype is not None, "Decoder dtype must be set before compilation"
        assert device is not None, (
            "Decoder device must be set before compilation"
        )
        return (
            TensorType(
                dtype, shape=["batch", "seq", num_channels], device=device
            ),
            # Shape carriers: lengths encode latent_h / latent_w as symbolic dims.
            # Content is never read; only the shapes matter.
            TensorType(
                DType.float32, shape=["latent_h"], device=DeviceRef.CPU()
            ),
            TensorType(
                DType.float32, shape=["latent_w"], device=DeviceRef.CPU()
            ),
            TensorType(dtype, shape=[num_channels], device=device),
            TensorType(dtype, shape=[num_channels], device=device),
        )

    def forward(
        self,
        latents_bsc: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
        bn_mean: Tensor,
        bn_var: Tensor,
    ) -> Tensor:
        """Run BN denorm + unpatchify + VAE decode in one fused graph.

        Args:
            latents_bsc: Packed latents of shape (B, S, C) where S = latent_h * latent_w.
            h_carrier: 1-D shape carrier of length latent_h (content unused).
            w_carrier: 1-D shape carrier of length latent_w (content unused).
            bn_mean: BatchNorm running mean of shape (C,).
            bn_var: BatchNorm running variance of shape (C,).

        Returns:
            Decoded image tensor of shape (B, 3, H*4, W*4).
        """
        batch = latents_bsc.shape[0]
        c = latents_bsc.shape[2]
        # Extract spatial dims from carrier shapes (symbolic Dims, not runtime values)
        h = h_carrier.shape[0]
        w = w_carrier.shape[0]

        # Assert seq == latent_h * latent_w so the reshape verifier accepts it,
        # then reshape packed (B, S, C) -> spatial (B, H, W, C).
        latents_bsc = F.rebind(latents_bsc, [batch, h * w, c])
        latents_bhwc = F.reshape(latents_bsc, (batch, h, w, c))

        # Permute: (B, H, W, C) -> (B, C, H, W)
        latents = F.permute(latents_bhwc, (0, 3, 1, 2))

        # BN denormalization
        bn_mean_r = F.reshape(bn_mean, (1, c, 1, 1))
        bn_var_r = F.reshape(bn_var, (1, c, 1, 1))
        bn_std = F.sqrt(bn_var_r + self.batch_norm_eps)
        latents = latents * bn_std + bn_mean_r

        # Unpatchify: (B, C, H, W) -> (B, C//4, H*2, W*2)
        latents = F.reshape(latents, (batch, c // 4, 2, 2, h, w))
        latents = F.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = F.reshape(latents, (batch, c // 4, h * 2, w * 2))

        return self.decoder(latents, None)
