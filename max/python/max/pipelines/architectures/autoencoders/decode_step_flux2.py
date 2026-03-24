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

"""Fused decode-step module for the Flux2 Module V2 pipeline."""

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn.layer import Module

from .vae import Decoder


class Flux2DecodeStep(Module):
    """Fused packed-latent postprocess and VAE decode step."""

    def __init__(self, decoder: Decoder, batch_norm_eps: float) -> None:
        super().__init__()
        self.decoder = decoder
        self.batch_norm_eps = batch_norm_eps

    def input_types(self) -> tuple[TensorType, ...]:
        num_channels = self.decoder.in_channels * 4
        dtype = self.decoder.dtype
        device = self.decoder.device
        assert dtype is not None
        assert device is not None
        return (
            TensorType(
                dtype, shape=["batch", "seq", num_channels], device=device
            ),
            TensorType(DType.float32, shape=["latent_h"], device=DeviceRef.CPU()),
            TensorType(DType.float32, shape=["latent_w"], device=DeviceRef.CPU()),
            TensorType(dtype, shape=[num_channels], device=device),
            TensorType(dtype, shape=[num_channels], device=device),
        )

    def __call__(
        self,
        latents_bsc: TensorValue,
        h_carrier: TensorValue,
        w_carrier: TensorValue,
        bn_mean: TensorValue,
        bn_var: TensorValue,
    ) -> TensorValue:
        batch = latents_bsc.shape[0]
        channels = latents_bsc.shape[2]
        height = h_carrier.shape[0]
        width = w_carrier.shape[0]

        latents_bsc = ops.rebind(latents_bsc, [batch, height * width, channels])
        latents = ops.reshape(latents_bsc, (batch, height, width, channels))
        latents = ops.permute(latents, (0, 3, 1, 2))

        bn_mean = ops.reshape(bn_mean, (1, channels, 1, 1))
        bn_var = ops.reshape(bn_var, (1, channels, 1, 1))
        bn_std = ops.sqrt(bn_var + self.batch_norm_eps)
        latents = latents * bn_std + bn_mean

        latents = ops.reshape(latents, (batch, channels // 4, 2, 2, height, width))
        latents = ops.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = ops.reshape(
            latents, (batch, channels // 4, height * 2, width * 2)
        )
        decoded = self.decoder(latents, None)
        decoded = ops.permute(decoded, (0, 2, 3, 1))
        decoded = decoded * 0.5 + 0.5
        decoded = ops.max(decoded, 0.0)
        decoded = ops.min(decoded, 1.0)
        decoded = decoded * 255.0
        return ops.transfer_to(ops.cast(decoded, DType.uint8), DeviceRef.CPU())
