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

import math

import max.experimental.functional as F
import max.nn.module_v3 as nn
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import TensorType

from ..model_config import LTX2VocoderConfig


class ResBlock(nn.Module[[Tensor], Tensor]):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: tuple[int, ...] = (1, 3, 5),
        leaky_relu_negative_slope: float = 0.1,
        padding_mode: str = "same",
    ):
        super().__init__()
        self.dilations = dilations
        self.negative_slope = leaky_relu_negative_slope

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_mode,
                )
                for dilation in dilations
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=1,
                    padding=padding_mode,
                )
                for _ in range(len(dilations))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=False):
            xt = F.max(0, x) + self.negative_slope * F.min(0, x)
            xt = conv1(xt)
            xt = F.max(0, xt) + self.negative_slope * F.min(0, xt)
            xt = conv2(xt)
            x = x + xt
        return x


class LTX2Vocoder(nn.Module[[Tensor, bool], Tensor]):
    r"""
    LTX 2.0 vocoder for converting generated mel spectrograms back to audio waveforms.
    """

    def __init__(
        self,
        config: LTX2VocoderConfig,
    ):
        super().__init__()
        self.config = config
        self.num_upsample_layers = len(config.upsample_kernel_sizes)
        self.resnets_per_upsample = len(config.resnet_kernel_sizes)
        self.out_channels = config.out_channels
        self.total_upsample_factor = math.prod(config.upsample_factors)
        self.negative_slope = config.leaky_relu_negative_slope

        if self.num_upsample_layers != len(config.upsample_factors):
            raise ValueError(
                f"`upsample_kernel_sizes` and `upsample_factors` should be lists of the same length but are length"
                f" {self.num_upsample_layers} and {len(config.upsample_factors)}, respectively."
            )

        if self.resnets_per_upsample != len(config.resnet_dilations):
            raise ValueError(
                f"`resnet_kernel_sizes` and `resnet_dilations` should be lists of the same length but are length"
                f" {self.resnets_per_upsample} and {len(config.resnet_dilations)}, respectively."
            )

        self.conv_in = nn.Conv1d(
            in_channels=config.in_channels,
            out_channels=config.hidden_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsamplers = nn.ModuleList()
        self.resnets = nn.ModuleList()
        input_channels = config.hidden_channels
        for stride, kernel_size in zip(
            config.upsample_factors, config.upsample_kernel_sizes, strict=False
        ):
            output_channels = input_channels // 2
            self.upsamplers.append(
                nn.ConvTranspose1d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

            for kernel_size, dilations in zip(
                config.resnet_kernel_sizes,
                config.resnet_dilations,
                strict=False,
            ):
                self.resnets.append(
                    ResBlock(
                        output_channels,
                        kernel_size,
                        dilations=dilations,
                        leaky_relu_negative_slope=config.leaky_relu_negative_slope,
                    )
                )
            input_channels = output_channels

        self.conv_out = nn.Conv1d(
            in_channels=output_channels,
            out_channels=config.out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

    def input_types(self) -> tuple[TensorType, ...]:  # type: ignore[override]
        """Define input tensor types for the model."""
        # Hardcoded for num_frames=121, frame_rate=24:
        #   out_channels (audio decoder output) = 2
        #   time = audio_num_frames * LATENT_DOWNSAMPLE_FACTOR - (LATENT_DOWNSAMPLE_FACTOR - 1)
        #        = 126 * 4 - 3 = 501  (causal audio VAE trims LATENT_DOWNSAMPLE_FACTOR-1 frames)
        #   num_mel_bins = 64
        # Use float32 because the cuDNN conv_transpose kernel hardcodes
        # CUDNN_DATA_FLOAT descriptors.
        hidden_states_type = TensorType(
            DType.float32,
            shape=[1, self.config.out_channels, 501, 64],
            device=self.config.device,
        )
        return (hidden_states_type,)

    def forward(self, hidden_states: Tensor, time_last: bool = False) -> Tensor:
        r"""
        Forward pass of the vocoder.

        Args:
            hidden_states (`Tensor`):
                Input Mel spectrogram tensor of shape `(batch_size, num_channels, time, num_mel_bins)` if `time_last`
                is `False` (the default) or shape `(batch_size, num_channels, num_mel_bins, time)` if `time_last` is
                `True`.
            time_last (`bool`, *optional*, defaults to `False`):
                Whether the last dimension of the input is the time/frame dimension or the Mel bins dimension.

        Returns:
            `Tensor`:
                Audio waveform tensor of shape (batch_size, out_channels, audio_length)
        """
        # Entire vocoder runs in float32 because cuDNN conv_transpose
        # kernel hardcodes CUDNN_DATA_FLOAT descriptors.

        # Ensure that the time/frame dimension is last
        if not time_last:
            hidden_states = hidden_states.transpose(2, 3)
        # Combine channels and frequency (mel bins) dimensions
        hidden_states = F.flatten(hidden_states, 1, 2)

        hidden_states = self.conv_in(hidden_states)

        for i in range(self.num_upsample_layers):
            hidden_states = F.max(
                0, hidden_states
            ) + self.negative_slope * F.min(0, hidden_states)
            hidden_states = self.upsamplers[i](hidden_states)

            # Run all resnets in parallel on hidden_states
            start = i * self.resnets_per_upsample
            end = (i + 1) * self.resnets_per_upsample
            resnet_outputs = F.stack(
                [self.resnets[j](hidden_states) for j in range(start, end)],
                axis=0,
            )

            hidden_states = F.mean(resnet_outputs, axis=0).squeeze(0)

        # NOTE: unlike the first leaky ReLU, this leaky ReLU is set to use the default F.leaky_relu negative slope of
        # 0.01 (whereas the others usually use a slope of 0.1). Not sure if this is intended
        hidden_states = F.max(0, hidden_states) + 0.01 * F.min(0, hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = F.tanh(hidden_states)

        return hidden_states
