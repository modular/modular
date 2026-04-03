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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import Module


class FusedLeakyReLU(Module):
    """FusedLeakyReLU: channel bias + leaky_relu(0.2) * sqrt(2).

    Weight key: ``bias`` (matching diffusers ``act_fn.bias``).
    """

    def __init__(
        self,
        channels: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.bias = Weight("bias", dtype, [channels], device)

    def __call__(self, x: TensorValue) -> TensorValue:
        x = x + self.bias
        # leaky_relu(x, 0.2) * sqrt(2)
        # Use x*0 to get a zero tensor matching shape/dtype/device.
        zero = x * 0
        pos_scale = float(2.0**0.5)
        neg_scale = float(0.2 * 2.0**0.5)
        return ops.where(x > zero, x * pos_scale, x * neg_scale)


class MotionConv2d(Module):
    """Conv2d with pre-baked weight scaling, optional FIR blur, optional activation.

    Expects NHWC input. Weight stored in RSCF layout ``[K, K, C_in, C_out]``
    with ``1/sqrt(fan_in)`` scaling already applied at load time.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        *,
        has_blur: bool = False,
        has_activation: bool = True,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self._stride = (stride, stride)
        self._padding = (padding, padding, padding, padding)
        self._has_blur = has_blur
        self._has_activation = has_activation

        # Weight in RSCF: [K, K, C_in, C_out] (scale already baked in)
        self.weight = Weight(
            "weight",
            dtype,
            [kernel_size, kernel_size, in_channels, out_channels],
            device,
        )

        if has_activation:
            self.act_fn = FusedLeakyReLU(
                out_channels, dtype=dtype, device=device
            )

        if has_blur:
            self._blur_in_ch = in_channels
            # Blur filter as a Weight (gets placed on correct device).
            # Values are injected during weight preprocessing.
            self.blur_filter = Weight(
                "blur_filter",
                dtype,
                [4, 4, 1, in_channels],
                device,
            )
            # Blur padding: p = (blur_k_len - stride) + (kernel_size - 1)
            blur_k_len = 4
            p = (blur_k_len - stride) + (kernel_size - 1)
            ph_before = (p + 1) // 2
            ph_after = p // 2
            self._blur_padding = (ph_before, ph_after, ph_before, ph_after)

    def __call__(self, x: TensorValue) -> TensorValue:
        if self._has_blur:
            x = ops.conv2d(
                x,
                self.blur_filter,
                stride=(1, 1),
                padding=self._blur_padding,
                groups=self._blur_in_ch,
            )

        x = ops.conv2d(
            x, self.weight, stride=self._stride, padding=self._padding
        )

        if self._has_activation:
            x = self.act_fn(x)
        return x


class MotionResBlock(Module):
    """Residual block: ``(conv2(conv1(x)) + skip(x)) / sqrt(2)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        # conv1: 3×3, stride=1, padding=1, activation
        self.conv1 = MotionConv2d(
            in_channels,
            in_channels,
            3,
            stride=1,
            padding=1,
            has_blur=False,
            has_activation=True,
            dtype=dtype,
            device=device,
        )
        # conv2: 3×3, stride=2, padding=0, blur + activation
        self.conv2 = MotionConv2d(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=0,
            has_blur=True,
            has_activation=True,
            dtype=dtype,
            device=device,
        )
        # conv_skip: 1×1, stride=2, padding=0, blur, no activation
        self.conv_skip = MotionConv2d(
            in_channels,
            out_channels,
            1,
            stride=2,
            padding=0,
            has_blur=True,
            has_activation=False,
            dtype=dtype,
            device=device,
        )
        self.inv_sqrt2 = float(1.0 / 2.0**0.5)

    def __call__(self, x: TensorValue) -> TensorValue:
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_skip = self.conv_skip(x)
        return (x_out + x_skip) * self.inv_sqrt2
