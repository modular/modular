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


class CausalConv1d(Module):
    """Causal 1D convolution with replicate-left padding.

    This is intentionally kept as a custom kernel path instead of raw
    ``max.nn.Conv1D`` because:
    - this unfold+matmul implementation has been observed to be faster for
      WAN-Animate face-encoder workloads.
    - the face encoder requires replicate-left causal padding (Conv1D uses
      zero padding semantics), and
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # Weight as [k * in_ch, out_ch] for matmul-based conv1d
        self.weight = Weight(
            "weight",
            dtype,
            [kernel_size * in_channels, out_channels],
            device,
        )
        self.bias = Weight("bias", dtype, [out_channels], device)

    def __call__(self, x: TensorValue) -> TensorValue:
        # x: [B, T, C_in]
        # Implement causal conv1d via unfold + matmul (no Conv2d layout issues)
        pad_size = self.kernel_size - 1
        if pad_size > 0:
            first_elem = x[:, :1, :]
            padding = ops.broadcast_to(
                first_elem,
                [x.shape[0], pad_size, x.shape[2]],
            )
            x = ops.concat([padding, x], axis=1)

        if self.kernel_size == 3 and self.stride == 1:
            x0 = x[:, :-2, :]
            x1 = x[:, 1:-1, :]
            x2 = x[:, 2:, :]
            unfolded = ops.concat([x0, x1, x2], axis=2)
        elif self.kernel_size == 3 and self.stride == 2:
            x0 = x[:, :-2:2, :]
            x1 = x[:, 1:-1:2, :]
            x2 = x[:, 2::2, :]
            unfolded = ops.concat([x0, x1, x2], axis=2)
        else:
            raise NotImplementedError(
                f"CausalConv1d: kernel_size={self.kernel_size}, stride={self.stride} not implemented"
            )

        return ops.matmul(unfolded, self.weight) + self.bias
