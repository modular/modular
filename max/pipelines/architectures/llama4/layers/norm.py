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

"""Llama4 normalization."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops


def l2_norm(x: TensorValue, eps=1e-6) -> TensorValue:
    """Computes the L2 norm of the input."""
    weight = ops.constant(
        1, DType.float32, device=DeviceRef.CPU()
    ).broadcast_to([x.shape[-1]])
    if x.device:
        weight = weight.to(x.device)
    original_dtype = x.dtype
    x = x.cast(DType.float32)
    return ops.custom(
        "rms_norm",
        x.device,
        [
            x,
            weight,
            ops.constant(eps, x.dtype, device=DeviceRef.CPU()),
            ops.constant(
                0.0, x.dtype, device=DeviceRef.CPU()
            ),  # weight_offset = 0.0
        ],
        [TensorType(dtype=DType.float32, shape=x.shape, device=x.device)],
        parameters={"multiply_before_cast": True},
    )[0].tensor.cast(original_dtype)
