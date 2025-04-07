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
from max.graph import TensorType, TensorValue, ops


def l2_norm(x: TensorValue, eps=1e-6) -> TensorValue:
    weight = ops.constant(1, DType.float32).broadcast_to([x.shape[0]])
    if x.device:
        weight = weight.to(x.device)
    return ops.custom(
        "rms_norm",
        [x, weight, ops.constant(eps, x.dtype)],
        [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
    )[0].tensor
