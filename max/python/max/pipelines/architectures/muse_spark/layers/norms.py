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

"""Muse Spark RMSNorm variants for the ModuleV3 API."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn import Module
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping


def weightless_rms_norm(x: Tensor, eps: float) -> Tensor:
    """Normalizes ``x`` over its last axis with no learned scale."""
    # Built on x's mesh, so the norm runs on whichever devices x lives on.
    ones = Tensor.ones(
        [x.shape[-1]], dtype=x.dtype, device=DeviceMapping.replicated(x.mesh)
    )
    return rms_norm(x, ones, eps, multiply_before_cast=True)


def input_scaled_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Scales ``x`` by ``1 + weight``, rounds to ``x``'s dtype, then normalizes."""
    return weightless_rms_norm(x * (1.0 + weight), eps)


def _routed_rows_rms_norm(
    x: TensorValue, weight: TensorValue, expert_offsets: TensorValue, eps: float
) -> TensorValue:
    return ops.custom(
        "norm.routed_rows_rms_norm",
        device=x.device,
        values=[
            x,
            weight,
            expert_offsets,
            ops.constant(eps, DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[x.type],
    )[0].tensor


#: :func:`input_scaled_rms_norm` of the rows below ``expert_offsets[-1]``;
#: the rows past it are left unwritten.
routed_rows_rms_norm = F.functional(_routed_rows_rms_norm)


def _routed_rows_rms_norm_finalize(
    x: TensorValue,
    weight: TensorValue,
    expert_offsets: TensorValue,
    restore_order: TensorValue,
    router_weight: TensorValue,
    eps: float,
    out_dtype: DType = DType.float32,
) -> TensorValue:
    return ops.custom(
        "norm.routed_rows_rms_norm_finalize",
        device=x.device,
        values=[
            x,
            weight,
            expert_offsets,
            restore_order,
            router_weight,
            ops.constant(eps, DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                out_dtype, [router_weight.shape[0], x.shape[1]], x.device
            )
        ],
    )[0].tensor


#: ``moe_finalize`` of :func:`routed_rows_rms_norm`'s rows, summed in float32
#: and stored as ``out_dtype``, a slot past ``expert_offsets[-1]`` adding
#: nothing.
routed_rows_rms_norm_finalize = F.functional(_routed_rows_rms_norm_finalize)


class WeightlessRMSNorm(Module[[Tensor], Tensor]):
    """RMSNorm without a scale (qk, o, embedding and post-attention norms)."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes ``x`` by its root mean square."""
        return weightless_rms_norm(x, self.eps)


class CenteredRMSNorm(Module[[Tensor], Tensor]):
    """Normalizes, then scales by ``1 + weight``."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = Tensor.zeros([dim])
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes ``x`` by its centred root mean square."""
        # The reference rounds 1 + weight to the weight's dtype before the
        # fp32 multiply; weight_offset=1 would add it in fp32 instead.
        return rms_norm(
            x, 1.0 + self.weight, self.eps, multiply_before_cast=True
        )


class InputScaledRMSNorm(Module[[Tensor], Tensor]):
    """Scales by ``1 + weight``, rounds to ``x``'s dtype, then normalizes."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = Tensor.zeros([dim])
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes ``x`` with the input-scaled RMS norm kernel."""
        return input_scaled_rms_norm(x, self.weight, self.eps)
