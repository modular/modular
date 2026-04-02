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

from __future__ import annotations

import operator
from functools import reduce
from typing import Literal

from max.driver import CPU
from max.dtype import DType
from max.experimental import random
from max.experimental.nn import Module, PinnedDeviceTensor
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.graph.ops import reshape
from max.nn.quant_config import QuantConfig
from max.nn.quant_ops import quantized_matmul


class NVFP4Linear(Module[[Tensor], Tensor]):
    """ModuleV3 Linear layer using NVFP4 block-scaled quantization."""

    weight: Tensor
    weight_scale: Tensor
    weight_scale_2: PinnedDeviceTensor
    input_scale: PinnedDeviceTensor
    bias: Tensor | Literal[0]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        quant_config: QuantConfig,
        bias: bool = False,
    ):
        packed_k = in_dim // 2

        # Placeholder tensors -- shapes/dtypes must match the loaded weights.
        self.weight = random.normal([out_dim, packed_k]).cast(DType.uint8)
        self.weight_scale = random.normal([out_dim, packed_k // 8]).cast(
            DType.float8_e4m3fn
        )
        self.weight_scale_2 = Tensor.full(
            [], 1.0, dtype=DType.float32, device=CPU()
        )
        self.input_scale = Tensor.full(
            [], 1.0, dtype=DType.float32, device=CPU()
        )
        self.bias = random.normal([out_dim]) if bias else 0
        self._quant_config = quant_config

    def forward(self, x: Tensor) -> Tensor:
        xv = TensorValue(x)

        # `matmul_float4` requires rank-2 input [M, K].  Flatten leading dims
        # (e.g. [batch, seq, hidden] -> [batch*seq, hidden]) and restore after.
        leading_dims = list(xv.shape[:-1])
        k_dim = xv.shape[-1]
        if xv.rank > 2:
            m_dim = reduce(operator.mul, leading_dims)
            xv = reshape(xv, [m_dim, k_dim])

        result_val = quantized_matmul(
            xv,
            TensorValue(self.weight),
            TensorValue(self.weight_scale),
            TensorValue(self.input_scale),
            self._quant_config,
            weight_scale_2=TensorValue(self.weight_scale_2),
            scales_pre_interleaved=True,
        )

        if len(leading_dims) > 1:
            out_dim = result_val.shape[-1]
            result_val = reshape(result_val, [*leading_dims, out_dim])

        return Tensor.from_graph_value(result_val) + self.bias
