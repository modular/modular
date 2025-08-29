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

"""LoRA Modules."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight
from max.graph.quantization import QuantizationEncoding
from max.nn.float8_config import Float8Config

from ..kernels import sgmv_lora_kernel
from ..linear import Linear


class LinearLoRA(Linear):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        max_num_loras: int,
        max_lora_rank: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        has_lora_bias: bool = False,
        name: str | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
        float8_config: Float8Config | None = None,
    ):
        """
        Applies a linear transformation and LoRA to input:

        :math:`y_l = (xA^T) @ B^T`.
        :math:`y = (xW^T + b) + y_l`

        Example:

        .. code-block:: python

            linear_layer = LinearLoRA(
                in_dim=256,
                out_dim=128,
                max_lora_rank=16,
                max_num_loras=100,
                dtype=dtype.float32,
                device=DeviceRef.GPU(),
                has_bias=True,
                has_lora_bias=True,
                name="lora_linear"
            )

            lora_ids: TensorValue # shape: [max_num_loras,]
            lora_ranks: TensorValue # shape: [max_num_loras,]
            input_row_offsets: TensorValue
            linear_layer.set_lora_batch_info(lora_ids, lora_ranks, input_row_offsets)

            # Input tensor of shape: [batch, ..., 256]
            input_tensor: TensorValue
            output = linear_layer(input_tensor)
        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
            quantization_encoding=quantization_encoding,
            float8_config=float8_config,
        )
        self.max_num_loras = max_num_loras
        self.max_lora_rank = max_lora_rank
        self.in_dim = in_dim

        self.lora_A = Weight(
            name=f"{name}.lora_A.weight" if name else "lora_A.weight",
            dtype=dtype,
            shape=[max_num_loras, max_lora_rank, in_dim],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )
        self.lora_B = Weight(
            name=f"{name}.lora_B.weight" if name else "lora_B.weight",
            dtype=dtype,
            shape=[max_num_loras, out_dim, max_lora_rank],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )
        self.lora_bias = (
            Weight(
                name=f"{name}.lora.bias" if name else "lora.bias",
                dtype=dtype,
                shape=[max_num_loras, out_dim],
                device=device,
                quantization_encoding=quantization_encoding,
                _has_alias=True,
            )
            if has_lora_bias
            else None
        )
        self.lora_ids: TensorValue | None = None
        self.lora_ranks: TensorValue | None = None

    def set_lora_batch_info(
        self,
        lora_ids: TensorValue,
        lora_ranks: TensorValue,
        lora_grouped_offsets: TensorValue,
    ) -> None:
        self.lora_ids = lora_ids
        self.lora_ranks = lora_ranks
        self.lora_grouped_offsets = lora_grouped_offsets

    def apply_lora(self, x: TensorValue) -> TensorValue:
        y = self(x)

        if self.lora_ids is None or self.lora_ranks is None:
            raise ValueError(
                "'set_lora_batch_info' not called before executing forward pass."
            )

        return y + sgmv_lora_kernel(
            input=x,
            lora_a=self.lora_A,
            lora_b=self.lora_B,
            lora_ids=self.lora_ids,
            lora_ranks=self.lora_ranks,
            grouped_row_offsets=self.lora_grouped_offsets,
            max_lora_seq_len=self.in_dim,
            bias=self.lora_bias,
        )
