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

"""LoRA Modules."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight
from max.graph.quantization import QuantizationEncoding
from max.nn.lora.interfaces import SupportsLoRA

from ..kernels import (
    sgmv_lora_kernel,
    sgmv_qkv_lora_kernel,
    sliced_add,
)
from ..kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from ..layer import Module


class LinearLoRA(Module, SupportsLoRA):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        max_num_loras: int,
        max_lora_rank: int,
        dtype: DType,
        device: DeviceRef,
        has_lora_bias: bool = False,
        name: str | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
    ):
        """Applies a linear transformation and LoRA to input:

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
        super().__init__()

        self.max_num_loras = max_num_loras
        self.max_lora_rank = max_lora_rank
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

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
        self.num_active_loras: TensorValue | None = None
        self.lora_end_idx: TensorValue | None = None
        self.batch_seq_len: TensorValue | None = None
        self.lora_grouped_offsets: TensorValue | None = None
        self.lora_ids_kv: TensorValue | None = None
        self.lora_grouped_offsets_kv: TensorValue | None = None

    def set_lora_batch_info(
        self,
        lora_ids: TensorValue,
        lora_ranks: TensorValue,
        lora_grouped_offsets: TensorValue,
        num_active_loras: TensorValue,
        lora_end_idx: TensorValue,
        batch_seq_len: TensorValue,
        lora_ids_kv: TensorValue,
        lora_grouped_offsets_kv: TensorValue,
    ) -> None:
        self.lora_ids = lora_ids
        self.lora_ranks = lora_ranks
        self.lora_grouped_offsets = lora_grouped_offsets
        self.num_active_loras = num_active_loras
        self.lora_end_idx = lora_end_idx
        self.batch_seq_len = batch_seq_len
        self.lora_ids_kv = lora_ids_kv
        self.lora_grouped_offsets_kv = lora_grouped_offsets_kv

    def __call__(self, x: TensorValue, y: TensorValue) -> TensorValue:
        if (
            self.lora_ids is None
            or self.lora_ranks is None
            or self.lora_grouped_offsets is None
            or self.num_active_loras is None
            or self.lora_end_idx is None
            or self.batch_seq_len is None
        ):
            raise ValueError(
                "'set_lora_batch_info' not called before executing forward pass."
            )

        y_lora = sgmv_lora_kernel(
            input=x,
            lora_a=self.lora_A,
            lora_b=self.lora_B,
            lora_ids=self.lora_ids,
            lora_ranks=self.lora_ranks,
            grouped_row_offsets=self.lora_grouped_offsets,
            max_lora_seq_len=self.in_dim,
            lora_end_idx=self.lora_end_idx,
            bias=self.lora_bias,
        )

        y = sliced_add(y, y_lora, self.lora_end_idx)

        return y


class QKVLinearLoRA(Module, SupportsLoRA):
    def __init__(
        self,
        in_dim: int,
        q_dim: int,
        kv_dim: int,
        max_num_loras: int,
        max_lora_rank: int,
        dtype: DType,
        device: DeviceRef,
        name: str | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
    ):
        super().__init__()

        self.max_num_loras = max_num_loras
        self.max_lora_rank = max_lora_rank
        self.in_dim = in_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim

        self.lora_A = Weight(
            name=f"{name}.lora_A.weight" if name else "lora_A.weight",
            dtype=dtype,
            shape=[max_num_loras, 3 * max_lora_rank, in_dim],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )

        self.lora_B_q = Weight(
            name=f"{name}.lora_B_q.weight" if name else "lora_B_q.weight",
            dtype=dtype,
            shape=[max_num_loras, q_dim, max_lora_rank],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )
        self.lora_B_kv = Weight(
            name=f"{name}.lora_B_kv.weight" if name else "lora_B_kv.weight",
            dtype=dtype,
            shape=[2 * max_num_loras, kv_dim, max_lora_rank],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )

        self.lora_ids: TensorValue | None = None
        self.lora_ranks: TensorValue | None = None
        self.num_active_loras: TensorValue | None = None
        self.lora_end_idx: TensorValue | None = None
        self.batch_seq_len: TensorValue | None = None
        self.lora_grouped_offsets: TensorValue | None = None
        self.lora_ids_kv: TensorValue | None = None
        self.lora_grouped_offsets_kv: TensorValue | None = None

    def set_lora_batch_info(
        self,
        lora_ids: TensorValue,
        lora_ranks: TensorValue,
        lora_grouped_offsets: TensorValue,
        num_active_loras: TensorValue,
        lora_end_idx: TensorValue,
        batch_seq_len: TensorValue,
        lora_ids_kv: TensorValue,
        lora_grouped_offsets_kv: TensorValue,
    ) -> None:
        self.lora_ids = lora_ids
        self.lora_ranks = lora_ranks
        self.lora_grouped_offsets = lora_grouped_offsets
        self.num_active_loras = num_active_loras
        self.lora_end_idx = lora_end_idx
        self.batch_seq_len = batch_seq_len
        self.lora_ids_kv = lora_ids_kv
        self.lora_grouped_offsets_kv = lora_grouped_offsets_kv

    def __call__(
        self,
        x: TensorValue,
        xq: TensorValue,
        kv_collection: PagedCacheValues,
        kv_params: KVCacheParams,
        input_row_offsets: TensorValue,
        layer_idx: TensorValue,
        max_seq_len: int,
    ) -> TensorValue:
        """Computes fused query, key, and value LoRAs with ragged input.

        Args:
            x (TensorValue): The input tensor of shape [total_tokens, hidden_dim].
            qkv_loras (list[LinearLoRA]): List of 3 LinearLoRA modules for Q, K, and V projections.
            input_row_offsets (TensorValue): 1D tensor indicating the start index of each sequence in `x`.
            kv_collection (PagedCacheValues):
                The key/value cache collection structure.
            layer_idx (TensorValue): Index of the current transformer layer (used for caching).

        Returns:
            TensorValue: The query projections.

        Raises:
            ValueError: If 'set_lora_batch_info' has not been called on the LoRAs.
        """
        if (
            self.lora_ids is None
            or self.lora_ranks is None
            or self.lora_grouped_offsets is None
            or self.num_active_loras is None
            or self.lora_end_idx is None
            or self.batch_seq_len is None
            or self.lora_ids_kv is None
            or self.lora_grouped_offsets_kv is None
        ):
            raise ValueError(
                "'set_lora_batch_info' not called before executing forward pass."
            )

        xq_lora = sgmv_qkv_lora_kernel(
            input=x,
            lora_a=self.lora_A,
            lora_b_q=self.lora_B_q,
            lora_b_kv=self.lora_B_kv,
            lora_ids=self.lora_ids,
            lora_ranks=self.lora_ranks,
            input_row_offsets=input_row_offsets,
            lora_grouped_offsets=self.lora_grouped_offsets,
            lora_end_idx=self.lora_end_idx,
            batch_seq_len=self.batch_seq_len,
            lora_ids_kv=self.lora_ids_kv,
            lora_grouped_offsets_kv=self.lora_grouped_offsets_kv,
            kv_collection=kv_collection,
            kv_params=kv_params,
            layer_idx=layer_idx,
            max_lora_seq_len=max_seq_len,
            max_rank=self.max_lora_rank,
            bias=None,
        )

        xq = sliced_add(xq, xq_lora, self.lora_end_idx)

        return xq
