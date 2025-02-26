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
"""A vanilla opaque KV Cache optimized attention mechanism."""

from dataclasses import dataclass
from typing import Union

from max.dtype import DType
from max.graph import TensorValue, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    PagedKVCacheCollection,
)

from ..kernels import flash_attention, fused_qkv_matmul
from .interfaces import AttentionImpl, AttentionImplQKV


@dataclass
class Attention(AttentionImpl):
    def __call__(
        self,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        **kwargs,
    ) -> TensorValue:
        if isinstance(kv_collection, PagedKVCacheCollection):
            raise ValueError(
                "Paged attention not supported for Attention on non-ragged tensors."
            )

        if "attention_mask" not in kwargs:
            raise ValueError("attention_mask not passed as input to Attention")
        attention_mask = kwargs["attention_mask"]
        if attention_mask.dtype != x.dtype:
            msg = (
                "expected attention_mask and x to have the same dtype, but got"
                f" {attention_mask.dtype} and {x.dtype}, respectively."
            )
            raise ValueError(msg)

        # Get attributes from inputs
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Call into fused qkv matmul.
        xq = fused_qkv_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
            n_heads=self.n_heads,
        )

        xq = ops.reshape(
            xq,
            [
                batch_size,
                seq_len,
                self.n_heads,
                self.kv_params.head_dim,
            ],
        )

        # Calculate Flash Attention
        attn_out = flash_attention(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
            attention_mask=attention_mask,
            valid_lengths=kwargs["valid_lengths"],
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[batch_size, seq_len, -1])

        return self.wo(attn_out)


@dataclass
class AttentionQKV(AttentionImplQKV):
    def __call__(
        self,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        **kwargs,
    ) -> TensorValue:
        if isinstance(kv_collection, PagedKVCacheCollection):
            raise ValueError(
                "Paged attention not supported for Attention on non-ragged tensors."
            )

        if "attention_mask" not in kwargs:
            raise ValueError("attention_mask not passed as input to Attention")
        attention_mask = kwargs["attention_mask"]
        if attention_mask.dtype != x.dtype:
            msg = (
                "expected attention_mask and x to have the same dtype, but got"
                f" {attention_mask.dtype} and {x.dtype}, respectively."
            )
            raise ValueError(msg)

        wqkv = ops.concat((self.wq, self.wk, self.wv)).transpose(0, 1)
        wqkv = ops.cast(wqkv, x.dtype)

        # Get attributes from inputs
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Call into fused qkv matmul.
        xq = fused_qkv_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            kv_collection=kv_collection,
            layer_idx=ops.constant(self.layer_idx, DType.uint32),
            n_heads=self.n_heads,
        )

        xq = ops.reshape(
            xq,
            [
                batch_size,
                seq_len,
                self.n_heads,
                self.kv_params.head_dim,
            ],
        )

        # Calculate Flash Attention
        attn_out = flash_attention(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=ops.constant(self.layer_idx, DType.uint32),
            attention_mask=attention_mask,
            valid_lengths=kwargs["valid_lengths"],
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[batch_size, seq_len, -1])

        return self.wo(attn_out)
