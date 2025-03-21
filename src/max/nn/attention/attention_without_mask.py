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
"""An opaque KV Cache optimized vanilla attention mechanism, with Mask Variants provided inside the Kernel."""

from dataclasses import dataclass
from typing import Union

from max.graph import TensorValue, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    PagedKVCacheCollection,
)

from ..kernels import (
    MHAMaskVariant,
    flash_attention_ragged,
    fused_qkv_ragged_matmul,
)
from .interfaces import AttentionImpl


@dataclass
class AttentionWithoutMask(AttentionImpl):
    mask_variant: MHAMaskVariant

    def __call__(
        self,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        **kwargs,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
            n_heads=self.n_heads,
        )

        # Reshape for flash attention.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Calculate Flash Attention.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=self.mask_variant,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.wo(attn_out)
