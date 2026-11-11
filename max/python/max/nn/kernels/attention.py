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

"""Attention and KV cache kernels as ``Tensor`` ops.

These have no sharding rule: on distributed inputs each runs once per
device.
"""

from max import functional as F
from max.nn._graph_ops import flash_attention as _flash_attention
from max.nn._graph_ops import flash_attention_gpu as _flash_attention_gpu
from max.nn._graph_ops import flash_attention_ragged as _flash_attention_ragged
from max.nn._graph_ops import (
    flash_attention_ragged_gpu as _flash_attention_ragged_gpu,
)
from max.nn._graph_ops import (
    fused_chunked_qkv_rope_store_ragged as _fused_chunked_qkv_rope_store_ragged,
)
from max.nn._graph_ops import fused_qk_ragged_rope as _fused_qk_ragged_rope
from max.nn._graph_ops import rms_norm_key_cache as _rms_norm_key_cache
from max.nn._graph_ops import rope_ragged as _rope_ragged
from max.nn._graph_ops import (
    rope_split_store_ragged as _rope_split_store_ragged,
)
from max.nn._graph_ops import store_k_cache_ragged as _store_k_cache_ragged
from max.nn._graph_ops import (
    store_k_scale_cache_ragged as _store_k_scale_cache_ragged,
)
from max.nn._graph_ops import store_v_cache_ragged as _store_v_cache_ragged

flash_attention_gpu = F.functional(_flash_attention_gpu)
flash_attention_ragged = F.functional(_flash_attention_ragged)
flash_attention_ragged_gpu = F.functional(_flash_attention_ragged_gpu)
rope_split_store_ragged = F.functional(_rope_split_store_ragged)
rms_norm_key_cache = F.functional(_rms_norm_key_cache)
rope_ragged = F.functional(_rope_ragged)
store_k_cache_ragged = F.functional(_store_k_cache_ragged)
store_k_scale_cache_ragged = F.functional(_store_k_scale_cache_ragged)
store_v_cache_ragged = F.functional(_store_v_cache_ragged)
fused_qk_ragged_rope = F.functional(_fused_qk_ragged_rope)
fused_chunked_qkv_rope_store_ragged = F.functional(
    _fused_chunked_qkv_rope_store_ragged
)


flash_attention = F.functional(_flash_attention)


__all__ = [
    "flash_attention",
    "flash_attention_gpu",
    "flash_attention_ragged",
    "flash_attention_ragged_gpu",
    "fused_chunked_qkv_rope_store_ragged",
    "fused_qk_ragged_rope",
    "rms_norm_key_cache",
    "rope_ragged",
    "rope_split_store_ragged",
    "store_k_cache_ragged",
    "store_k_scale_cache_ragged",
    "store_v_cache_ragged",
]
