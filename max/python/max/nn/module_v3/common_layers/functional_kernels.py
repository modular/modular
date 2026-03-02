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

from max.experimental import functional as F
from max.graph import ops
from max.nn.kernels import (
    flash_attention_ragged as _flash_attention_ragged,
)
from max.nn.kernels import fused_qk_ragged_rope as _fused_qk_ragged_rope
from max.nn.kernels import (
    fused_qkv_ragged_matmul as _fused_qkv_ragged_matmul,
)
from max.nn.kernels import (
    grouped_matmul_ragged as _grouped_matmul_ragged,
)
from max.nn.kernels import moe_create_indices as _moe_create_indices
from max.nn.kernels import rms_norm_key_cache as _rms_norm_key_cache

grouped_matmul_ragged = F.functional(_grouped_matmul_ragged)
moe_create_indices = F.functional(_moe_create_indices)

inplace_custom = F.functional(ops.inplace_custom)
flash_attention_ragged = F.functional(_flash_attention_ragged)
fused_qkv_ragged_matmul = F.functional(_fused_qkv_ragged_matmul)
fused_qk_ragged_rope = F.functional(_fused_qk_ragged_rope)
rms_norm_key_cache = F.functional(_rms_norm_key_cache)

__all__ = [
    "flash_attention_ragged",
    "fused_qk_ragged_rope",
    "fused_qkv_ragged_matmul",
    "grouped_matmul_ragged",
    "moe_create_indices",
    "rms_norm_key_cache",
]
