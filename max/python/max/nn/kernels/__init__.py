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
"""MAX kernels as :class:`~max.tensor.Tensor` ops.

Each op wraps a graph kernel with :func:`~max.functional.functional` and,
where it has one, the sharding rule that says how it runs on distributed
inputs. The submodules group them by family.
"""

from .attention import (
    flash_attention,
    flash_attention_gpu,
    flash_attention_ragged,
    flash_attention_ragged_gpu,
    fused_chunked_qkv_rope_store_ragged,
    fused_qk_ragged_rope,
    rms_norm_key_cache,
    rope_ragged,
    rope_split_store_ragged,
    store_k_cache_ragged,
    store_k_scale_cache_ragged,
    store_v_cache_ragged,
)
from .hyper_connection import hyper_connection_gates, mhc_post, mhc_pre
from .mla import (
    cp_merge_topk,
    flare_mla_prefill_plan,
    mla_decode_graph,
    mla_decode_graph_cp,
    mla_fp8_index_top_k,
    mla_fp8_index_top_k_physical,
    mla_fp8_index_top_k_physical_cp,
    mla_prefill_decode_graph,
    mla_prefill_graph,
    store_mla_latent_ragged,
)
from .moe import (
    fused_silu,
    fused_silu_quantized,
    grouped_matmul_ragged,
    moe_create_indices,
    moe_create_indices_packed,
    moe_finalize,
    moe_router_group_limited,
    moe_sigmoid_gemv_router,
    moe_single_group_router_packed,
)
from .norm import mtp_eh_norm
from .quant import (
    block_scales_interleave,
    dynamic_block_scaled_matmul,
    dynamic_block_scaled_matmul_amd,
    dynamic_scaled_matmul,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_ffn_blockwise_fp8_amd,
    grouped_matmul_block_scaled,
    grouped_matmul_blocked_swiglu,
    grouped_quantize_dynamic_block_scaled,
    moe_fp8_host_usage_stats,
    mxfp4_dequant,
    mxfp8_dequant,
    nvfp4_dequant,
    preshuffle_blockwise_fp8_b,
    quantize_dynamic_block_scaled,
    quantize_dynamic_block_scaled_mxfp4,
    quantize_dynamic_scaled_float8,
    silu_quantize_fp8,
    stack_blockwise_mega_ffn_b,
)

__all__ = [
    "block_scales_interleave",
    "cp_merge_topk",
    "dynamic_block_scaled_matmul",
    "dynamic_block_scaled_matmul_amd",
    "dynamic_scaled_matmul",
    "flare_mla_prefill_plan",
    "flash_attention",
    "flash_attention_gpu",
    "flash_attention_ragged",
    "flash_attention_ragged_gpu",
    "fused_chunked_qkv_rope_store_ragged",
    "fused_qk_ragged_rope",
    "fused_silu",
    "fused_silu_quantized",
    "grouped_dynamic_scaled_fp8_matmul",
    "grouped_ffn_blockwise_fp8_amd",
    "grouped_matmul_block_scaled",
    "grouped_matmul_blocked_swiglu",
    "grouped_matmul_ragged",
    "grouped_quantize_dynamic_block_scaled",
    "hyper_connection_gates",
    "mhc_post",
    "mhc_pre",
    "mla_decode_graph",
    "mla_decode_graph_cp",
    "mla_fp8_index_top_k",
    "mla_fp8_index_top_k_physical",
    "mla_fp8_index_top_k_physical_cp",
    "mla_prefill_decode_graph",
    "mla_prefill_graph",
    "moe_create_indices",
    "moe_create_indices_packed",
    "moe_finalize",
    "moe_fp8_host_usage_stats",
    "moe_router_group_limited",
    "moe_sigmoid_gemv_router",
    "moe_single_group_router_packed",
    "mtp_eh_norm",
    "mxfp4_dequant",
    "mxfp8_dequant",
    "nvfp4_dequant",
    "preshuffle_blockwise_fp8_b",
    "quantize_dynamic_block_scaled",
    "quantize_dynamic_block_scaled_mxfp4",
    "quantize_dynamic_scaled_float8",
    "rms_norm_key_cache",
    "rope_ragged",
    "rope_split_store_ragged",
    "silu_quantize_fp8",
    "stack_blockwise_mega_ffn_b",
    "store_k_cache_ragged",
    "store_k_scale_cache_ragged",
    "store_mla_latent_ragged",
    "store_v_cache_ragged",
]
