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
"""KV Cache transfer kernels for external cache integration (e.g., LMCache).

This module provides GPU kernels to efficiently transfer KV cache data between
MAX's paged KV cache format and external contiguous formats like LMCache's KV_2LTD.

MAX PagedKVCacheCollection Layout:
    [total_num_blocks, kv_dim, num_layers, page_size, num_heads, head_size]
    where kv_dim = 2 (K and V) for standard attention, 1 for MLA

External Contiguous Layout (KV_2LTD):
    [kv_dim, num_layers, num_tokens, hidden_dim]
    where hidden_dim = num_heads * head_size

The kernels use slot_mapping to transfer data between the formats.
slot_mapping[token_idx] gives the physical slot in the paged cache:
    block_id = slot // page_size
    offset_in_block = slot % page_size
"""

from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from runtime.tracing import Trace, TraceLevel


# ===----------------------------------------------------------------------=== #
# Offload from MAX paged cache to external contiguous buffer
# ===----------------------------------------------------------------------=== #


fn _lmcache_offload_kernel[
    dtype: DType,
    page_size: Int,
    num_kv_heads: Int,
    head_dim: Int,
    kv_dim: Int,  # 2 for standard, 1 for MLA
](
    output: LayoutTensor[mut=True, dtype, Layout.row_major[4](), MutAnyOrigin],
    paged_cache: LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin],
    slot_mapping: LayoutTensor[
        DType.int64, Layout.row_major[1](), MutAnyOrigin
    ],
    start_token: Int,
    num_tokens: Int,
    num_layers: Int,
):
    """GPU kernel to offload KV data from MAX paged cache to external format.

    Grid layout:
        - block_idx.x: token index (0 to num_tokens-1)
        - block_idx.y: layer index
        - block_idx.z: kv index (0=K, 1=V for standard; 0 only for MLA)
    Block layout:
        - thread_idx.x: position within hidden_dim (strided)
    """
    comptime hidden_dim_total = num_kv_heads * head_dim

    var token_idx = Int(block_idx.x)
    var layer_idx = Int(block_idx.y)
    var kv_idx = Int(block_idx.z)

    if token_idx >= num_tokens or kv_idx >= kv_dim:
        return

    var slot = Int(slot_mapping[start_token + token_idx])

    if slot < 0:
        return

    var block_id = slot // page_size
    var offset_in_block = slot % page_size

    var thread_id = Int(thread_idx.x)
    var num_threads = Int(block_dim.x)

    for hidden_idx in range(thread_id, hidden_dim_total, num_threads):
        var head_idx = hidden_idx // head_dim
        var head_dim_idx = hidden_idx % head_dim

        # Read from paged cache: [total_num_blocks, kv_dim, num_layers, page_size, num_heads, head_dim]
        var src_val = paged_cache[
            block_id, kv_idx, layer_idx, offset_in_block, head_idx, head_dim_idx
        ]

        # Write to output: [kv_dim, num_layers, num_tokens, hidden_dim]
        output[kv_idx, layer_idx, token_idx, hidden_idx] = src_val


# ===----------------------------------------------------------------------=== #
# Onload from external contiguous buffer to MAX paged cache
# ===----------------------------------------------------------------------=== #


fn _lmcache_onload_kernel[
    dtype: DType,
    page_size: Int,
    num_kv_heads: Int,
    head_dim: Int,
    kv_dim: Int,
](
    paged_cache: LayoutTensor[
        mut=True, dtype, Layout.row_major[6](), MutAnyOrigin
    ],
    input: LayoutTensor[dtype, Layout.row_major[4](), MutAnyOrigin],
    slot_mapping: LayoutTensor[
        DType.int64, Layout.row_major[1](), MutAnyOrigin
    ],
    start_token: Int,
    num_tokens: Int,
    num_layers: Int,
):
    """GPU kernel to onload KV data from external format to MAX paged cache.

    Grid layout:
        - block_idx.x: token index (0 to num_tokens-1)
        - block_idx.y: layer index
        - block_idx.z: kv index
    Block layout:
        - thread_idx.x: position within hidden_dim (strided)
    """
    comptime hidden_dim_total = num_kv_heads * head_dim

    var token_idx = Int(block_idx.x)
    var layer_idx = Int(block_idx.y)
    var kv_idx = Int(block_idx.z)

    if token_idx >= num_tokens or kv_idx >= kv_dim:
        return

    # Get physical slot from mapping
    var slot = Int(slot_mapping[start_token + token_idx])

    if slot < 0:
        return

    var block_id = slot // page_size
    var offset_in_block = slot % page_size

    var thread_id = Int(thread_idx.x)
    var num_threads = Int(block_dim.x)

    for hidden_idx in range(thread_id, hidden_dim_total, num_threads):
        var head_idx = hidden_idx // head_dim
        var head_dim_idx = hidden_idx % head_dim

        # Read from input: [kv_dim, num_layers, num_tokens, hidden_dim]
        var src_val = input[kv_idx, layer_idx, token_idx, hidden_idx]

        # Write to paged cache: [total_num_blocks, kv_dim, num_layers, page_size, num_heads, head_dim]
        paged_cache[
            block_id, kv_idx, layer_idx, offset_in_block, head_idx, head_dim_idx
        ] = src_val


fn lmcache_offload[
    dtype: DType,
    page_size: Int,
    num_kv_heads: Int,
    head_dim: Int,
    kv_dim: Int,
    target: StaticString = "gpu",
](
    output: LayoutTensor[mut=True, dtype, Layout.row_major[4](), MutAnyOrigin],
    paged_cache: LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin],
    slot_mapping: LayoutTensor[
        DType.int64, Layout.row_major[1](), MutAnyOrigin
    ],
    start_token: Int,
    end_token: Int,
    ctx: DeviceContext,
) raises:
    """Offload KV cache data from MAX paged format to external contiguous format.

    Parameters:
        dtype: Data type of the cache.
        page_size: Number of tokens per page in the paged cache.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension of each attention head.
        kv_dim: KV dimension (2 for standard K/V, 1 for MLA).
        target: Target device ("gpu" or "cpu").

    Args:
        output: Destination tensor [kv_dim, num_layers, num_tokens, hidden_dim].
        paged_cache: Source tensor [total_num_blocks, kv_dim, num_layers, page_size, num_heads, head_dim].
        slot_mapping: Token to slot mapping [total_tokens].
        start_token: Starting token index in slot_mapping.
        end_token: Ending token index (exclusive) in slot_mapping.
        ctx: Device context for kernel launch.
    """
    var num_tokens = end_token - start_token
    if num_tokens <= 0:
        return

    var num_layers = paged_cache.dim[2]()
    var threads_per_block = min(num_kv_heads * head_dim, 256)
    var grid_x = num_tokens
    var grid_y = num_layers
    var grid_z = kv_dim

    with Trace[TraceLevel.OP, target=target](
        "mo.lmcache_offload.page_"
        + String(page_size)
        + ".nhead_"
        + String(num_kv_heads)
        + ".hdim_"
        + String(head_dim),
    ):
        comptime kernel = _lmcache_offload_kernel[
            dtype, page_size, num_kv_heads, head_dim, kv_dim
        ]
        ctx.enqueue_function[kernel, kernel](
            output,
            paged_cache,
            slot_mapping,
            start_token,
            num_tokens,
            num_layers,
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(threads_per_block, 1, 1),
        )


fn lmcache_onload[
    dtype: DType,
    page_size: Int,
    num_kv_heads: Int,
    head_dim: Int,
    kv_dim: Int,
    target: StaticString = "gpu",
](
    paged_cache: LayoutTensor[
        mut=True, dtype, Layout.row_major[6](), MutAnyOrigin
    ],
    input: LayoutTensor[dtype, Layout.row_major[4](), MutAnyOrigin],
    slot_mapping: LayoutTensor[
        DType.int64, Layout.row_major[1](), MutAnyOrigin
    ],
    start_token: Int,
    end_token: Int,
    ctx: DeviceContext,
) raises:
    """Onload KV cache data from external contiguous format to MAX paged format.

    Parameters:
        dtype: Data type of the cache.
        page_size: Number of tokens per page in the paged cache.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension of each attention head.
        kv_dim: KV dimension (2 for standard K/V, 1 for MLA).
        target: Target device ("gpu" or "cpu").

    Args:
        paged_cache: Destination tensor [total_num_blocks, kv_dim, num_layers, page_size, num_heads, head_dim].
        input: Source tensor [kv_dim, num_layers, num_tokens, hidden_dim].
        slot_mapping: Token to slot mapping [total_tokens].
        start_token: Starting token index in slot_mapping.
        end_token: Ending token index (exclusive) in slot_mapping.
        ctx: Device context for kernel launch.
    """
    var num_tokens = end_token - start_token
    if num_tokens <= 0:
        return

    var num_layers = paged_cache.dim[2]()
    var threads_per_block = min(num_kv_heads * head_dim, 256)
    var grid_x = num_tokens
    var grid_y = num_layers
    var grid_z = kv_dim

    with Trace[TraceLevel.OP, target=target](
        "mo.lmcache_onload.page_"
        + String(page_size)
        + ".nhead_"
        + String(num_kv_heads)
        + ".hdim_"
        + String(head_dim),
    ):
        comptime kernel = _lmcache_onload_kernel[
            dtype, page_size, num_kv_heads, head_dim, kv_dim
        ]
        ctx.enqueue_function[kernel, kernel](
            paged_cache,
            input,
            slot_mapping,
            start_token,
            num_tokens,
            num_layers,
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(threads_per_block, 1, 1),
        )
