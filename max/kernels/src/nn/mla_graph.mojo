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


from buffer import Dim, DimList, NDBuffer
from math import align_up
from memory import AddressSpace, LegacyUnsafePointer as UnsafePointer
from sys import simd_width_of, size_of
from utils.index import Index, IndexList

from algorithm.functional import _elementwise_impl_gpu
from gpu.host import DeviceContext, get_gpu_target
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from linalg.fp8_quantization import (
    matmul_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
)

from nn.kv_cache import KVCollectionT
from nn.kv_cache_ragged import generic_flare_mla_prefill_kv_cache_ragged
from nn.mla import _k_cache_to_buffer


# ===-----------------------------------------------------------------------===#
# Manually fused MLA prefill branch (FP8)
# ===-----------------------------------------------------------------------===#


fn _to_value_or_dim(value: Int) -> Dim:
    if value != UNKNOWN_VALUE:
        return Dim(value)
    else:
        return Dim()


@always_inline
fn _layout_tensor_to_nd_buffer[
    dtype: DType,
    layout: Layout,
    origin: Origin,
](
    tensor: LayoutTensor[
        dtype,
        layout,
        origin,
        address_space = AddressSpace.GENERIC, **_,
    ],
    out result: NDBuffer[
        dtype,
        2,
        origin,
        DimList(
            _to_value_or_dim(Int(layout.shape[0])),
            _to_value_or_dim(Int(layout.shape[1])),
        ),
    ],
):
    """
    Helper function to convert a layout tensor to an NDBuffer. Will be removed
    once quantize_dynamic_scaled_fp8 and matmul_dynamic_scaled_fp8 support
    layout tensors.
    """
    constrained[layout.rank() == 2, "layout should be 2D"]()
    return type_of(result)(
        tensor.ptr,
        rebind[IndexList[2]](tensor.runtime_layout.shape.value.canonicalize()),
    )


fn mla_prefill_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT, //,
    qk_nope_head_dim: Int,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, **_
    ],
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, **_
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    buffer_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, **_
    ],
    cache_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, **_
    ],
    buffer_length: Int,
    kv_b_proj: LayoutTensor[
        fp8_dtype, address_space = AddressSpace.GENERIC, **_
    ],
    kv_b_proj_scale: LayoutTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, **_
    ],
    ctx: DeviceContext,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Copy the KV latent values from PagedKVCache to a contiguous buffer.
    - Quantize the KV latent values to fp8.
    - Up-project the latent KV values to full K and V through a matmul.
    - Split the concatenated KV into K and V.
    - Perform MLA prefill.

    Parameters:
        dtype: Data type of the input and output tensors.
        fp8_dtype: Data type of the fp8 input and output tensors.
        fp8_scale_dtype: Data type of the fp8 scale input and output tensors.
        collection_t: Type of the KV collection.
        qk_nope_head_dim: Dimension of non-rope parts of the Q/K heads.
        m_scale_granularity: Granularity of the scale for M dimension of the
            matrix multiplication.
        n_scale_granularity: Granularity of the scale for N dimension of the
            matrix multiplication.
        k_scale_granularity: Granularity of the scale for K dimension of the
            matrix multiplication.
        mask_str: Mask variant.
        score_mod_str: Positional encoding variant.
        target: Target device.

    Args:
        output: Output tensor of shape [tot_seq_len, num_heads, v_head_dim].
        q: Query tensor of shape [tot_seq_len, num_heads,
            qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        buffer_row_offsets: Indicates where each request's KV latent values
            should be stored in the contiguous K buffer. This is a 1D tensor
            of shape [num_batches + 1].
        cache_offsets: Indicates the starting token position in the KV cache
            from which to copy KV latent values for each request. This is a 1D
            tensor of shape [num_batches + 1].
        buffer_length: The total number of tokens in the KV cache. Scalar.
        kv_b_proj: Weight matrix for up-projecting the KV latent values to full
            K and V. Shape: [num_heads * (qk_nope_head_dim + v_head_dim),
            kv_latent_dim].
        kv_b_proj_scale: The scale for the weight matrix. Shape varies
            depending on the float8_config.
        ctx: Device context.
    """
    comptime kv_params = collection_t.kv_params
    constrained[kv_params.is_mla, "kv_params.is_mla should be true"]()
    constrained[kv_params.num_heads == 1, "kv_params.num_heads should be 1"]()

    comptime num_heads = q.shape[1]()
    comptime v_head_dim = output.shape[2]()

    constrained[
        kv_b_proj.layout.shape.all_known(), "kv_b_proj's shape should be static"
    ]()
    constrained[
        kv_b_proj.layout.shape[0].value()
        == num_heads * (qk_nope_head_dim + v_head_dim),
        (
            "kv_b_proj.layout.shape[0] should be equal to num_heads *"
            " (qk_nope_head_dim + v_head_dim)"
        ),
    ]()
    comptime kv_latent_dim = kv_b_proj.layout.shape[1].value()

    constrained[m_scale_granularity == 1, "m_scale_granularity should be 1"]()
    constrained[
        n_scale_granularity == k_scale_granularity == 128,
        "n, k scale_granularity should be 128",
    ]()

    # Return early if we have no tokens to process.
    if buffer_length == 0:
        return

    # First, dump the k cache to a contiguous buffer
    # allocate a buffer for raw latent KV values
    comptime k_latent_layout = Layout.row_major(UNKNOWN_VALUE, kv_latent_dim)
    var k_latent_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * kv_latent_dim
    )
    var k_latent = LayoutTensor[dtype, k_latent_layout](
        k_latent_buf.unsafe_ptr(),
        RuntimeLayout[k_latent_layout].row_major(
            Index(buffer_length, kv_latent_dim)
        ),
    )

    # copy the k cache to the latent buffer
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    _k_cache_to_buffer(
        buffer_row_offsets,
        cache_offsets,
        k_cache,
        buffer_length,
        k_latent,
        ctx,
    )

    # quantize the latent KV values to fp8
    # allocate buffers for fp8 latent KV values and scales
    # TODO: Fused the _k_cache_to_buffer with the quantize_dynamic_scaled_fp8
    var fp8_k_latent_buf = ctx.enqueue_create_buffer[fp8_dtype](
        buffer_length * kv_latent_dim
    )
    var fp8_k_latent = LayoutTensor[fp8_dtype, k_latent_layout](
        fp8_k_latent_buf.unsafe_ptr(),
        RuntimeLayout[k_latent_layout].row_major(
            Index(buffer_length, kv_latent_dim)
        ),
    )

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    comptime fp8_k_latent_scale_layout = Layout.row_major(
        kv_latent_dim // k_scale_granularity, UNKNOWN_VALUE
    )
    var scales_padded_m = align_up(buffer_length, scales_m_padding)
    var fp8_k_latent_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        scales_padded_m * kv_latent_dim // k_scale_granularity
    )
    var fp8_k_latent_scale = LayoutTensor[
        fp8_scale_dtype, fp8_k_latent_scale_layout
    ](
        fp8_k_latent_scale_buf.unsafe_ptr(),
        RuntimeLayout[fp8_k_latent_scale_layout].row_major(
            Index(kv_latent_dim // k_scale_granularity, scales_padded_m)
        ),
    )

    quantize_dynamic_scaled_fp8[k_scale_granularity](
        _layout_tensor_to_nd_buffer(fp8_k_latent),
        _layout_tensor_to_nd_buffer(fp8_k_latent_scale),
        _layout_tensor_to_nd_buffer(k_latent),
        1200.0,
        ctx,
    )
    _ = k_latent_buf^

    # allocate buffers for concatenated KV
    comptime kv_layout = Layout.row_major(
        UNKNOWN_VALUE, num_heads * (qk_nope_head_dim + v_head_dim)
    )
    var kv_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * (qk_nope_head_dim + v_head_dim)
    )
    var kv = LayoutTensor[dtype, kv_layout](
        kv_buf.unsafe_ptr(),
        RuntimeLayout[kv_layout].row_major(
            Index(
                buffer_length,
                num_heads * (qk_nope_head_dim + v_head_dim),
            )
        ),
    )

    # up-project the latent KV values to full K and V
    matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        transpose_b=True,
        target=target,
    ](
        _layout_tensor_to_nd_buffer(kv),
        _layout_tensor_to_nd_buffer(fp8_k_latent),
        _layout_tensor_to_nd_buffer(kv_b_proj),
        _layout_tensor_to_nd_buffer(fp8_k_latent_scale),
        _layout_tensor_to_nd_buffer(kv_b_proj_scale),
        ctx,
    )
    _ = fp8_k_latent_buf^
    _ = fp8_k_latent_scale_buf^

    # allocate buffers for full K and V
    comptime k_layout = Layout.row_major(
        UNKNOWN_VALUE, num_heads, qk_nope_head_dim
    )
    comptime v_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, v_head_dim)
    var k_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * qk_nope_head_dim
    )
    var v_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * v_head_dim
    )
    var k = LayoutTensor[dtype, k_layout](
        k_buf.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(buffer_length, num_heads, qk_nope_head_dim)
        ),
    )
    var v = LayoutTensor[dtype, v_layout](
        v_buf.unsafe_ptr(),
        RuntimeLayout[v_layout].row_major(
            Index(buffer_length, num_heads, v_head_dim)
        ),
    )

    # split the concatenated KV into K and V
    # TODO: Remove this once matmul_dynamic_scaled_fp8 supports epilogue
    @always_inline
    @parameter
    @__copy_capture(kv, k, v)
    fn split_kv_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        constrained[rank == 2, "rank should be equal to 2"]()

        constrained[
            qk_nope_head_dim % width == 0,
            "qk_nope_head_dim should be divisible by simd width",
        ]()
        constrained[
            v_head_dim % width == 0,
            "v_head_dim should be divisible by simd width",
        ]()
        var idx = rebind[IndexList[2]](idx_arg)
        var token_idx = idx[0]
        var hid_idx = idx[1]

        var val = kv.aligned_load[width=width](token_idx, hid_idx)

        var head_idx, head_dim_idx = divmod(
            hid_idx, qk_nope_head_dim + v_head_dim
        )

        if head_dim_idx < qk_nope_head_dim:
            k.store[width=width](
                IndexList[3](token_idx, head_idx, head_dim_idx), val
            )
        else:
            head_dim_idx -= qk_nope_head_dim
            v.store[width=width](
                IndexList[3](token_idx, head_idx, head_dim_idx), val
            )

    var launch_shape = IndexList[2](buffer_length, kv.dim(1))
    comptime target_simd_width = simd_width_of[
        dtype, target = get_gpu_target()
    ]()
    _elementwise_impl_gpu[
        func=split_kv_fn, simd_width = UInt(target_simd_width)
    ](launch_shape, ctx)
    _ = kv_buf^

    generic_flare_mla_prefill_kv_cache_ragged[
        write_softmax_info=False,
        use_cascade_attention=False,
        target=target,
        mask_str=mask_str,
        score_mod_str=score_mod_str,
    ](
        q,
        k,
        v,
        buffer_row_offsets,
        cache_offsets,
        input_row_offsets,
        kv_collection,
        layer_idx,
        scale,
        output,
        LayoutTensor[DType.float32, Layout.row_major(1, 1, 1)](
            UnsafePointer[Float32]()
        ),
        ctx,
    )
    _ = k_buf^
    _ = v_buf^
