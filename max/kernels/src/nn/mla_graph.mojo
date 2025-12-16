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
from linalg.bmm import batched_matmul_dynamic_scaled_fp8
from linalg.fp8_quantization import (
    matmul_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
    batched_quantize_dynamic_scaled_fp8,
)

from nn.kv_cache import KVCollectionT
from nn.kv_cache_ragged import (
    generic_flare_mla_decode_kv_cache_ragged,
    generic_flare_mla_prefill_kv_cache_ragged,
)
from nn.mla import _k_cache_to_buffer


# ===-----------------------------------------------------------------------===#
# Manually fused MLA prefill branch (FP8)
# ===-----------------------------------------------------------------------===#


fn _to_value_or_dim(value: Int) -> Dim:
    if value != UNKNOWN_VALUE:
        return Dim(value)
    else:
        return Dim()


fn _layout_shape_to_dim_list[rank: Int](layout: Layout) -> DimList:
    __comptime_assert rank == 2 or rank == 3, "rank should be 2 or 3"

    if rank == 2:
        return DimList(
            _to_value_or_dim(Int(layout.shape[0])),
            _to_value_or_dim(Int(layout.shape[1])),
        )
    elif rank == 3:
        return DimList(
            _to_value_or_dim(Int(layout.shape[0])),
            _to_value_or_dim(Int(layout.shape[1])),
            _to_value_or_dim(Int(layout.shape[2])),
        )
    else:
        return DimList.create_unknown[1]()


@always_inline
fn _layout_tensor_to_nd_buffer[
    dtype: DType,
    layout: Layout,
    origin: Origin, //,
    rank: Int = 2,
](
    tensor: LayoutTensor[
        dtype,
        layout,
        origin,
        address_space = AddressSpace.GENERIC, **_,
    ],
    out result: NDBuffer[
        dtype,
        rank,
        origin,
        _layout_shape_to_dim_list[rank](layout),
    ],
):
    """
    Helper function to convert a layout tensor to an NDBuffer. Will be removed
    once quantize_dynamic_scaled_fp8 and matmul_dynamic_scaled_fp8 support
    layout tensors.
    """
    __comptime_assert (
        rank == layout.rank()
    ), "rank should be equal to layout rank"

    return type_of(result)(
        tensor.ptr,
        rebind[IndexList[rank]](
            tensor.runtime_layout.shape.value.canonicalize()
        ),
    )


fn mla_prefill_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT, //,
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
    q_nope: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    q_rope: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
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
        q_nope: Query tensor of shape [tot_seq_len, num_heads,
            qk_nope_head_dim].
        q_rope: Query tensor of shape [tot_seq_len, num_heads,
            qk_rope_head_dim].
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

    comptime num_heads = q_nope.shape[1]()
    comptime qk_nope_head_dim = q_nope.shape[2]()
    comptime qk_rope_head_dim = q_rope.shape[2]()
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

    # concatenate the q_nope and q_rope tensors
    var seq_len = q_nope.dim(0)
    comptime q_layout = Layout.row_major(
        UNKNOWN_VALUE, num_heads, qk_nope_head_dim + qk_rope_head_dim
    )
    var q_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * (qk_nope_head_dim + qk_rope_head_dim)
    )
    var q = LayoutTensor[dtype, q_layout](
        q_buf.unsafe_ptr(),
        RuntimeLayout[q_layout].row_major(
            Index(seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim)
        ),
    )

    @always_inline
    @parameter
    @__copy_capture(q_nope, q_rope, q)
    fn concat_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        constrained[rank == 3, "rank should be equal to 3"]()
        var idx = rebind[IndexList[3]](idx_arg)
        var token_idx = idx[0]
        var head_idx = idx[1]
        var dim_idx = idx[2]

        if dim_idx < qk_nope_head_dim:
            q.store[width=width](
                idx,
                q_nope.load[width=width](Index(token_idx, head_idx, dim_idx)),
            )
        else:
            q.store[width=width](
                idx,
                q_rope.load[width=width](
                    Index(token_idx, head_idx, dim_idx - qk_nope_head_dim)
                ),
            )

    var concat_launch_shape = IndexList[3](
        seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim
    )
    comptime concat_simd_width = simd_width_of[
        dtype, target = get_gpu_target()
    ]()
    _elementwise_impl_gpu[func=concat_fn, simd_width = UInt(concat_simd_width)](
        concat_launch_shape, ctx
    )

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

    _ = q_buf^
    _ = k_buf^
    _ = v_buf^


# ===-----------------------------------------------------------------------===#
# Manually fused MLA decode branch (FP8)
# ===-----------------------------------------------------------------------===#


@always_inline
fn quantize_and_bmm_fp8_helper[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    target: StaticString = "cpu",
](
    c: LayoutTensor[mut=True, dtype, address_space = AddressSpace.GENERIC, **_],
    a: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    b: LayoutTensor[fp8_dtype, address_space = AddressSpace.GENERIC, **_],
    b_scales: LayoutTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, **_
    ],
    ctx: DeviceContext,
) raises:
    """
    Helper function to quantize and perform a batched matrix multiplication.
    """

    comptime B = a.shape[0]()
    comptime K = a.shape[2]()
    comptime N = b.shape[1]()

    var m = a.dim(1)

    # allocate buffers for quantized a and its scales
    var fp8_a_buf = ctx.enqueue_create_buffer[fp8_dtype](B * m * K)
    var fp8_a = LayoutTensor[fp8_dtype, a.layout](
        fp8_a_buf.unsafe_ptr(),
        a.runtime_layout,
    )

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    comptime fp8_a_scale_layout = Layout.row_major(
        B, K // k_scale_granularity, UNKNOWN_VALUE
    )
    var scales_padded_m = align_up(m, scales_m_padding)
    var fp8_a_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        B * (K // k_scale_granularity) * scales_padded_m
    )
    var fp8_a_scale = LayoutTensor[fp8_scale_dtype, fp8_a_scale_layout](
        fp8_a_scale_buf.unsafe_ptr(),
        RuntimeLayout[fp8_a_scale_layout].row_major(
            Index(B, K // k_scale_granularity, scales_padded_m)
        ),
    )

    batched_quantize_dynamic_scaled_fp8[k_scale_granularity](
        _layout_tensor_to_nd_buffer[3](fp8_a),
        _layout_tensor_to_nd_buffer[3](fp8_a_scale),
        _layout_tensor_to_nd_buffer[3](a),
        1200.0,
        ctx,
    )

    batched_matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        transpose_b=True,
        target=target,
    ](
        _layout_tensor_to_nd_buffer[3](c),
        _layout_tensor_to_nd_buffer[3](fp8_a),
        _layout_tensor_to_nd_buffer[3](b),
        _layout_tensor_to_nd_buffer[3](fp8_a_scale),
        _layout_tensor_to_nd_buffer[3](b_scales),
        ctx,
    )

    _ = fp8_a_buf^
    _ = fp8_a_scale_buf^


@always_inline
fn transpose_helper[
    dtype: DType
](
    output_tensor: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, **_
    ],
    input_tensor: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, **_
    ],
    ctx: DeviceContext,
) raises:
    """
    Helper function to transpose a tensor from [B, N, K] to [N, B, K] (or vice versa).
    """

    @always_inline
    @parameter
    @__copy_capture(input_tensor, output_tensor)
    fn tranpose_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        __comptime_assert rank == 3, "rank should be equal to 3"
        var idx = rebind[IndexList[3]](idx_arg)

        output_tensor.store[width=width](
            Index(idx[1], idx[0], idx[2]), input_tensor.load[width=width](idx)
        )

    var launch_shape = Index(
        input_tensor.dim(0), input_tensor.dim(1), input_tensor.dim(2)
    )
    comptime target_simd_width = simd_width_of[
        dtype, target = get_gpu_target()
    ]()
    _elementwise_impl_gpu[
        func=tranpose_fn, simd_width = UInt(target_simd_width)
    ](launch_shape, ctx)


fn mla_decode_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT, //,
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
    q_nope: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    q_rope: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, **_
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    w_uk: LayoutTensor[fp8_dtype, address_space = AddressSpace.GENERIC, **_],
    w_uk_scale: LayoutTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, **_
    ],
    w_uv: LayoutTensor[fp8_dtype, address_space = AddressSpace.GENERIC, **_],
    w_uv_scale: LayoutTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, **_
    ],
    ctx: DeviceContext,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Project q_nope to kv_latent_dim through a fp8 batched matmul:
        q_nope_proj = q_nope_t @ w_uk.
    - Concatenate q_nope_proj and q_rope:
        q_full = concat(q_nope_proj, q_rope, axis=2).
    - Perform MLA decode.
    - Project raw_output to v_head_dim through another fp8 batched matmul:
        output = raw_output_t @ w_uv.

    Parameters:
        dtype: Data type of the input and output tensors.
        fp8_dtype: Data type of the fp8 input and output tensors.
        fp8_scale_dtype: Data type of the fp8 scale input and output tensors.
        collection_t: Type of the KV collection.
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
        q_nope: Query tensor of shape [tot_seq_len, num_heads,
            qk_nope_head_dim].
        q_rope: Rope query tensor of shape [tot_seq_len, num_heads,
            qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        w_uk: Weight matrix for projecting the non-rope part of each query head to
            KV latent space. Shape: [num_heads, kv_latent_dim, qk_nope_head_dim].
        w_uk_scale: The scale for the w_uk weight matrix. Shape varies
            depending on the float8_config.
        w_uv: Weight matrix for projecting the output of the attention back to
            each head's original space. Shape: [num_heads, v_head_dim, kv_latent_dim].
        w_uv_scale: The scale for the w_uv weight matrix. Shape varies
            depending on the float8_config.
        ctx: Device context.
    """

    comptime kv_params = collection_t.kv_params
    __comptime_assert kv_params.is_mla, "kv_params.is_mla should be true"
    __comptime_assert (
        kv_params.num_heads == 1
    ), "kv_params.num_heads should be 1"

    comptime num_heads = q_nope.shape[1]()
    comptime qk_nope_head_dim = q_nope.shape[2]()
    comptime qk_rope_head_dim = q_rope.shape[2]()
    comptime v_head_dim = output.shape[2]()

    __comptime_assert (
        w_uk.layout.shape.all_known() and w_uv.layout.shape.all_known()
    ), "w_uk and w_uv's shapes should be static"
    __comptime_assert (
        w_uk.layout.shape[2].value() == qk_nope_head_dim
    ), "w_uk.layout.shape[2] should be equal to qk_nope_head_dim"
    __comptime_assert (
        w_uv.layout.shape[1].value() == v_head_dim
    ), "w_uv.layout.shape[1] should be equal to v_head_dim"
    comptime kv_latent_dim = w_uk.layout.shape[1].value()
    __comptime_assert kv_latent_dim + qk_rope_head_dim == Int(
        kv_params.head_size
    ), "kv_latent_dim + qk_rope_head_dim should be equal to kv_params.head_size"

    var seq_len = q_nope.dim(0)

    if seq_len == 0:
        return

    # First transpose the q_nope tensor from [tot_seq_len, num_heads,
    # qk_nope_head_dim] to [num_heads, tot_seq_len, qk_nope_head_dim].
    # TODO: This will be fused with the batched_quantize_dynamic_scaled_fp8 kernel in the future.
    comptime q_nope_t_layout = Layout.row_major(
        num_heads, UNKNOWN_VALUE, qk_nope_head_dim
    )
    var q_nope_t_buf = ctx.enqueue_create_buffer[dtype](
        num_heads * seq_len * qk_nope_head_dim
    )
    var q_nope_t = LayoutTensor[dtype, q_nope_t_layout](
        q_nope_t_buf.unsafe_ptr(),
        RuntimeLayout[q_nope_t_layout].row_major(
            Index(num_heads, seq_len, qk_nope_head_dim)
        ),
    )
    transpose_helper[dtype](q_nope_t, q_nope, ctx)

    # Proceed with the fp8 batched matmul
    comptime q_nope_proj_layout = Layout.row_major(
        num_heads, UNKNOWN_VALUE, kv_latent_dim
    )
    var q_nope_proj_buf = ctx.enqueue_create_buffer[dtype](
        num_heads * seq_len * kv_latent_dim
    )
    var q_nope_proj = LayoutTensor[dtype, q_nope_proj_layout](
        q_nope_proj_buf.unsafe_ptr(),
        RuntimeLayout[q_nope_proj_layout].row_major(
            Index(num_heads, seq_len, kv_latent_dim)
        ),
    )

    quantize_and_bmm_fp8_helper[
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](q_nope_proj, q_nope_t, w_uk, w_uk_scale, ctx)
    _ = q_nope_t_buf^

    # concatenate the transposed q_nope_proj and q_rope tensors
    comptime q_full_layout = Layout.row_major(
        UNKNOWN_VALUE, num_heads, kv_latent_dim + qk_rope_head_dim
    )
    var q_full_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * (kv_latent_dim + qk_rope_head_dim)
    )
    var q_full = LayoutTensor[dtype, q_full_layout](
        q_full_buf.unsafe_ptr(),
        RuntimeLayout[q_full_layout].row_major(
            Index(seq_len, num_heads, kv_latent_dim + qk_rope_head_dim)
        ),
    )

    @always_inline
    @parameter
    @__copy_capture(q_nope_proj, q_rope, q_full)
    fn concat_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        __comptime_assert rank == 3, "rank should be equal to 3"
        var idx = rebind[IndexList[3]](idx_arg)

        var token_idx = idx[0]
        var head_idx = idx[1]
        var dim_idx = idx[2]

        if dim_idx < kv_latent_dim:
            q_full.store[width=width](
                idx,
                q_nope_proj.load[width=width](
                    Index(head_idx, token_idx, dim_idx)
                ),
            )
        else:
            q_full.store[width=width](
                idx,
                q_rope.load[width=width](
                    Index(token_idx, head_idx, dim_idx - kv_latent_dim)
                ),
            )

    var concat_launch_shape = IndexList[3](
        seq_len, num_heads, kv_latent_dim + qk_rope_head_dim
    )
    comptime concat_simd_width = simd_width_of[
        dtype, target = get_gpu_target()
    ]()
    _elementwise_impl_gpu[func=concat_fn, simd_width = UInt(concat_simd_width)](
        concat_launch_shape, ctx
    )
    _ = q_nope_proj_buf^

    # Perform MLA decode
    comptime raw_output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_heads, kv_latent_dim
    )
    var raw_output_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = LayoutTensor[dtype, raw_output_layout](
        raw_output_buf.unsafe_ptr(),
        RuntimeLayout[raw_output_layout].row_major(
            Index(seq_len, num_heads, kv_latent_dim)
        ),
    )

    generic_flare_mla_decode_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
        score_mod_str=score_mod_str,
    ](
        q_full,
        input_row_offsets,
        kv_collection,
        layer_idx,
        scale,
        raw_output,
        ctx,
    )
    _ = q_full_buf^

    comptime raw_output_t_layout = Layout.row_major(
        num_heads, UNKNOWN_VALUE, kv_latent_dim
    )
    var raw_output_t_buf = ctx.enqueue_create_buffer[dtype](
        num_heads * seq_len * kv_latent_dim
    )
    var raw_output_t = LayoutTensor[dtype, raw_output_t_layout](
        raw_output_t_buf.unsafe_ptr(),
        RuntimeLayout[raw_output_t_layout].row_major(
            Index(num_heads, seq_len, kv_latent_dim)
        ),
    )
    transpose_helper[dtype](raw_output_t, raw_output, ctx)
    _ = raw_output_buf^

    # Another batched matmul to project the raw output to the original space
    comptime output_t_layout = Layout.row_major(
        num_heads, UNKNOWN_VALUE, v_head_dim
    )
    var output_t_buf = ctx.enqueue_create_buffer[dtype](
        num_heads * seq_len * v_head_dim
    )
    var output_t = LayoutTensor[dtype, output_t_layout](
        output_t_buf.unsafe_ptr(),
        RuntimeLayout[output_t_layout].row_major(
            Index(num_heads, seq_len, v_head_dim)
        ),
    )

    quantize_and_bmm_fp8_helper[
        dtype=dtype,
        fp8_dtype=fp8_dtype,
        fp8_scale_dtype=fp8_scale_dtype,
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](output_t, raw_output_t, w_uv, w_uv_scale, ctx)
    _ = raw_output_t_buf^

    # Transpose the output tensor from [num_heads, tot_seq_len, v_head_dim] to [tot_seq_len, num_heads, v_head_dim].
    transpose_helper[dtype](output, output_t, ctx)
    _ = output_t_buf^


# ===-----------------------------------------------------------------------===#
# MLA prefill-decode graph (FP8)
# ===-----------------------------------------------------------------------===#


@always_inline
fn mla_prefill_decode_graph_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT, //,
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
    q_nope: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    q_rope: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
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
    max_seq_len: Int,
    kv_b_proj: LayoutTensor[
        fp8_dtype, address_space = AddressSpace.GENERIC, **_
    ],
    kv_b_proj_scale: LayoutTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, **_
    ],
    w_uk: LayoutTensor[fp8_dtype, address_space = AddressSpace.GENERIC, **_],
    w_uk_scale: LayoutTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, **_
    ],
    w_uv: LayoutTensor[fp8_dtype, address_space = AddressSpace.GENERIC, **_],
    w_uv_scale: LayoutTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, **_
    ],
    ctx: DeviceContext,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Perform MLA prefill or decode based on the maximum sequence length.
    """

    var seq_len = q_nope.dim(0)

    if seq_len == 0:
        return

    if max_seq_len == 1:
        mla_decode_branch_fp8[
            m_scale_granularity=m_scale_granularity,
            n_scale_granularity=n_scale_granularity,
            k_scale_granularity=k_scale_granularity,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            target=target,
        ](
            output,
            q_nope,
            q_rope,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            w_uk,
            w_uk_scale,
            w_uv,
            w_uv_scale,
            ctx,
        )

    else:
        mla_prefill_branch_fp8[
            m_scale_granularity=m_scale_granularity,
            n_scale_granularity=n_scale_granularity,
            k_scale_granularity=k_scale_granularity,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            target=target,
        ](
            output,
            q_nope,
            q_rope,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            buffer_row_offsets,
            cache_offsets,
            buffer_length,
            kv_b_proj,
            kv_b_proj_scale,
            ctx,
        )
