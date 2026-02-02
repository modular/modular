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


from math import align_up

from sys import simd_width_of, size_of
from utils.index import Index, IndexList

from algorithm.functional import _elementwise_impl_gpu
from gpu.host import DeviceContext, get_gpu_target
from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import Layout, row_major
from layout._tile_tensor import TileTensor
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


fn mla_prefill_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q_nope: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    q_rope: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    buffer_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_length: Int,
    kv_b_proj: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    kv_b_proj_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
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
    __comptime_assert kv_params.is_mla, "kv_params.is_mla should be true"
    __comptime_assert (
        kv_params.num_heads == 1
    ), "kv_params.num_heads should be 1"

    comptime num_heads = q_nope.static_shape[1]
    comptime qk_nope_head_dim = q_nope.static_shape[2]
    comptime qk_rope_head_dim = q_rope.static_shape[2]
    comptime v_head_dim = output.static_shape[2]

    __comptime_assert (
        kv_b_proj.shape_known
    ), "kv_b_proj's shape should be static"
    __comptime_assert kv_b_proj.static_shape[0] == num_heads * (
        qk_nope_head_dim + v_head_dim
    ), (
        "kv_b_proj.layout.shape[0] should be equal to num_heads *"
        " (qk_nope_head_dim + v_head_dim)"
    )
    comptime kv_latent_dim = kv_b_proj.static_shape[1]

    __comptime_assert (
        m_scale_granularity == 1
    ), "m_scale_granularity should be 1"
    __comptime_assert (
        n_scale_granularity == k_scale_granularity == 128
    ), "n, k scale_granularity should be 128"

    # Return early if we have no tokens to process.
    if buffer_length == 0:
        return

    # concatenate the q_nope and q_rope tensors
    var seq_len = Int(q_nope.dim(0))
    var q_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * (qk_nope_head_dim + qk_rope_head_dim)
    )
    var q = TileTensor(
        q_buf,
        row_major(
            (
                Idx(seq_len),
                Idx[num_heads](),
                Idx[qk_nope_head_dim + qk_rope_head_dim](),
            )
        ),
    )

    @always_inline
    @parameter
    @__copy_capture(q_nope, q_rope, q)
    fn concat_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        __comptime_assert q_rope.rank == 3, "rank should be equal to 3"

        var token_idx = idx[0]
        var head_idx = idx[1]
        var dim_idx = idx[2]

        var store_coord = Coord(Idx(token_idx), Idx(head_idx), Idx(dim_idx))
        __comptime_assert store_coord.rank == q.rank

        if dim_idx < qk_nope_head_dim:
            __comptime_assert store_coord.rank == q_nope.rank
            q.store[width=width](
                store_coord, q_nope.load[width=width](store_coord)
            )
        else:
            var load_coord = Coord(
                Idx(token_idx), Idx(head_idx), Idx(dim_idx - qk_nope_head_dim)
            )
            __comptime_assert load_coord.rank == q_rope.rank
            q.store[width=width](
                store_coord, q_rope.load[width=width](load_coord)
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
    var k_latent_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * kv_latent_dim
    )
    var k_latent = TileTensor(
        k_latent_buf,
        row_major((Idx(buffer_length), Idx[kv_latent_dim]())),
    )

    # copy the k cache to the latent buffer
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    _k_cache_to_buffer(
        buffer_row_offsets.to_layout_tensor(),
        cache_offsets.to_layout_tensor(),
        k_cache,
        Int32(buffer_length),
        k_latent.to_layout_tensor(),
        ctx,
    )

    # quantize the latent KV values to fp8
    # allocate buffers for fp8 latent KV values and scales
    # TODO: Fused the _k_cache_to_buffer with the quantize_dynamic_scaled_fp8
    var fp8_k_latent_buf = ctx.enqueue_create_buffer[fp8_dtype](
        buffer_length * kv_latent_dim
    )
    var fp8_k_latent = TileTensor(
        fp8_k_latent_buf,
        row_major((Idx(buffer_length), Idx[kv_latent_dim]())),
    )

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    var scales_padded_m = align_up(buffer_length, scales_m_padding)
    var fp8_k_latent_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        scales_padded_m * kv_latent_dim // k_scale_granularity
    )
    var fp8_k_latent_scale = TileTensor(
        fp8_k_latent_scale_buf,
        row_major(
            (Idx[kv_latent_dim // k_scale_granularity](), Idx(scales_padded_m))
        ),
    )

    @__copy_capture(k_latent)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[k_latent.dtype, width]:
        return k_latent.load[width=width]((Idx(row), Idx(col)))

    quantize_dynamic_scaled_fp8[
        input_fn, k_scale_granularity, k_latent.static_shape[1]
    ](
        fp8_k_latent._to_ndbuffer(),
        fp8_k_latent_scale._to_ndbuffer(),
        1200.0,
        ctx,
        Int(k_latent.dim[0]()),
    )

    # allocate buffers for concatenated KV
    var kv_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * (qk_nope_head_dim + v_head_dim)
    )
    var kv = TileTensor(
        kv_buf,
        row_major(
            (
                Idx(buffer_length),
                Idx[num_heads * (qk_nope_head_dim + v_head_dim)](),
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
        kv,
        fp8_k_latent,
        kv_b_proj,
        fp8_k_latent_scale,
        kv_b_proj_scale,
        ctx,
    )

    # allocate buffers for full K and V
    var k_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * qk_nope_head_dim
    )
    var v_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * v_head_dim
    )
    var k = TileTensor(
        k_buf,
        row_major(
            (Idx(buffer_length), Idx[num_heads](), Idx[qk_nope_head_dim]())
        ),
    )
    var v = TileTensor(
        v_buf,
        row_major((Idx(buffer_length), Idx[num_heads](), Idx[v_head_dim]())),
    )

    # split the concatenated KV into K and V
    # TODO: Remove this once matmul_dynamic_scaled_fp8 supports epilogue
    @always_inline
    @parameter
    @__copy_capture(kv, k, v)
    fn split_kv_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        __comptime_assert rank == 2, "rank should be equal to 2"

        __comptime_assert (
            qk_nope_head_dim % width == 0
        ), "qk_nope_head_dim should be divisible by simd width"
        __comptime_assert (
            v_head_dim % width == 0
        ), "v_head_dim should be divisible by simd width"
        var idx = rebind[IndexList[2]](idx_arg)
        var token_idx = idx[0]
        var hid_idx = idx[1]

        var val = kv.load[width=width]((Idx(token_idx), Idx(hid_idx)))

        var head_idx, head_dim_idx = divmod(
            hid_idx, qk_nope_head_dim + v_head_dim
        )

        if head_dim_idx < qk_nope_head_dim:
            k.store[width=width](
                (Idx(token_idx), Idx(head_idx), Idx(head_dim_idx)), val
            )
        else:
            head_dim_idx -= qk_nope_head_dim
            v.store[width=width](
                (Idx(token_idx), Idx(head_idx), Idx(head_dim_idx)), val
            )

    var launch_shape = IndexList[2](buffer_length, Int(kv.dim(1)))
    comptime target_simd_width = simd_width_of[
        dtype, target = get_gpu_target()
    ]()
    _elementwise_impl_gpu[
        func=split_kv_fn, simd_width = UInt(target_simd_width)
    ](launch_shape, ctx)

    generic_flare_mla_prefill_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
        score_mod_str=score_mod_str,
    ](
        q.to_layout_tensor(),
        k.to_layout_tensor(),
        v.to_layout_tensor(),
        buffer_row_offsets.to_layout_tensor(),
        cache_offsets.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        kv_collection,
        layer_idx,
        scale,
        output.to_layout_tensor(),
        ctx,
    )


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
    c: TileTensor[mut=True, dtype, address_space = AddressSpace.GENERIC, ...],
    a: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    b: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    b_scales: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """
    Helper function to quantize and perform a batched matrix multiplication.
    This function uses the transposed view of the input tensor `a`.
    """

    comptime B = a.static_shape[1]
    comptime K = a.static_shape[2]
    comptime N = b.static_shape[1]

    var m = Int(a.dim(0))

    # allocate buffers for quantized a and its scales
    var fp8_a_buf = ctx.enqueue_create_buffer[fp8_dtype](B * m * K)
    var fp8_a = TileTensor(fp8_a_buf, row_major((Idx[B](), Idx(m), Idx[K]())))

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    var scales_padded_m = align_up(m, scales_m_padding)
    var fp8_a_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        B * (K // k_scale_granularity) * scales_padded_m
    )
    var fp8_a_scale = TileTensor(
        fp8_a_scale_buf,
        row_major(
            (Idx[B](), Idx[K // k_scale_granularity](), Idx(scales_padded_m))
        ),
    )

    var a_ndbuffer = a._to_ndbuffer()

    @parameter
    @__copy_capture(a)
    @always_inline
    fn input_fn[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) capturing -> SIMD[dtype, width]:
        # First transpose the q_nope tensor from [row, batch, col] to [batch, row, col].
        __comptime_assert a.rank == 3
        return a.load[width=width]((Idx(row), Idx(batch), Idx(col)))

    batched_quantize_dynamic_scaled_fp8[
        input_fn=input_fn,
        group_size_or_per_token=k_scale_granularity,
        num_cols=K,
    ](
        fp8_a._to_ndbuffer(),
        fp8_a_scale._to_ndbuffer(),
        1200.0,
        ctx,
        num_rows=m,
        batch_size=B,
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
        c.to_layout_tensor(),
        fp8_a.to_layout_tensor(),
        b.to_layout_tensor(),
        fp8_a_scale.to_layout_tensor(),
        b_scales.to_layout_tensor(),
        ctx,
    )


@always_inline
fn transpose_helper[
    dtype: DType
](
    output_tensor: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_tensor: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
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
    ](idx: IndexList[rank]):
        __comptime_assert rank == 3, "rank should be equal to 3"
        # Transpose by swapping first two dimensions: [B, N, K] -> [N, B, K]
        var input_coord = Coord(idx)
        var output_coord = Coord(Idx(idx[1]), Idx(idx[0]), Idx(idx[2]))
        __comptime_assert input_coord.rank == input_tensor.rank
        __comptime_assert output_coord.rank == output_tensor.rank

        output_tensor.store[width=width](
            output_coord, input_tensor.load[width=width](input_coord)
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
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q_nope: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    q_rope: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    w_uk: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_uk_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    w_uv: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_uv_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
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

    comptime num_heads = q_nope.static_shape[1]
    comptime qk_nope_head_dim = q_nope.static_shape[2]
    comptime qk_rope_head_dim = q_rope.static_shape[2]
    comptime v_head_dim = output.static_shape[2]

    __comptime_assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    __comptime_assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    __comptime_assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"
    comptime kv_latent_dim = w_uk.static_shape[1]
    __comptime_assert kv_latent_dim + qk_rope_head_dim == Int(
        kv_params.head_size
    ), "kv_latent_dim + qk_rope_head_dim should be equal to kv_params.head_size"

    var seq_len = Int(q_nope.dim(0))

    if seq_len == 0:
        return

    # Proceed with the fp8 batched matmul
    var q_nope_proj_buf = ctx.enqueue_create_buffer[dtype](
        num_heads * seq_len * kv_latent_dim
    )
    var q_nope_proj = TileTensor(
        q_nope_proj_buf,
        row_major((Idx[num_heads](), Idx(seq_len), Idx[kv_latent_dim]())),
    )

    # This helper function uses the transposed view of the input tensor `q_nope`.
    quantize_and_bmm_fp8_helper[
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](q_nope_proj, q_nope, w_uk, w_uk_scale, ctx)

    # concatenate the transposed q_nope_proj and q_rope tensors
    var q_full_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * (kv_latent_dim + qk_rope_head_dim)
    )
    var q_full = TileTensor(
        q_full_buf,
        row_major(
            (
                Idx(seq_len),
                Idx[num_heads](),
                Idx[kv_latent_dim + qk_rope_head_dim](),
            ),
        ),
    )

    @always_inline
    @parameter
    @__copy_capture(q_nope_proj, q_rope, q_full)
    fn concat_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        __comptime_assert rank == 3, "rank should be equal to 3"
        var coord = Coord(idx)
        __comptime_assert coord.rank == q_full.rank
        __comptime_assert q_rope.rank == 3

        var token_idx = idx[0]
        var head_idx = idx[1]
        var dim_idx = idx[2]

        if dim_idx < kv_latent_dim:
            q_full.store[width=width](
                coord,
                q_nope_proj.load[width=width](
                    (Idx(head_idx), Idx(token_idx), Idx(dim_idx))
                ),
            )
        else:
            q_full.store[width=width](
                coord,
                q_rope.load[width=width](
                    (
                        Idx(token_idx),
                        Idx(head_idx),
                        Idx(dim_idx - kv_latent_dim),
                    )
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

    # Perform MLA decode
    var raw_output_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major((Idx(seq_len), Idx[num_heads](), Idx[kv_latent_dim]())),
    )

    generic_flare_mla_decode_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
        score_mod_str=score_mod_str,
    ](
        q_full.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        kv_collection,
        layer_idx,
        scale,
        raw_output.to_layout_tensor(),
        ctx,
    )

    # Create a view of the output tensor with logical shape
    # [num_heads, seq_len, v_head_dim], and map directly to
    # [seq_len, num_heads, v_head_dim] physical memory.
    var output_t = TileTensor(
        output.ptr,
        Layout(
            (Idx[num_heads](), Idx(seq_len), Idx[v_head_dim]()),
            (Idx[v_head_dim](), Idx[num_heads * v_head_dim](), Idx(1)),
        ),
    )

    # Another batched matmul to project the raw output to the original space
    # This helper function uses the transposed view of the input tensor `raw_output`.
    quantize_and_bmm_fp8_helper[
        dtype=dtype,
        fp8_dtype=fp8_dtype,
        fp8_scale_dtype=fp8_scale_dtype,
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](output_t, raw_output, w_uv, w_uv_scale, ctx)


# ===-----------------------------------------------------------------------===#
# MLA prefill-decode graph (FP8)
# ===-----------------------------------------------------------------------===#


@always_inline
fn mla_prefill_decode_graph_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString = "cpu",
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q_nope: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    q_rope: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    buffer_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_length: Int,
    max_seq_len: Int,
    kv_b_proj: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    kv_b_proj_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    w_uk: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_uk_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    w_uv: TileTensor[fp8_dtype, address_space = AddressSpace.GENERIC, ...],
    w_uv_scale: TileTensor[
        fp8_scale_dtype, address_space = AddressSpace.GENERIC, ...
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

    # TODO: Remove this once prefill and decode branches support FP8 KV cache KERN-2394.
    __comptime_assert (
        collection_t.dtype == dtype
    ), "This KVCache DType is not supported."

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
