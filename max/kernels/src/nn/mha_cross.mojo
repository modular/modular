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

from math import ceildiv
from sys import align_of, simd_width_of

from algorithm.functional import vectorize
from gpu import block_idx, global_idx
from gpu.host import DeviceContext, DeviceBuffer
from kv_cache.types import KVCacheT
from layout._coord import Coord, Idx
from layout._layout import Layout, TensorLayout, row_major
from layout._tile_tensor import TileTensor
from nn.mha import MHAConfig, _kernel_mask
from nn.mha_mask import MHAMask
from nn.softmax import _softmax_gpu

from utils.index import Index, IndexList
from utils.numerics import get_accum_type


@always_inline
fn _bmm0_bs[
    QLayoutType: TensorLayout,
    KVLayoutType: TensorLayout,
    //,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    q_type: DType,
    p_type: DType,
](
    p_ptr: UnsafePointer[Scalar[p_type], MutAnyOrigin],
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k_cache: cache_t,
    q_input_row_offsets: TileTensor[DType.uint32, QLayoutType, MutAnyOrigin],
    kv_input_row_offsets: TileTensor[DType.uint32, KVLayoutType, MutAnyOrigin],
    scale: Float32,
    batch_size: Int,
    q_max_seq_len: Int,
    # The maximum current sequence length in the KV cache.
    kv_max_seq_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    mask_functor: mask_t,
):
    comptime assert q_input_row_offsets.flat_rank == 1
    comptime assert kv_input_row_offsets.flat_rank == 1

    # total_context_length
    var x = global_idx.x
    # prompt_length
    var y = global_idx.y

    comptime k_type = cache_t.dtype
    comptime kv_num_heads = cache_t.kv_params.num_heads

    var batch_head = block_idx.z
    var batch, head = divmod(batch_head, UInt(num_heads))

    var cur_query_len: Int
    var cur_kv_len: Int
    var q_offset: Int
    var num_keys: Int
    var padded_num_keys = kv_max_seq_len + max_cache_size
    var p_offset = batch_head * UInt(q_max_seq_len) * UInt(padded_num_keys)

    q_seq_start = Int(q_input_row_offsets[batch])
    q_seq_end = Int(q_input_row_offsets[batch + 1])
    cur_query_len = q_seq_end - q_seq_start
    q_offset = (q_seq_start * num_heads + Int(head)) * depth

    kv_seq_start = Int(kv_input_row_offsets[batch])
    kv_seq_end = Int(kv_input_row_offsets[batch + 1])
    cur_kv_len = kv_seq_end - kv_seq_start
    # num_heads * kv_max_seq_len * batch * depth + depth * head
    num_keys = cur_kv_len + k_cache.cache_length(Int(batch))

    debug_assert(cur_kv_len <= kv_max_seq_len, "Invalid cur_kv_len")
    debug_assert(num_keys <= padded_num_keys, "Invalid max_cache_size")

    if x >= UInt(kv_max_seq_len + max_cache_size) or y >= UInt(q_max_seq_len):
        return

    var q = q_ptr + q_offset

    var kv_head = Int(head // UInt(group))

    var p = p_ptr + Int(p_offset)

    var accum = Scalar[p_type](0.0)

    # Set total KV length: KV written previous to and during this forward.
    if x < UInt(num_keys) and y < UInt(cur_query_len):
        var accum_vec = SIMD[p_type, simd_width_of[p_type]()](0)
        var k_ptr = k_cache.block_paged_ptr[tile_size=1](
            Int(batch), Int(x), kv_head, 0
        )

        fn accum_fn[width: Int](offset: Int) unified {mut}:
            comptime alignment = align_of[SIMD[p_type, width]]()
            var q_val = q.load[width=width, alignment=alignment](
                y * UInt(num_heads) * UInt(depth) + UInt(offset)
            ).cast[k_type]()
            var k_val = k_ptr.load[width=width, alignment=alignment](offset)
            var qk_val = (q_val * k_val).cast[p_type]()

            @parameter
            if width == 1:
                accum += rebind[type_of(accum)](qk_val)
            else:
                accum_vec += rebind[type_of(accum_vec)](qk_val)

        vectorize[simd_width_of[p_type]()](depth, accum_fn)
        accum += accum_vec.reduce_add()

    var score_row = y
    var score_col = x
    p[y * UInt(padded_num_keys) + x] = mask_functor.mask(
        Index(Int(batch), Int(head), Int(score_row), Int(score_col)),
        accum * scale.cast[p_type](),
    )
    p[y * UInt(padded_num_keys) + x] = _kernel_mask(
        Index(score_row, score_col),
        Index(cur_query_len, num_keys),
        p[y * UInt(padded_num_keys) + x],
    )


@always_inline
fn _bmm1_bs[
    QLayoutType: TensorLayout,
    KVLayoutType: TensorLayout,
    //,
    cache_t: KVCacheT,
    p_type: DType,
    output_type: DType,
](
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    p_ptr: UnsafePointer[Scalar[p_type], ImmutAnyOrigin],
    v_cache: cache_t,
    q_input_row_offsets: TileTensor[DType.uint32, QLayoutType, MutAnyOrigin],
    kv_input_row_offsets: TileTensor[DType.uint32, KVLayoutType, MutAnyOrigin],
    q_max_seq_len: Int,
    kv_max_seq_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
):
    comptime assert q_input_row_offsets.flat_rank == 1
    comptime assert kv_input_row_offsets.flat_rank == 1

    comptime v_type = cache_t.dtype
    comptime kv_num_heads = cache_t.kv_params.num_heads

    # head_size
    var x = global_idx.x
    # query seq_len
    var y = global_idx.y

    var batch_head = block_idx.z
    var batch, head = divmod(batch_head, UInt(num_heads))

    var cur_query_len: Int
    var cur_kv_len: Int
    var output_offset: Int
    var padded_num_keys = kv_max_seq_len + max_cache_size
    var p_offset = batch_head * UInt(q_max_seq_len) * UInt(padded_num_keys)

    q_seq_start = Int(q_input_row_offsets[batch])
    q_seq_end = Int(q_input_row_offsets[batch + 1])
    cur_query_len = q_seq_end - q_seq_start

    output_offset = (q_seq_start * num_heads + Int(head)) * depth

    kv_seq_start = Int(kv_input_row_offsets[batch])
    kv_seq_end = Int(kv_input_row_offsets[batch + 1])
    cur_kv_len = kv_seq_end - kv_seq_start

    debug_assert(cur_query_len <= q_max_seq_len, "Invalid cur_query_len")
    debug_assert(cur_kv_len <= kv_max_seq_len, "Invalid cur_kv_len")

    if x >= UInt(depth) or y >= UInt(cur_query_len):
        return

    var p = p_ptr + p_offset

    var kv_head = Int(head // UInt(group))
    var output = output_ptr + output_offset

    var accum = Float32(0.0)

    for i in range(cur_kv_len + v_cache.cache_length(Int(batch))):
        var v_ptr = v_cache.block_paged_ptr[tile_size=1](
            Int(batch), i, kv_head, Int(x)
        )
        accum += (
            p[y * UInt(padded_num_keys) + UInt(i)].cast[v_type]() * v_ptr[0]
        ).cast[DType.float32]()

    output[y * UInt(num_heads) * UInt(depth) + x] = accum.cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Naive GPU multihead cross attention supporting flexible dimensions and
# batch_size > 1.
# ===-----------------------------------------------------------------------===#


fn mha_cross_gpu_naive[
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    //,
    rank: Int,
](
    output: TileTensor[address_space = AddressSpace.GENERIC, ...],
    q: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    q_input_row_offsets: TileTensor[DType.uint32, ...],
    q_max_seq_len: Int,
    k: cache_t,
    v: cache_t,
    kv_input_row_offsets: TileTensor[DType.uint32, ...],
    mask_functor: mask_t,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    """Naive cross attention on GPU.

    Note that this assumes ragged tensor inputs and uses a mask functor.

    Computes:
        (1) Transpose (Q) BSHD -> BHSD;
        (2) Transpose (K) BSHD -> BHSD;
        (3) Transpose (V) BSHD -> BHSD;
        (4) P = Bmm(Q, K), P is also called "score";
        (5) P = P * scale + mask;
        (6) P = softmax(P);
        (7) O = Bmm(P, V)
        (8) Output = Transpose(O).

    B, S, H, D denote batch size, sequence length, head count and depth, respectively.
    (1), (2), (3) happens while loading the data into shared memory.
    (8) happens when writing output to global memory.

    All inputs (query, key, and value) must have BSHD layout. The mask can be
    BSS or BHSS.

    This kernel also handles grouped attention optimization. In this case the shape of
    K and V are BShD where h = H / num_groups.
    """
    comptime assert rank == 3, "only support rank 3 inputs for ragged inputs."
    comptime assert (
        q.dtype == cache_t.dtype == cache_t.dtype == output.dtype
    ), "Q, K, V, output should have same type."
    comptime assert (
        q.dtype == DType.float32 or q.dtype.is_half_float()
    ), "Only support single and half precision."

    comptime config = MHAConfig[dtype](
        UInt(Int(q.static_shape[rank - 2])),
        UInt(Int(q.static_shape[rank - 1])),
    )

    comptime num_heads = Int(config.num_heads)
    comptime depth = Int(config.depth)
    comptime kv_num_heads = cache_t.kv_params.num_heads
    comptime group = config.num_heads // kv_num_heads
    var kv_max_seq_len = Int(k.max_prompt_length())
    var batch_size = Int(q_input_row_offsets.dim[0]()) - 1
    var max_cache_size = Int(k.max_context_length())

    comptime q_type = q.dtype
    comptime k_type = cache_t.dtype
    comptime v_type = cache_t.dtype

    # Assume self attention if the query sequence length isn't passed.
    var num_keys = kv_max_seq_len + max_cache_size
    comptime p_type = get_accum_type[q_type]()
    var p_device = ctx.enqueue_create_buffer[p_type](
        batch_size * num_heads * q_max_seq_len * num_keys
    )

    # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
    var p_buffer = TileTensor(
        p_device.unsafe_ptr(),
        row_major(
            (Idx(batch_size * num_heads), Idx(q_max_seq_len), Idx(num_keys))
        ),
    )
    var q_device = DeviceBuffer[q_type](ctx, q.ptr, q.numel(), owning=False)

    comptime kernel_0 = _bmm0_bs[
        QLayoutType = q.LayoutType,
        KVLayoutType = kv_input_row_offsets.LayoutType,
        type_of(k),
        mask_t,
        q_type,
        p_type,
    ]
    ctx.enqueue_function_experimental[kernel_0](
        p_device,
        q_device,
        k,
        q_input_row_offsets,
        kv_input_row_offsets,
        scale,
        batch_size,
        q_max_seq_len,
        kv_max_seq_len,
        max_cache_size,
        num_heads,
        depth,
        Int(group),
        mask_functor,
        grid_dim=(
            ceildiv(num_keys, 32),
            ceildiv(q_max_seq_len, 16),
            num_heads * batch_size,
        ),
        block_dim=(32, 16, 1),
    )

    @parameter
    @__copy_capture(p_buffer)
    fn input_fn_device[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[p_type, _simd_width]:
        var p_coord = Coord(coords)
        comptime assert p_coord.flat_rank == p_buffer.flat_rank
        return p_buffer.load[width=_simd_width](p_coord)

    _softmax_gpu[p_type, 1, 3, input_fn_device](
        Index(batch_size * num_heads, q_max_seq_len, num_keys),
        p_buffer.to_layout_tensor(),
        2,
        ctx,
    )
    var output_device = DeviceBuffer[output.dtype](
        ctx, output.ptr, output.numel(), owning=False
    )

    comptime kernel_1 = _bmm1_bs[
        QLayoutType = q.LayoutType,
        KVLayoutType = kv_input_row_offsets.LayoutType,
        type_of(v),
        p_type,
        output.dtype,
    ]
    ctx.enqueue_function_experimental[kernel_1](
        output_device,
        p_device,
        v,
        q_input_row_offsets,
        kv_input_row_offsets,
        q_max_seq_len,
        kv_max_seq_len,
        max_cache_size,
        num_heads,
        depth,
        Int(group),
        grid_dim=(
            ceildiv(depth, 32),
            ceildiv(q_max_seq_len, 16),
            num_heads * batch_size,
        ),
        block_dim=(32, 16, 1),
    )

    _ = p_device
