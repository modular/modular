# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from memory import UnsafePointer
from layout.layout import Layout
from layout.layout_tensor import (
    LayoutTensor,
    copy_dram_to_sram,
    copy_sram_to_dram,
)
from layout.tensor_builder import LayoutTensorBuild as tb, static
from sys import sizeof, alignof, simdwidthof
from layout.fillers import arange
from python import Python
from math import exp, ceildiv, align_up
from testing import assert_almost_equal
from layout.layout import UNKNOWN_VALUE
from layout import IntTuple
from layout.int_tuple import Index
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from gpu.host import DeviceContext
from memory import memset
from gpu.memory import external_memory, AddressSpace
from gpu.shuffle import (
    lane_group_sum,
    lane_group_max,
    warp_sum,
    shuffle_idx,
    _static_log2,
    shuffle_down,
    shuffle_idx,
)
from pathlib import Path
from layout.element import Element
from gpu import (
    barrier,
    ThreadIdx,
    BlockIdx,
    lane_id,
    WARP_SIZE,
    BlockDim,
    GridDim,
)
from layout.tensor_core import get_accum_type


fn mha_decoding_cpu_seq[
    mask_rank: Int,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    *,
    head_size: Int,
    num_heads: Int,
    group: Int,
](
    # [B, S, H, D]
    q_ptr: UnsafePointer[Scalar[q_type]],
    # [B, K, H, D]
    k_ptr: UnsafePointer[Scalar[k_type]],
    # [B, K, H, D]
    v_ptr: UnsafePointer[Scalar[v_type]],
    # [B, S, H, D]
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    num_keys: Int,
    head_idx: Int,
    batch_idx: Int,
):
    # we assume batch size is 1 also seq_length is 1
    alias batch_size = 1
    alias seq_length = 1
    alias kv_num_heads = num_heads // group
    var kv_head_idx = head_idx // group
    alias accum_type = get_accum_type[q_type]()
    var logits = UnsafePointer[Scalar[accum_type]].alloc(num_keys)

    var qk_max = Scalar[accum_type].MIN
    # calculate k.q
    var q = tb[q_type]().row_major[
        batch_size, seq_length, num_heads, head_size
    ]().view(q_ptr)
    var output = tb[output_type]().row_major[
        batch_size, seq_length, num_heads, head_size
    ]().view(output_ptr)

    var k = tb[k_type]().row_major(
        static[batch_size](),
        num_keys,
        static[kv_num_heads](),
        static[head_size](),
    ).view(k_ptr)

    var v = tb[v_type]().row_major(
        static[batch_size](),
        num_keys,
        static[kv_num_heads](),
        static[head_size](),
    ).view(v_ptr)

    var qi = q.tile[batch_size, seq_length, 1, head_size](0, 0, head_idx, 0)
    var oi = output.tile[batch_size, seq_length, 1, head_size](
        0, 0, head_idx, 0
    )
    for key_idx in range(num_keys):
        # this memory is contiguous so we can use LayoutTensor
        var k = k.tile[batch_size, 1, 1, head_size](0, key_idx, kv_head_idx, 0)
        var logits_i = Scalar[accum_type](0)
        for d in range(head_size):
            logits_i += (
                k[0, 0, 0, d][0].cast[accum_type]()
                * qi[0, 0, 0, d][0].cast[accum_type]()
            )
        logits_i = logits_i * scale.cast[logits_i.type]()
        logits[key_idx] = logits_i
        qk_max = max(logits_i, qk_max)

    # softmax
    var exp_sum = Scalar[accum_type](0)

    for i in range(num_keys):
        var exp_ = exp(logits[i] - qk_max)
        exp_sum += rebind[__type_of(exp_sum)](exp_)
        logits[i] = exp_

    for i in range(num_keys):
        logits[i] = logits[i] / exp_sum

    var accumulator = tb[accum_type]().layout[head_size]().alloc().fill(0)

    for key_idx in range(num_keys):
        var v = v.tile[batch_size, 1, 1, head_size](0, key_idx, kv_head_idx, 0)
        var logits_i = logits[key_idx]
        for d in range(head_size):
            accumulator[d] += logits_i * v[0, 0, 0, d][0].cast[accum_type]()

    for d in range(head_size):
        oi[d] = accumulator[d].cast[output_type]()
    logits.free()


fn mha_decoding_cpu[
    mask_rank: Int,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    *,
    head_size: Int,
    num_heads: Int,
    group: Int,
](
    # [B, S, H, D]
    q_ptr: UnsafePointer[Scalar[q_type]],
    # [B, K, H, D]
    k_ptr: UnsafePointer[Scalar[k_type]],
    # [B, K, H, D]
    v_ptr: UnsafePointer[Scalar[v_type]],
    # [B, S, H, D]
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    num_keys: Int,
    batch_size: Int,
):
    for head_idx in range(num_heads):
        for batch_idx in range(batch_size):
            mha_decoding_cpu_seq[
                mask_rank, head_size=head_size, num_heads=num_heads, group=group
            ](
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                scale,
                num_keys,
                head_idx,
                batch_idx,
            )


@always_inline
fn block_sum_broadcast[
    num_warps: Int
](
    reduction_smem: LayoutTensor[*_, address_space = AddressSpace.SHARED],
    val: Scalar,
) -> __type_of(val):
    var local_warp_sum = warp_sum(val)
    var warp = ThreadIdx.x() // WARP_SIZE
    var lane = lane_id()
    if lane == 0:
        reduction_smem[warp][0] = rebind[Scalar[reduction_smem.dtype]](
            local_warp_sum
        )
    barrier()
    # all the warps perform the same reduction so at the end
    # all threads in the block contains the same sum
    var sum = __type_of(val)(0)
    if lane < num_warps:
        sum = rebind[__type_of(sum)](reduction_smem[lane])
    sum = lane_group_sum[nthreads=num_warps](sum)
    # TODO: We can use shfl_xor in lane_group_max to remove the need of this broadcast
    return shuffle_idx(sum, 0)


@always_inline
fn inner_product[
    accum_type: DType
](x: LayoutTensor, y: LayoutTensor) -> Scalar[accum_type] as res:
    constrained[
        x.layout.all_dims_known()
        and y.layout.all_dims_known()
        and x.element_type.size == 1
    ]()
    var out = __type_of(res)(0)

    @parameter
    for i in range(x.shape[0]()):
        out += rebind[__type_of(out)](
            x[i][0].cast[accum_type]() * y[i][0].cast[accum_type]()
        )
    return out


fn mha_decoding_gpu_seq[
    mask_rank: Int,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    head_size: Int,
    num_heads: Int,
    group: Int,
    num_threads: Int,
    block_size: Int,  # number of rows of keys one warp processes, 32 means one row per thread
](
    # [B, S, H, D]
    q_ptr: UnsafePointer[Scalar[q_type]],
    # vllm uses BHKD for k and BHDK for v, this kernel is slightly different
    # from vllm because of this
    # [B, K, H, D]
    k_ptr: UnsafePointer[Scalar[k_type]],
    # [B, K, H, D]
    v_ptr: UnsafePointer[Scalar[v_type]],
    # [B, S, H, D]
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    num_keys: Int,
):
    alias accum_type = get_accum_type[q_type]()
    alias k_simdwidth = simdwidthof[k_type]()
    alias v_simdwidth = 4  # because one warp reads/writes head_size (= 128) elements, i.e. 4 elements per thread
    alias q_simdwidth = simdwidthof[q_type]()
    alias batch_size = 1
    alias seq_length = 1
    alias num_warps = num_threads // WARP_SIZE

    # one group of threads will process one row of keys, for block size = 32 thread_group_size == 1 so this
    # is unnecessary but keeping it here in case we need to optimize for lower head_size
    alias thread_group_size = WARP_SIZE // block_size
    alias num_thread_groups = num_threads // thread_group_size
    alias num_elems_per_thread = head_size // thread_group_size
    alias kv_num_heads = num_heads // group

    var warp_idx = ThreadIdx.x() // WARP_SIZE

    # check that block_size == 1
    debug_assert(GridDim.y() == 1)
    # var batch_idx = BlockIdx.y()

    var head_idx = BlockIdx.z()
    var kv_head_idx = head_idx // group

    # logits shared memory, used to store intermediate calculations
    var logits_smem_ptr = external_memory[
        Scalar[accum_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[accum_type, q_simdwidth]](),
    ]()

    # can't create layout tensor without this cast
    var logits_ptr = logits_smem_ptr.bitcast[
        Scalar[accum_type], alignment = alignof[Scalar[accum_type]]()
    ]()

    var logits = tb[accum_type]().layout(num_keys).shared().view(logits_ptr)

    var q = tb[q_type]().row_major[
        batch_size, seq_length, num_heads, head_size
    ]().view(q_ptr)

    var k = tb[k_type]().row_major(
        static[batch_size](),
        num_keys,
        static[kv_num_heads](),
        static[head_size](),
    ).view(k_ptr)

    var v = tb[v_type]().row_major(
        static[batch_size](),
        num_keys,
        static[kv_num_heads](),
        static[head_size](),
    ).view(v_ptr)

    var q_gmem_tile = q.tile[batch_size, seq_length, 1, head_size](
        0, 0, head_idx, 0
    ).reshape[Layout.row_major(thread_group_size, num_elems_per_thread)]()

    var q_smem_tile = tb[q_type]().row_major[
        thread_group_size, num_elems_per_thread
    ]().shared().alloc()

    alias thread_layout = Layout.row_major(
        thread_group_size, num_threads // thread_group_size
    )
    copy_dram_to_sram[thread_layout=thread_layout](q_smem_tile, q_gmem_tile)
    barrier()

    var qk_max = Scalar[accum_type].MIN
    var nblocks = ceildiv(num_keys, block_size)

    # This block calculates dot(q, k) for all the tokens and blocks.
    # The output is stored in shared memory, we also reduce qk_max over blocks here
    # constrained[num_warps == block_size]()
    var thread_group_idx = lane_id() // thread_group_size
    var thread_group_offset = lane_id() % thread_group_size
    for block_idx in range(warp_idx, nblocks, num_warps):
        # this memory is contiguous so we can use LayoutTensor
        var key_idx = thread_group_idx + block_idx * block_size
        var k_gmem = k.tile[batch_size, 1, 1, num_elems_per_thread](
            0, key_idx, kv_head_idx, thread_group_offset
        ).reshape[Layout(num_elems_per_thread)]()

        var k_register = tb[k_type]().layout[
            num_elems_per_thread
        ]().local().alloc()

        var q_register = tb[q_type]().layout[
            num_elems_per_thread
        ]().local().alloc()

        # copy sram to registers
        constrained[num_elems_per_thread % k_simdwidth == 0]()

        k_register.vectorize[k_simdwidth]().copy_from(
            k_gmem.vectorize[k_simdwidth]()
        )

        q_register.vectorize[q_simdwidth]().copy_from(
            q_smem_tile.tile[1, num_elems_per_thread](
                thread_group_offset, 0
            ).vectorize[1, q_simdwidth]()
        )

        # dot product
        var qk = inner_product[accum_type](k_register, q_register)
        # reduce in a thread_group
        qk = (
            lane_group_sum[nthreads=thread_group_size](qk)
            * scale.cast[qk.type]()
        )

        # 0th thread of the thread group writes to the shared memory and also
        # update qk_max
        if thread_group_offset == 0:
            if key_idx < num_keys:
                logits[key_idx] = qk
                qk_max = max(qk, qk_max)

    # To calculate qk_max, we will have to reduce over threads in a warp and then warp in a block
    # In one warp only threads with thread_group_offset == 0 contains the valid value of qk_max
    # Example: if num_threads_groups == 4 then only thread 0, 8, 16 and 24 contains the valid value

    # do reduction in in a warp
    @parameter
    for i in range(
        _static_log2[WARP_SIZE // 2](),
        _static_log2[thread_group_size]() - 1,
        -1,
    ):
        alias offset = 1 << i
        qk_max = max(qk_max, shuffle_down(qk_max, offset))

    # do reduction in a block
    var qk_max_reduction_smem = tb[accum_type]().layout[
        num_warps
    ]().shared().alloc()
    # reduce over warps
    if lane_id() == 0:
        qk_max_reduction_smem[warp_idx] = qk_max
    barrier()

    # this reduction operation is identical on all warps
    # since all threads need the value qk_max
    if lane_id() < num_warps:
        qk_max = rebind[__type_of(qk_max)](qk_max_reduction_smem[lane_id()])
    else:
        qk_max = Scalar[accum_type].MIN
    qk_max = lane_group_max[nthreads=num_warps](qk_max)
    # TODO: We can use shfl_xor in lane_group_max to remove the need of this broadcast
    qk_max = shuffle_idx(qk_max, 0)

    # compute softmax using all threads in a block
    # need softmax_simdwidth = 4 for best performance
    alias softmax_simdwidth = 4

    var exp_sum_ = SIMD[accum_type, softmax_simdwidth](0)

    for token in range(
        ThreadIdx.x(), num_keys // softmax_simdwidth, num_threads
    ):
        var logits_v = logits.tile[softmax_simdwidth](token).vectorize[
            softmax_simdwidth
        ]()
        var val = exp(logits_v[0] - qk_max)
        exp_sum_ += rebind[__type_of(exp_sum_)](val)
        logits_v[0] = val

    for token_ in range(
        ThreadIdx.x(), num_keys % softmax_simdwidth, num_threads
    ):
        var token = token_ + num_keys // softmax_simdwidth * softmax_simdwidth
        var val = exp(logits[token] - qk_max)
        exp_sum_[0] = val[0] + exp_sum_[0]
        logits[token] = val

    var exp_sum = exp_sum_.reduce_add()

    # shared memory used for block reduction
    var block_sum_reduction_smem = tb[accum_type]().layout[
        num_warps
    ]().shared().alloc()

    exp_sum = block_sum_broadcast[num_warps=num_warps](
        block_sum_reduction_smem, exp_sum
    )

    var inv_exp_sum = 1.0 / exp_sum
    for token in range(
        ThreadIdx.x(), num_keys // softmax_simdwidth, num_threads
    ):
        var logits_v = logits.tile[softmax_simdwidth](token).vectorize[
            softmax_simdwidth
        ]()
        logits_v[0] = logits_v[0] * inv_exp_sum

    for token_ in range(
        ThreadIdx.x(), num_keys % softmax_simdwidth, num_threads
    ):
        var token = token_ + num_keys // softmax_simdwidth * softmax_simdwidth
        logits[token] = logits[token] * inv_exp_sum

    # wait for the completion of softmax
    barrier()

    alias num_rows_per_thread = head_size // WARP_SIZE
    constrained[num_rows_per_thread % v_simdwidth == 0]()

    var accumulator = tb[accum_type]().layout[
        num_rows_per_thread
    ]().local().alloc().fill(0)

    for block_idx in range(warp_idx, nblocks, num_warps):
        var logits_reg = tb[accum_type]().layout[block_size]().local().alloc()
        logits_reg.vectorize[v_simdwidth]().copy_from(
            logits.tile[block_size](block_idx).vectorize[v_simdwidth]()
        )

        @parameter
        for token in range(block_size):
            var key_idx = token + block_size * block_idx
            # TODO: find a better way to check oob
            # this is faster than simple continue or break
            # but this and min in loading v
            # drops the performance by ~6%
            var logits_key = logits_reg[token][0] if key_idx < num_keys else 0

            @parameter
            for i in range(num_rows_per_thread // v_simdwidth):
                var v = v.tile[batch_size, 1, 1, num_rows_per_thread](
                    0,
                    int(min(Int32(key_idx), Int32(num_keys - 1))),
                    kv_head_idx,
                    lane_id(),
                ).reshape[Layout(head_size)]()
                var v_vec = v.vectorize[v_simdwidth]()[i]
                var accumulator_vec = accumulator.vectorize[v_simdwidth]()
                accumulator_vec[i] += rebind[accumulator_vec.element_type](
                    logits_key * v_vec.cast[accum_type]()
                )

    barrier()

    alias nstages = _static_log2[num_warps]()

    # reduce output over warps
    @parameter
    for i in reversed(range(1, nstages + 1)):
        alias mid = (1 << i) // 2
        if warp_idx >= mid and warp_idx < (i << 1):
            var dst = logits.tile[head_size](warp_idx - mid).tile[
                num_rows_per_thread
            ](lane_id())
            dst.vectorize[v_simdwidth]().copy_from(
                accumulator.vectorize[v_simdwidth]()
            )
        barrier()
        if warp_idx < mid:
            var src = logits.tile[head_size](warp_idx).tile[
                num_rows_per_thread
            ](lane_id())

            @parameter
            for t in range(num_rows_per_thread // v_simdwidth):
                var acc_vec = accumulator.vectorize[v_simdwidth]()
                acc_vec[t] += rebind[acc_vec.element_type](
                    src.vectorize[v_simdwidth]()[t]
                )

        barrier()

    var output = tb[output_type]().row_major[
        batch_size, seq_length, num_heads, head_size
    ]().view(output_ptr)

    var output_vec = output.tile[
        batch_size, seq_length, 1, num_rows_per_thread
    ](0, 0, head_idx, lane_id()).reshape[
        Layout(num_rows_per_thread)
    ]().vectorize[
        v_simdwidth
    ]()

    constrained[num_rows_per_thread % v_simdwidth == 0]()

    if warp_idx == 0:
        # warp zero writes to the global memory
        @parameter
        for i in range(num_rows_per_thread // v_simdwidth):
            output_vec[i] = rebind[output_vec.element_type](
                accumulator.vectorize[v_simdwidth]()[i].cast[output_type]()
            )


fn mha_decoding_gpu[
    mask_rank: Int,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    *,
    head_size: Int,
    num_heads: Int,
    group: Int,
](
    ctx: DeviceContext,
    # [B, S, H, D]
    q_ptr: UnsafePointer[Scalar[q_type]],
    # [B, K, H, D]
    k_ptr: UnsafePointer[Scalar[k_type]],
    # [B, K, H, D]
    v_ptr: UnsafePointer[Scalar[v_type]],
    # [B, S, H, D]
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    num_keys: Int,
    batch_size: Int,
) raises:
    alias num_threads = 128
    alias block_size = 32
    var func = ctx.compile_function[
        mha_decoding_gpu_seq[
            mask_rank,
            q_type,
            k_type,
            v_type,
            output_type,
            head_size,
            num_heads,
            group,
            num_threads,
            block_size,
        ],
        dump_ptx=False,
    ]()
    alias accum_type = get_accum_type[q_type]()
    alias num_warps = num_threads // WARP_SIZE
    var shared_mem_bytes = max(
        # shared memory to store number of logits
        # align up by block size because we don't check oob in copy_from when doing the second gevm
        align_up(num_keys, block_size),
        # shared memory for scratch space when doing block reductions
        head_size * num_warps // 2,
    ) * sizeof[accum_type]()
    ctx.enqueue_function(
        func,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        scale,
        num_keys,
        grid_dim=(1, batch_size, num_heads),
        block_dim=num_threads,
        shared_mem_bytes=shared_mem_bytes,
    )
