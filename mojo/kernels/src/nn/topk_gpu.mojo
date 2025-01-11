# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections import OptionalReg
from math import ceildiv, exp
from sys import alignof, simdwidthof, sizeof
from bit import log2_floor

from buffer import Buffer, NDBuffer
from buffer.dimlist import DimList
from builtin.io import _printf
from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    GridDim,
    ThreadIdx,
    barrier,
    lane_id,
    shuffle_down,
    warp_broadcast,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.dim import Dim
from gpu.memory import AddressSpace, external_memory
from gpu.random import Random
from gpu.shuffle import warp_reduce
from memory import UnsafePointer, stack_allocation
from nn.reshape import reshape

from utils import IndexList
from utils.numerics import max_or_inf, min_or_neg_inf

alias SEED = 0
alias DEBUG_FILE = False


@always_inline("nodebug")
fn _topk_dead_val[T: DType, largest: Bool = True]() -> Scalar[T]:
    @parameter
    if largest:
        return min_or_neg_inf[T]()
    else:
        return max_or_inf[T]()


# Define the TopK_2 structure to keep track of the top element per thread
@value
@register_passable("trivial")
struct TopK_2[T: DType, largest: Bool = True]:
    var p: Int  # flattened index of the element
    var u: Scalar[T]  # value of the element

    fn __init__(out self):
        self.p = -1
        self.u = _topk_dead_val[T, largest]()

    fn insert(mut self, elem: Scalar[T], elem_id: Int):
        @parameter
        if largest:
            if elem > self.u:
                self.u = elem
                self.p = elem_id
        else:
            if elem < self.u:
                self.u = elem
                self.p = elem_id


# Function to perform warp-level reduction to find the maximum TopK_2
@always_inline
@parameter
fn warp_reduce_topk[
    T: DType, largest: Bool
](val: TopK_2[T, largest]) -> TopK_2[T, largest]:
    """
    Performs warp-level reduction to find the maximum TopK_2 element.
    Uses shuffle down operations to efficiently compute the warp-wide
    maximum of TopK_2 values across all threads in a warp.

    Parameters:
        T: DType - Data type of the values being compared.
        largest: Bool - Whether to find the maximum or minimum value.

    Arguments:
        val: TopK_2[T, largest] - TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T, largest] - Maximum TopK_2 value across the warp.
    """
    var res = val

    # Shuffle down function for TopK_2 structure
    @parameter
    fn shuffle_down_topk2(
        v: TopK_2[T, largest], offset: Int
    ) -> TopK_2[T, largest]:
        return TopK_2[T, largest](
            u=shuffle_down(v.u, offset),  # u is the value
            p=Int(
                shuffle_down(Scalar[DType.int32](v.p), offset)
            ),  # p is the index
        )

    @parameter
    fn reduce_fn(
        a: TopK_2[T, largest], b: TopK_2[T, largest]
    ) -> TopK_2[T, largest]:
        @parameter
        if largest:
            return a if a.u > b.u else b
        else:
            return a if a.u < b.u else b

    # Reimplement `warp_reduce` for TopK_2 reduce and shuffle function
    alias limit = log2_floor(WARP_SIZE)

    @parameter
    for i in reversed(range(limit)):
        alias mask = 1 << i
        res = reduce_fn(res, shuffle_down_topk2(res, mask))

    return res


# Function to perform block-level reduction to find the maximum TopK_2
@always_inline
fn block_reduce_topk[
    T: DType, largest: Bool
](val: TopK_2[T, largest]) -> TopK_2[T, largest]:
    """
    Performs a block-level reduction to find the maximum TopK_2 element.

    This function takes a TopK_2 value from each thread in a block and performs
    a reduction to find the maximum across all threads. It uses shared memory
    and warp-level reductions to efficiently compute the block-wide maximum.

    Parameters:
        T: DType - The data type of the values being compared.
        largest: Bool - Whether to find the maximum or minimum value.

    Arguments:
        val: TopK_2[T, largest] - The TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T, largest] - The maximum TopK_2 value across all threads in the block.

    Note:
    This function assumes that BLOCK_SIZE is a multiple of WARP_SIZE.
    It uses shared memory to store intermediate results and performs
    a final warp-level reduction to compute the block-wide maximum.
    """
    alias MAX_BLOCK_SIZE = 1024
    constrained[
        MAX_BLOCK_SIZE % WARP_SIZE == 0,
        "block size must be a multiple of the warp size",
    ]()

    # Calculate sizes for shared memory allocation
    alias p_width = simdwidthof[DType.index]()
    alias u_width = simdwidthof[Scalar[T]]()

    # Allocate shared memory for indices and values
    var p_sram = stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * p_width,
        Scalar[DType.index],
        address_space = AddressSpace.SHARED,
    ]()
    var u_sram = stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * u_width,
        Scalar[T],
        address_space = AddressSpace.SHARED,
    ]()

    # Calculate warp id and thread information
    var warp: UInt = warp_broadcast(ThreadIdx.x // WARP_SIZE)
    alias num_warps_needed = MAX_BLOCK_SIZE // WARP_SIZE

    # Each warp reduces its own TopK_2 value
    var warp_accum: TopK_2[T, largest] = warp_reduce_topk[T, largest](val)

    # Store warp-level results in shared memory
    if lane_id() == 0 and warp < num_warps_needed:
        # Note: Potential bank conflict for sub 4 byte data elements
        p_sram[Int(warp) * p_width] = Scalar[DType.index](warp_accum.p)
        u_sram[Int(warp) * u_width] = warp_accum.u
    barrier()

    # Load warp results into final warp for block-level reduction
    var block_accum = TopK_2[T, largest]()
    var thread_in_final_warp = ThreadIdx.x < (BlockDim.x // WARP_SIZE)
    if thread_in_final_warp:
        var p_idx = p_sram[lane_id() * p_width]  # loaded value is a scalar
        block_accum = TopK_2[T, largest](
            p=Int(p_idx), u=u_sram[lane_id() * u_width]  # Convert back to int
        )
    else:
        # Initialize unused threads with dummy values
        block_accum.p = -1
        block_accum.u = _topk_dead_val[T, largest]()

    # Perform final warp-level reduction for block result
    return warp_reduce_topk[T, largest](block_accum)


fn topk_stage1[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    K: Int,
    num_elements: Int,
    num_blocks_per_input: Int,
    in_buffer: UnsafePointer[Scalar[T]],
    local_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # Output buffer of size num_blocks_per_input * K
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type]
    ],  # Output buffer of size num_blocks_per_input * K
):
    """
    Computes the Top-K elements within each block.

    This kernel function is the first stage of a two-stage Top-K algorithm.
    Each thread block processes a portion of the input data and finds its local top-K elements.
    The local top-K results are stored in global memory for further processing in stage 2.

    Parameters:
        T: Data type of the elements.
        out_idx_type: DType - The data type of the output indices.
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        K: Number of top elements to select per block.
        num_elements: Size of last dimension of input buffer (vocab size).
        num_blocks_per_input: Number of blocks used to process the input data.
        in_buffer: Input buffer containing the elements to process.
        local_topk_vals: Output buffer to store the local top-K values.
        local_topk_idxs: Output buffer to store the indices of local top-K elements.

    Note:
        The output buffers (local_topk_vals and local_topk_idxs) should be of size num_blocks_per_input * K.
    """
    tid = ThreadIdx.x
    bid = BlockIdx.x
    block_size = BlockDim.x

    batch_id = bid // num_blocks_per_input
    block_lane = bid % num_blocks_per_input

    _in_buffer = in_buffer + batch_id * num_elements

    # # Allocate shared memory for the values and indices
    var topk_sram = external_memory[
        TopK_2[T, largest],
        address_space = AddressSpace.SHARED,
        alignment = alignof[TopK_2[T, largest]](),
    ]()

    # Pack the topk_vals and topk_idxs into shared memory
    var block_offset = block_lane * block_size
    var stride = block_size * num_blocks_per_input
    topk_sram[tid] = TopK_2[T, largest]()
    for i in range(tid + block_offset, num_elements, stride):
        topk_sram[tid].insert(_in_buffer[i], i)

    barrier()

    # Prepare for K iterations to find the local top-K elements
    for k in range(K):
        # Initialize each thread with its own TopK_2 value and index
        var partial = topk_sram[tid]

        # Perform block-level reduction to find the maximum TopK_2
        var total = block_reduce_topk[T, largest](partial)

        if tid == 0:
            # Store the local top-K values and indices in global memory
            var vector_idx = total.p
            local_topk_vals[bid * K + k] = total.u
            local_topk_idxs[bid * K + k] = Scalar[DType.index](vector_idx).cast[
                out_idx_type
            ]()

            # Remove the found maximum from consideration in the next iteration
            var orig_tid = (vector_idx - block_offset) % stride
            topk_sram[orig_tid].u = _topk_dead_val[T, largest]()

        barrier()


@always_inline("nodebug")
fn _get_shmem_size_stg_1[type: DType](block_size: Int) -> Int:
    # Get dynamic shared memory size for stage 1
    return Int(block_size * sizeof[TopK_2[type]]())


fn topk_stage2[
    T: DType,
    out_idx_type: DType,
    sampling: Bool = True,
    largest: Bool = True,
](
    K: Int,
    num_blocks_per_input: Int,
    local_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # Input array of size n_batch * num_blocks_per_input * K
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type]
    ],  # Input array of size n_batch * num_blocks_per_input * K
    global_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # sampling ? undefined : output array of size K
    global_topk_idxs: UnsafePointer[
        Scalar[out_idx_type]
    ],  # sampling ? sampled token : Output array of size K
):
    """
    Computes the global Top-K elements from the local Top-K results produced by stage 1.

    This kernel is designed to be executed with a single block, performing the final
    reduction step to obtain the global Top-K elements.

    Parameters:
        T: Data type of the elements.
        out_idx_type: DType - The data type of the output indices.
        sampling: Bool - Whether to sample a token from the top-K distribution.
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        K: Number of top elements to select.
        num_blocks_per_input: Number of blocks used in stage 1.
        local_topk_vals: Pointer to local Top-K values from stage 1 (size: batch_size * num_blocks_per_input * K).
        local_topk_idxs: Pointer to local Top-K indices from stage 1 (size: batch_size * num_blocks_per_input * K).
        global_topk_vals: Pointer to store the final global Top-K values (size: batch_size * K).
        global_topk_idxs: Pointer to store the final global Top-K indices (size: batch_size * (1 if sampling else K)).

    The function uses shared memory to store and process the local Top-K results,
    and performs a block-level reduction to find the global Top-K elements.
    """
    # compute the total number of elements reduced from stage 1
    var num_elem_reduced = num_blocks_per_input * K

    var tid = ThreadIdx.x
    var batch_id = BlockIdx.x
    # assert (BlockIdx.x == 0)
    # assert (GridDim.x == 1)
    var batch_i_topk_vals = global_topk_vals + batch_id * K
    var batch_i_topk_idxs = global_topk_idxs + batch_id * (1 if sampling else K)
    var _local_topk_vals = local_topk_vals + batch_id * num_elem_reduced
    var _local_topk_idxs = local_topk_idxs + batch_id * num_elem_reduced

    # Handle the case where stage 1 is executed with a single block
    if num_blocks_per_input == 1:
        if tid < K and not sampling:
            batch_i_topk_vals[tid] = _local_topk_vals[tid]
            # cast to out_idx_type
            batch_i_topk_idxs[tid] = _local_topk_idxs[tid]
            return

    # Allocate shared memory for values and indices
    var num_e_rounded = ceildiv(num_elem_reduced, WARP_SIZE) * WARP_SIZE
    var vals_smem_size = num_e_rounded
    var vals_sram = external_memory[
        Scalar[T],
        address_space = AddressSpace.SHARED,
        alignment = alignof[Scalar[T]](),
    ]()
    var idxs_sram = (vals_sram + vals_smem_size).bitcast[Int]()

    # [begin] TODO Make this ONLY for sampling by defining on if @param sampling
    # var s_val2: UnsafePointer[Scalar[T], address_space = AddressSpace.SHARED]
    # var s_id: UnsafePointer[Int, address_space = AddressSpace.SHARED]
    # Storing the top-K logits in shmem for sampling
    s_id = (idxs_sram + vals_smem_size).bitcast[
        Int
    ]()  # 2* for warp align safety
    s_val2 = (s_id + 2 * K).bitcast[Scalar[T]]()
    # [end] TODO Make this ONLY for sampling

    var s_sum = stack_allocation[
        1, Scalar[T], address_space = AddressSpace.SHARED
    ]()
    s_sum[0] = Scalar[T](0)
    var max_logit = Scalar[T](0)

    # Cache local top-K results from stage 1 into shared memory
    for i in range(tid, num_elem_reduced, BlockDim.x):
        vals_sram[i] = _local_topk_vals[i]
        idxs_sram[i] = i
    barrier()

    for k in range(K):
        # Re-initialize partial for each thread
        var partial = TopK_2[T, largest]()
        # TODO: unroll this
        for i in range(tid, num_elem_reduced, BlockDim.x):
            partial.insert(vals_sram[i], i)

        barrier()
        # Perform block-level reduction to find the maximum TopK_2
        var total: TopK_2[T, largest] = block_reduce_topk[T, largest](partial)

        if tid == 0:

            @parameter
            if sampling:
                if k == 0:
                    max_logit = total.u

            # Remove the found maximum from consideration in the next iteration
            idxs_sram[total.p] = -1
            vals_sram[total.p] = _topk_dead_val[T, largest]()

            @parameter
            if sampling:
                batch_i_topk_vals[k] = total.u
                s_id[k] = total.p
                total.u = exp(total.u - max_logit)
                s_val2[k] = total.u
                s_sum[0] += total.u
            else:
                # Store the global top-K values and indices
                batch_i_topk_vals[k] = total.u
                batch_i_topk_idxs[k] = _local_topk_idxs[total.p]

            # Early exit if no valid index
            if total.p == -1:
                break
        barrier()

    # do sampling
    @parameter
    if sampling:
        if tid == 0:
            var rng_state = Random(seed=SEED)
            var rng = rng_state.step_uniform()
            var softmax_norm = s_sum[0]
            var r = softmax_norm * rng[0].cast[T]()
            for ki in range(K):
                var exp_logit = s_val2[ki]

                # TMP (debug - store prob of largest logit)
                @parameter
                if DEBUG_FILE:
                    batch_i_topk_vals[ki] = exp_logit / softmax_norm
                r -= exp_logit
                if r <= 0.0 or ki == K - 1:
                    # uncomment below to return prob of largest logit
                    # batch_i_topk_vals[0] = exp_logit / softmax_norm
                    var idx: Int = s_id[ki]
                    batch_i_topk_idxs[0] = _local_topk_idxs[idx]
                    break


fn _topk_gpu[
    type: DType,
    rank: Int,
    out_idx_type: DType = DType.index,
    sampling: Bool = True,
    largest: Bool = True,
](
    ctx: DeviceContext,
    K: Int,  # num top elements to keep
    input_buf: NDBuffer[type, rank],
    device_local_topk_vals: NDBuffer[type, rank],
    device_local_topk_idxs: NDBuffer[out_idx_type, rank],
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[out_idx_type, rank],
    block_size: Int = 256,
    num_blocks_per_input: OptionalReg[Int] = None,
) raises:
    """Computes the Top-K elements from the input tensor using a GPU-accelerated two-stage algorithm.

    This function implements a two-stage Top-K algorithm:
    1. Stage 1 (topk_stage1): Divides the input into blocks and computes local Top-K for each block.
    2. Stage 2 (topk_stage2): Merges the local Top-K results to obtain the global Top-K.

    Parameters:
        type: DType - The data type of the input tensor.
        rank: Int - The rank of the input tensor (must be 2 right now, first dim is batch size).
        out_idx_type: DType - The data type of the output indices (default is DType.index).
        sampling: Bool - Whether to return token samples from topK dist (default is True).
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        K: Int - The number of top elements to keep.
        input_buf: NDBuffer[type, rank, DimList(batch_size,N)]
            Input tensor as a device NDBuffer.
        device_local_topk_vals: NDBuffer[type, 2, DimList(batch_size, num_blocks_per_input * K)]
            Temporary buffer for locally reduced top-K values from stage 1.
        device_local_topk_idxs: NDBuffer[DType.index, 2, DimList(batch_size, num_blocks_per_input * K)]
            Temporary buffer for locally reduced top-K indices from stage 1.
        out_vals: NDBuffer[type, 2, DimList(batch_size, K)]
            Output buffer on device for the K largest values.
        out_idxs: NDBuffer[DType.index, 2, DimList(batch_size, 1 if sampling else K)]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: OptionalReg[Int]
            Number of blocks per input (default computed from input size and block size).
            This is the equivalent of "BLOCKS_PER_BEAM" in TRT-LLM kernel allowing for much larger
            batch sizes through packing several elements per thread in the first stage.

    The implementation uses shared memory and warp-level primitives for efficient GPU execution.
    It's modeled from the following similar algos in [InternLM]
    (https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/sampling_topk_kernels.cu)
    and [TRT-LLM]
    (https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/samplingTopKKernels.cu).

    """
    constrained[rank == 2, "rank must be 2"]()
    constrained[
        not (sampling and not largest),
        "sampling not supported for largest=False",
    ]()
    # Use largest number of threads per block
    var batch_size = input_buf.get_shape()[0] if rank == 2 else 1
    var N = input_buf.get_shape()[1]
    # Define the number of blocks per grid
    var num_blocks_per_input_: Int = ceildiv(
        N, block_size
    ) if not num_blocks_per_input else num_blocks_per_input.value()
    # Calculate largest num bytes of shmem for each stage
    if block_size % WARP_SIZE != 0:
        # TODO: Need to pad in this case
        raise Error("block_size must be a multiple of WARP_SIZE")

    var shared_mem_bytes_1 = _get_shmem_size_stg_1[type](block_size)

    # Compile the kernels
    var gpu_fn_stage1 = ctx.compile_function[
        topk_stage1[type, out_idx_type, largest], dump_asm=False
    ]()
    # Define grid and block dimensions for stage 1
    var griddim_stage1 = Dim(num_blocks_per_input_ * batch_size)
    var blockdim_stage1 = Dim(block_size)

    # Enqueue the first kernel (stage 1)
    @parameter
    if DEBUG_FILE:
        _printf["[DEBUG] stg 1 grid_dim: %d\n"](griddim_stage1)
        _printf["[DEBUG] stg 1 block_dim: %d\n"](blockdim_stage1)
        _printf["[DEBUG] stg 1 shared_mem_bytes: %d\n"](shared_mem_bytes_1)
        _printf["[DEBUG] stg 1 num_blocks_per_input: %d\n"](
            num_blocks_per_input_
        )

    ctx.enqueue_function(
        gpu_fn_stage1,
        K,
        N,
        num_blocks_per_input_,
        input_buf.data,
        device_local_topk_vals.data,
        device_local_topk_idxs.data,
        grid_dim=griddim_stage1,
        block_dim=blockdim_stage1,
        shared_mem_bytes=shared_mem_bytes_1,
    )

    var num_elem_reduced = ceildiv(
        num_blocks_per_input_ * K, WARP_SIZE
    ) * WARP_SIZE
    var num_bytes_sample_cache = K * (
        sizeof[Scalar[type]]() + sizeof[DType.index]()
    )
    var shared_mem_bytes_2 = num_elem_reduced * (
        sizeof[Scalar[type]]() + sizeof[DType.index]()
    ) + num_bytes_sample_cache
    shared_mem_bytes_2 = Int(
        ceildiv(shared_mem_bytes_2, WARP_SIZE) * WARP_SIZE
    )  # align to warp size

    var gpu_fn_stage2 = ctx.compile_function[
        topk_stage2[type, out_idx_type, sampling, largest], dump_asm=False
    ]()

    # Define grid and block dimensions for stage 2
    var griddim_stage2 = Dim(
        batch_size
    )  # Single block since num_elements_stage2 is small
    var blockdim_stage2 = Dim(block_size)

    @parameter
    if DEBUG_FILE:
        _printf["[DEBUG] stg2 num_blocks_per_input_: %d\n"](
            num_blocks_per_input_
        )
        _printf["[DEBUG] stg2 grid_dim: %d\n"](griddim_stage2)
        _printf["[DEBUG] stg2 block_dim: %d\n"](blockdim_stage2)
        _printf["[DEBUG] stg2 shared_mem_bytes: %d\n"](shared_mem_bytes_2)

    # Enqueue the second kernel (stage 2)
    ctx.enqueue_function(
        gpu_fn_stage2,
        K,
        num_blocks_per_input_,
        device_local_topk_vals.data,
        device_local_topk_idxs.data,
        out_vals.data,
        out_idxs.data,
        grid_dim=griddim_stage2,
        block_dim=blockdim_stage2,
        shared_mem_bytes=shared_mem_bytes_2,
    )


@always_inline
fn topk_gpu[
    type: DType,
    rank: Int,
    out_idx_type: DType = DType.index,
    sampling: Bool = True,
    largest: Bool = True,
](
    ctx: DeviceContext,
    K: Int,  # num top elements to keep
    input: NDBuffer[type, rank],
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[out_idx_type, rank],
    block_size: OptionalReg[Int] = None,
    num_blocks_per_input: OptionalReg[Int] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with/without sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume or the top K values and indices across the tensor.

    Parameters:
        type: DType - The data type of the input tensor.
        rank: Int - The rank of the input tensor.
        out_idx_type: DType - The data type of the output indices (default is DType.index).
        sampling: Bool - Whether to return token samples from topK dist (default is True).
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        K: Int - The number of top elements to keep.
        input: NDBuffer[type, rank]
            Input tensor as a device NDBuffer.
        out_vals: NDBuffer[type, rank]
            Output buffer on device for the K largest values.
        out_idxs: NDBuffer[DType.index, rank]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
            Last dimension is 1 if sampling is True, otherwise K.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: OptionalReg[Int]
            Number of blocks per input (default computed from input size and block size).
            This is the equivalent of "BLOCKS_PER_BEAM" in TRT-LLM kernel allowing for much larger
            batch sizes through packing several elements per thread in the first stage.
    """
    constrained[rank > 0, "Input rank must be positive"]()
    var orig_in_shape: IndexList[rank] = input.get_shape()
    var N = orig_in_shape[rank - 1]
    var last_idx_dim = 1 if sampling else K

    # heuristic to set block size
    var block_size_: Int
    if input.size() <= 1024 * 64 * 3:
        block_size_ = 256
    elif input.size() <= 32000 * 256:
        block_size_ = 512
    else:
        block_size_ = 1024
    block_size_ = block_size.value() if block_size else block_size_

    # This section handles different input ranks by reshaping to a 2D tensor
    var internal_bs: Int  # Internal batch size
    alias internal_rank = 2  # We always reshape to 2D for internal processing
    var internal_input: NDBuffer[type, internal_rank]
    var internal_out_idxs: NDBuffer[out_idx_type, internal_rank]
    var internal_out_vals: NDBuffer[type, internal_rank]

    @parameter
    if rank == 1:
        # Handle 1D input: treat it as a single batch with one element
        internal_bs = 1
        var internal_in_shape = IndexList[internal_rank](1, input.size())
        var internal_out_vals_shape = IndexList[internal_rank](1, K)
        var internal_out_idxs_shape = IndexList[internal_rank](1, last_idx_dim)

        # Reshape 1D inputs to 2D
        internal_input = reshape(input, internal_in_shape)
        internal_out_idxs = reshape(out_idxs, internal_out_idxs_shape)
        internal_out_vals = reshape(out_vals, internal_out_vals_shape)
    elif rank == internal_rank:
        # Input is already 2D, no reshaping needed
        internal_bs = orig_in_shape[0]
        internal_input = rebind[NDBuffer[type, internal_rank]](input)
        internal_out_idxs = rebind[NDBuffer[out_idx_type, internal_rank]](
            out_idxs
        )
        internal_out_vals = rebind[NDBuffer[type, internal_rank]](out_vals)
    else:  # rank > 2
        # Handle higher dimensional inputs by flattening all but the last dimension
        var _last_dim = orig_in_shape[rank - 1]
        internal_bs = Int(orig_in_shape.flattened_length() / _last_dim)

        var internal_in_shape = IndexList[internal_rank](internal_bs, _last_dim)
        var internal_out_idxs_shape = IndexList[internal_rank](
            internal_bs, last_idx_dim
        )
        var internal_out_vals_shape = IndexList[internal_rank](internal_bs, K)

        # Reshape higher dimensional inputs to 2D
        internal_input = reshape(input, internal_in_shape)
        internal_out_idxs = reshape(out_idxs, internal_out_idxs_shape)
        internal_out_vals = reshape(out_vals, internal_out_vals_shape)

    # Calculate the number of blocks per input
    var num_blocks_per_input_ = min(
        ceildiv(N, block_size_), 8
    ) if not num_blocks_per_input else num_blocks_per_input.value()

    # Define shape for the kernel's internal cache buffers
    var internal_cache_shape = DimList(internal_bs, num_blocks_per_input_ * K)

    # Create temporary buffer for local top-K values
    var internal_vals_buf = ctx.enqueue_create_buffer[type](
        Int(internal_cache_shape.product())
    )
    var device_local_topk_vals = NDBuffer[type, internal_rank](
        internal_vals_buf.ptr, internal_cache_shape
    )

    # Create temporary buffer for local top-K indices
    var internal_idxs_buf = ctx.enqueue_create_buffer[out_idx_type](
        Int(internal_cache_shape.product())
    )
    var device_local_topk_idxs = NDBuffer[out_idx_type, internal_rank](
        internal_idxs_buf.ptr, internal_cache_shape
    )

    @parameter
    if DEBUG_FILE:
        print("[DEBUG] internal_input shape: ", internal_input.get_shape())
        print(
            "[DEBUG] internal_out_vals shape: ", internal_out_vals.get_shape()
        )
        print(
            "[DEBUG] internal_out_idxs shape: ", internal_out_idxs.get_shape()
        )
        print("[DEBUG] internal_cache_shape: ", internal_cache_shape)

    _topk_gpu[sampling=sampling, largest=largest](
        ctx,
        K,
        internal_input,
        device_local_topk_vals,
        device_local_topk_idxs,
        internal_out_vals,
        internal_out_idxs,
        block_size=block_size_,
        num_blocks_per_input=num_blocks_per_input_,
    )

    _ = internal_vals_buf^
    _ = internal_idxs_buf^


@always_inline
fn topk_fused_sampling_gpu[
    type: DType,
    rank: Int,
    out_idx_type: DType,
](
    ctx: DeviceContext,
    K: Int,  # num top elements to keep
    input: NDBuffer[type, rank],
    out_idxs: NDBuffer[out_idx_type, rank],
    block_size: OptionalReg[Int] = None,
    num_blocks_per_input: OptionalReg[Int] = None,
) raises:
    """
    Top K algorithm with fused sampling.
    Returns the sampled indices from the Top-K of the innermost
    dimension of the input tensor for each row/subvolume.
    """

    var out_vals_shape = input.get_shape()
    out_vals_shape[rank - 1] = K
    var out_vals_buf = ctx.enqueue_create_buffer[type](
        out_vals_shape.flattened_length()
    )
    var out_vals = NDBuffer[type, rank](out_vals_buf.ptr, out_vals_shape)

    topk_gpu[sampling=True, largest=True](
        ctx,
        K,
        input,
        out_vals,
        out_idxs,
        block_size=block_size,
        num_blocks_per_input=num_blocks_per_input,
    )

    _ = out_vals_buf^
