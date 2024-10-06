# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from math import ceildiv, exp
from sys import alignof, simdwidthof, sizeof

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
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.dim import Dim
from gpu.memory import AddressSpace, external_memory
from gpu.random import Random
from gpu.shuffle import _floorlog2, _static_log2, warp_reduce
from memory import UnsafePointer, stack_allocation

from utils.numerics import min_or_neg_inf

alias idx_t = DType.index
alias SEED = 0


# Define the TopK_2 structure to keep track of the top element per thread
@value
@register_passable("trivial")
struct TopK_2[T: DType]:
    var p: Int  # flattened index of the element
    var u: Scalar[T]  # value of the element

    fn __init__(inout self):
        self.p = -1
        self.u = min_or_neg_inf[T]()

    fn insert(inout self, elem: Scalar[T], elem_id: Int):
        if elem > self.u:
            self.u = elem
            self.p = elem_id


# Function to perform warp-level reduction to find the maximum TopK_2
@always_inline
@parameter
fn warp_reduce_topk_max[T: DType](val: TopK_2[T]) -> TopK_2[T]:
    """
    Performs warp-level reduction to find the maximum TopK_2 element.
    Uses shuffle down operations to efficiently compute the warp-wide
    maximum of TopK_2 values across all threads in a warp.

    Parameters:
        T: DType - Data type of the values being compared.

    Arguments:
        val: TopK_2[T] - TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T] - Maximum TopK_2 value across the warp.
    """
    var res = val

    # Shuffle down function for TopK_2 structure
    @parameter
    fn shuffle_down_topk2(v: TopK_2[T], offset: Int) -> TopK_2[T]:
        return TopK_2(
            u=shuffle_down(v.u, offset),  # u is the value
            p=int(
                shuffle_down(Scalar[DType.int32](v.p), offset)
            ),  # p is the index
        )

    @parameter
    fn reduce_fn(a: TopK_2[T], b: TopK_2[T]) -> TopK_2[T]:
        return a if a.u > b.u else b

    # Reimplement `warp_reduce` for TopK_2 reduce and shuffle function
    alias limit = _static_log2[WARP_SIZE]()

    @parameter
    for i in reversed(range(limit)):
        alias mask = 1 << i
        res = reduce_fn(res, shuffle_down_topk2(res, mask))

    return res


# Function to perform block-level reduction to find the maximum TopK_2
@always_inline
fn block_reduce_max[
    T: DType, BLOCK_SIZE: Int = 1024
](val: TopK_2[T]) -> TopK_2[T]:
    """
    Performs a block-level reduction to find the maximum TopK_2 element.

    This function takes a TopK_2 value from each thread in a block and performs
    a reduction to find the maximum across all threads. It uses shared memory
    and warp-level reductions to efficiently compute the block-wide maximum.

    Parameters:
        T: DType - The data type of the values being compared.
        BLOCK_SIZE: Int - The number of threads in the block. Default to max (1024).

    Arguments:
        val: TopK_2[T] - The TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T] - The maximum TopK_2 value across all threads in the block.

    Note:
    This function assumes that BLOCK_SIZE is a multiple of WARP_SIZE.
    It uses shared memory to store intermediate results and performs
    a final warp-level reduction to compute the block-wide maximum.
    """
    constrained[
        BLOCK_SIZE % WARP_SIZE == 0,
        "block size must be a multiple of the warp size",
    ]()

    # Calculate sizes for shared memory allocation
    alias p_width = simdwidthof[idx_t]()
    alias u_width = simdwidthof[Scalar[T]]()

    # Allocate shared memory for indices and values
    var p_sram = stack_allocation[
        (BLOCK_SIZE // WARP_SIZE) * p_width,
        Scalar[idx_t],
        address_space = AddressSpace.SHARED,
    ]()
    var u_sram = stack_allocation[
        (BLOCK_SIZE // WARP_SIZE) * u_width,
        Scalar[T],
        address_space = AddressSpace.SHARED,
    ]()

    # Calculate warp id and thread information
    var warp: UInt = ThreadIdx.x() // WARP_SIZE
    alias num_warps_needed = BLOCK_SIZE // WARP_SIZE

    # Each warp reduces its own TopK_2 value
    var warp_accum: TopK_2[T] = warp_reduce_topk_max[T](val)

    # Store warp-level results in shared memory
    if lane_id() == 0 and warp < num_warps_needed:
        # Note: Potential bank conflict for sub 4 byte data elements
        p_sram[int(warp) * p_width] = Scalar[idx_t](warp_accum.p)
        u_sram[int(warp) * u_width] = warp_accum.u
    barrier()

    # Load warp results into final warp for block-level reduction
    var block_accum = TopK_2[T]()
    var thread_in_final_warp = ThreadIdx.x() < (BlockDim.x() // WARP_SIZE)
    if thread_in_final_warp:
        var p_idx = p_sram[lane_id() * p_width]  # loaded value is a scalar
        block_accum = TopK_2[T](
            p=int(p_idx), u=u_sram[lane_id() * u_width]  # Convert back to int
        )
    else:
        # Initialize unused threads with dummy values
        block_accum.p = -1
        block_accum.u = min_or_neg_inf[T]()

    # Perform final warp-level reduction for block result
    return warp_reduce_topk_max[T](block_accum)


fn topk_stage1[
    T: DType,
](
    K: Int,
    in_buffer: UnsafePointer[Scalar[T]],
    local_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # Output buffer of size NUM_BLOCKS_STG1 * K
    local_topk_idxs: UnsafePointer[
        Int
    ],  # Output buffer of size NUM_BLOCKS_STG1 * K
    num_elements: Int,
):
    """
    Computes the Top-K elements within each block.

    This kernel function is the first stage of a two-stage Top-K algorithm.
    Each thread block processes a portion of the input data and finds its local top-K elements.
    The local top-K results are stored in global memory for further processing in stage 2.

    Parameters:
        T: Data type of the elements.

    Args:
        K: Number of top elements to select per block.
        in_buffer: Input buffer containing the elements to process.
        local_topk_vals: Output buffer to store the local top-K values.
        local_topk_idxs: Output buffer to store the indices of local top-K elements.
        num_elements: Total number of elements in the input buffer.

    Note:
        The output buffers (local_topk_vals and local_topk_idxs) should be of size NUM_BLOCKS_STG1 * K.
    """
    tid = ThreadIdx.x()
    bid = BlockIdx.x()
    block_size = BlockDim.x()
    global_tid = bid * BlockDim.x() + tid

    # Allocate shared memory for the values and indices
    var vals_smem_size = block_size
    var vals_sram = external_memory[
        Scalar[T],
        address_space = AddressSpace.SHARED,
        alignment = alignof[Scalar[T]](),
    ]()
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at vals' end.
    var idxs_sram = (vals_sram + vals_smem_size).bitcast[Int]()

    # Each thread loads its own element if within num_elements
    # TODO: This assumes that all thread in all blocks are assigned to
    # a single element in the broader tensor exclusively, without needing to
    # pack more elements, this could be a problem if tensor.flat_size > block_size*num_blocks
    if global_tid < num_elements:
        vals_sram[tid] = in_buffer[global_tid]
        idxs_sram[tid] = tid
    else:
        vals_sram[tid] = min_or_neg_inf[T]()  # Dead values
        idxs_sram[tid] = -1
    barrier()

    # Prepare for K iterations to find the local top-K elements
    for k in range(K):
        # Initialize each thread with its own TopK_2 value and index
        var partial = TopK_2[T]()
        partial.insert(vals_sram[tid], idxs_sram[tid])

        # Perform block-level reduction to find the maximum TopK_2
        var total = block_reduce_max[T](partial)

        if tid == 0:
            # Store the local top-K values and indices in global memory
            var g_idx = bid * BlockDim.x() + total.p
            local_topk_vals[bid * K + k] = total.u
            local_topk_idxs[bid * K + k] = g_idx

            # Remove the found maximum from consideration in the next iteration
            vals_sram[total.p] = min_or_neg_inf[T]()
            idxs_sram[total.p] = -1

        barrier()


fn topk_stage2[
    T: DType,
    sampling: Bool = True,
](
    K: Int,
    NUM_BLOCKS_STG1: Int,
    local_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # Input array of size NUM_BLOCKS_STG1 * K
    local_topk_idxs: UnsafePointer[
        Int
    ],  # Input array of size NUM_BLOCKS_STG1 * K
    global_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # sampling ? undefined : output array of size K
    global_topk_idxs: UnsafePointer[
        Int
    ],  # sampling ? sampled token : Output array of size K
):
    """
    Computes the global Top-K elements from the local Top-K results produced by stage 1.

    This kernel is designed to be executed with a single block, performing the final
    reduction step to obtain the global Top-K elements.

    Parameters:
        T: Data type of the elements.
        sampling: Bool - Whether to sample a token from the top-K distribution.

    Args:
        K: Number of top elements to select.
        NUM_BLOCKS_STG1: Number of blocks used in stage 1.
        local_topk_vals: Pointer to local Top-K values from stage 1 (size: NUM_BLOCKS_STG1 * K).
        local_topk_idxs: Pointer to local Top-K indices from stage 1 (size: NUM_BLOCKS_STG1 * K).
        global_topk_vals: Pointer to store the final global Top-K values (size: K).
        global_topk_idxs: Pointer to store the final global Top-K indices (size: K).

    The function uses shared memory to store and process the local Top-K results,
    and performs a block-level reduction to find the global Top-K elements.
    """
    # compute the total number of elements reduced from stage 1
    var num_elem_reduced = NUM_BLOCKS_STG1 * K

    var tid = ThreadIdx.x()
    # assert (BlockIdx.x() == 0)
    # assert (GridDim.x() == 1)

    # Handle the case where stage 1 is executed with a single block
    if NUM_BLOCKS_STG1 == 1:
        if tid < K and not sampling:
            global_topk_vals[tid] = local_topk_vals[tid]
            global_topk_idxs[tid] = local_topk_idxs[tid]
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
    s_val2 = (idxs_sram + vals_smem_size).bitcast[Scalar[T]]()
    s_id = (s_val2 + 2 * K).bitcast[Int]()  # 2* for warp align safety
    # [end] TODO Make this ONLY for sampling

    var s_sum = stack_allocation[
        1, Scalar[T], address_space = AddressSpace.SHARED
    ]()
    var max_logit = Scalar[T](0)

    # Load local top-K results into shared memory
    for i in range(tid, num_elem_reduced, BlockDim.x()):
        vals_sram[i] = local_topk_vals[i]
        idxs_sram[i] = i
    barrier()

    for k in range(K):
        # Re-initialize partial for each thread
        var partial = TopK_2[T]()
        # TODO: unroll this
        for i in range(tid, num_elem_reduced, BlockDim.x()):
            partial.insert(vals_sram[i], i)

        barrier()
        # Perform block-level reduction to find the maximum TopK_2
        var total: TopK_2[T] = block_reduce_max[T](partial)

        if tid == 0:

            @parameter
            if sampling:
                if k == 0:
                    max_logit = total.u

            # Remove the found maximum from consideration in the next iteration
            idxs_sram[total.p] = -1
            vals_sram[total.p] = min_or_neg_inf[T]()

            @parameter
            if sampling:
                s_id[k] = total.p
                total.u = exp(total.u - max_logit)
                s_val2[k] = total.u
                s_sum[0] += total.u
            else:
                # Store the global top-K values and indices
                global_topk_vals[k] = total.u
                global_topk_idxs[k] = local_topk_idxs[total.p]

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
                # TMP (debug - store prob of max logit)
                # global_topk_vals[ki] = exp_logit / softmax_norm
                r -= exp_logit
                if r <= 0.0 or ki == K - 1:
                    # uncomment below to return prob of max logit
                    # global_topk_vals[0] = exp_logit / softmax_norm
                    var idx: Int = s_id[ki]
                    global_topk_idxs[0] = local_topk_idxs[idx]
                    break


fn _topk_gpu[
    type: DType,
    rank: Int = 1,  # TODO (KERN-1016) support higher rank tensors
    sampling: Bool = True,
](
    ctx: DeviceContext,
    K: Int,  # num top elements to keep
    input_buf: NDBuffer[type, rank],
    device_local_topk_vals: NDBuffer[type, 1],
    device_local_topk_idxs: NDBuffer[idx_t, 1],
    out_vals: NDBuffer[type, 1],
    out_idxs: NDBuffer[idx_t, 1],
    axis: Int = -1,  # TODO support axis
    block_size: Int = 256,
) raises:
    """Computes the Top-K elements from the input tensor using a GPU-accelerated two-stage algorithm.

    This function implements a two-stage Top-K algorithm:
    1. Stage 1 (topk_stage1): Divides the input into blocks and computes local Top-K for each block.
    2. Stage 2 (topk_stage2): Merges the local Top-K results to obtain the global Top-K.

    Parameters:
        type: DType - The data type of the input tensor.
        rank: Int - The rank of the input tensor (default is 1).
        sampling: Bool - Whether to return token samples from topK dist (default is True).

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        K: Int - The number of top elements to keep.
        input_buf: NDBuffer[type, rank, DimList(N)]
            Input tensor as a device NDBuffer.
        device_local_topk_vals: NDBuffer[type, 1, DimList(num_blocks_1 * K)]
            Temporary buffer for locally reduced top-K values from stage 1.
        device_local_topk_idxs: NDBuffer[idx_t, 1, DimList(num_blocks_1 * K)]
            Temporary buffer for locally reduced top-K indices from stage 1.
        out_vals: NDBuffer[type, 1, DimList(K)]
            Output buffer on device for the K largest values.
        out_idxs: NDBuffer[idx_t, 1, DimList(K)]
            Output buffer on device for the indices of the K largest values.
        axis: Int
            Axis along which to compute the top-K (currently not supported, defaults to -1).
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).

    The implementation uses shared memory and warp-level primitives for efficient GPU execution.
    It's modeled from the following similar algos in [InternLM]
    (https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/sampling_topk_kernels.cu)
    and [TRT-LLM]
    (https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/samplingTopKKernels.cu).

    Note: Currently supports only 1D tensors. Higher rank tensor support is planned shortly.
    """
    # Use max number of threads per blocjk
    var N = input_buf.num_elements()
    # Define the number of blocks per grid
    var NUM_BLOCKS_STG1 = ceildiv(N, block_size)
    # Calculate max num bytes of shmem for each stage
    if block_size % WARP_SIZE != 0:
        # TODO: Need to pad in this case
        raise Error("block_size must be a multiple of WARP_SIZE")

    var shared_mem_bytes_1 = block_size * (
        sizeof[Scalar[type]]() + sizeof[idx_t]()
    )
    var num_elem_reduced = ceildiv(NUM_BLOCKS_STG1 * K, WARP_SIZE) * WARP_SIZE
    var num_bytes_sample_cache = 2 * K * (
        sizeof[Scalar[type]]() + sizeof[idx_t]()
    )
    var shared_mem_bytes_2 = num_elem_reduced * (
        sizeof[Scalar[type]]() + sizeof[idx_t]()
    ) + num_bytes_sample_cache

    # Compile the kernels
    var gpu_fn_stage1 = ctx.compile_function[topk_stage1[type], dump_ptx=False](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            shared_mem_bytes_1
        ),
    )
    # Define grid and block dimensions for stage 1
    var griddim_stage1 = Dim(NUM_BLOCKS_STG1)
    var blockdim_stage1 = Dim(block_size)

    # Enqueue the first kernel (stage 1)
    ctx.enqueue_function(
        gpu_fn_stage1,
        K,
        input_buf.data,
        device_local_topk_vals.data,
        device_local_topk_idxs.data,
        N,
        grid_dim=griddim_stage1,
        block_dim=blockdim_stage1,
        shared_mem_bytes=shared_mem_bytes_1,
    )

    var gpu_fn_stage2 = ctx.compile_function[
        topk_stage2[type, sampling], dump_ptx=False
    ](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            shared_mem_bytes_2
        ),
    )

    # Define grid and block dimensions for stage 2
    var griddim_stage2 = Dim(
        1
    )  # Single block since num_elements_stage2 is small
    var blockdim_stage2 = Dim(block_size)

    # Enqueue the second kernel (stage 2)
    ctx.enqueue_function(
        gpu_fn_stage2,
        K,
        NUM_BLOCKS_STG1,
        device_local_topk_vals.data,
        device_local_topk_idxs.data,
        out_vals.data,
        out_idxs.data,
        grid_dim=griddim_stage2,
        block_dim=blockdim_stage2,
        shared_mem_bytes=shared_mem_bytes_2,
    )
