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

from algorithm import parallelize
from complex import ComplexScalar
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.cluster import cluster_arrive_relaxed, cluster_wait
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, Vendor
from layout import Layout, LayoutTensor
from bit import next_power_of_two
from math import ceil
from runtime.asyncrt import parallelism_level
from sys.info import has_accelerator, num_logical_cores, size_of


from .utils import (
    _get_dtype,
    _get_twiddle_factors,
    _get_ordered_bases_processed_list,
    _get_flat_twfs,
    _mixed_radix_digit_reverse,
)

comptime _DEFAULT_DEVICE = "cpu" if not has_accelerator() else "gpu"


fn _check_layout_conditions[in_layout: Layout, out_layout: Layout]():
    constrained[len(in_layout) == 3, "in_layout must have rank 3"]()
    constrained[
        1 <= in_layout.shape[2].value() <= 2,
        "The layout should match one of: {(batches, sequence_length, 1), ",
        "(batches, sequence_length, 2)}",
    ]()
    constrained[len(out_layout) == 3, "out_layout must have rank 3"]()
    constrained[
        out_layout.shape[0].value() == in_layout.shape[0].value(),
        "out_layout must have the same number of batches in axis 0",
    ]()
    constrained[
        out_layout.shape[1].value() == in_layout.shape[1].value(),
        "out_layout must have the same sequence length in axis 1",
    ]()
    constrained[
        out_layout.shape[2].value() == 2,
        "out_layout must have the third axis equal to 2 for the complex values",
    ]()


fn _estimate_best_bases[
    in_layout: Layout, out_layout: Layout, target: StaticString
](out bases: List[UInt]):
    _check_layout_conditions[in_layout, out_layout]()
    comptime length = out_layout.shape[1].value()
    comptime max_radix_number = 64

    @parameter
    if not is_cpu[target]():
        # NOTE: The more threads the better, but estimate the best ranges such
        # that the thread number preferably fits in a block.
        comptime common_thread_block_size = 1024
        comptime min_radix_for_block = length // common_thread_block_size

        @parameter
        if length // max_radix_number <= common_thread_block_size:
            var radixes = range(min_radix_for_block, max_radix_number)
            # TODO: replace this with inline for generators once they properly
            # preallocate the sequence length (`[i for i in range(...)]`)
            bases = List[UInt](capacity=len(radixes))
            for i in range(UInt(len(radixes))):
                bases.append(i)
        else:
            return [7, 5, 3, 2]  # common prime factors
    else:
        # NOTE: We want the bases (length // base determines the amount of
        # threads per block) such that the amount of threads per block is as
        # close as possible to the num_logical_cores() for most stages. This is
        # guesswork.
        var close_pow2_num_cores = next_power_of_two(num_logical_cores())
        bases = List[UInt](capacity=close_pow2_num_cores)
        for i in reversed(range(2, close_pow2_num_cores + 1)):
            bases.append(UInt(i))


fn fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    inverse: Bool = False,
    target: StaticString = _DEFAULT_DEVICE,
    bases: List[UInt] = _estimate_best_bases[in_layout, out_layout, target](),
    # TODO: we'd need to know the cudaOccupancyMaxPotentialClusterSize for
    # every device to not use the portable 8
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters
    max_cluster_size: UInt = 8,
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
    *,
    cpu_workers: Optional[UInt] = None,
) raises:
    """Calculate the Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        inverse: Whether to run the inverse fourier transform.
        target: Target device ("cpu" or "gpu").
        bases: The list of bases for which to build the mixed-radix algorithm.
        max_cluster_size: In the case of NVIDIA GPUs, what the maximum cluster
            size for the device is.

    Args:
        output: The output tensor.
        x: The input tensor.
        ctx: The `DeviceContext`.
        cpu_workers: The amount of workers to use when running on CPU.

    Constraints:
        The layout should match one of: `{(batches, sequence_length, 1),
        (batches, sequence_length, 2)}`

    Notes:
        - This function automatically runs the rfft if the input is real-valued.
        - If the given bases list does not multiply together to equal the
        length, the builtin algorithm duplicates the biggest (CPU) / smallest
        (GPU) values that can still divide the length until reaching it.
        - For very long sequences on GPUs, it is worth considering bigger radix
        factors, due to the potential of being able to run the fft within a
        single block. Keep in mind that the amount of threads that will be
        launched is equal to the `sequence_length // smallest_base`.
    """
    _check_layout_conditions[in_layout, out_layout]()
    comptime batches = UInt(in_layout.shape[0].value())
    comptime sequence_length = UInt(in_layout.shape[1].value())
    comptime do_rfft = in_layout.shape[2].value() == 1
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()

    comptime bases_processed = _get_ordered_bases_processed_list[
        sequence_length, bases, target
    ]()
    comptime ordered_bases = bases_processed[0]
    comptime processed_list = bases_processed[1]

    @parameter
    fn _calc_total_offsets() -> Tuple[UInt, List[UInt]]:
        comptime last_base = ordered_bases[len(ordered_bases) - 1]
        var bases = materialize[ordered_bases]()
        var c = Int((sequence_length // last_base) * (last_base - 1))
        var offsets = List[UInt](capacity=c * len(bases))
        var val = UInt(0)
        for base in bases:
            offsets.append(val)
            val += (sequence_length // base) * (base - 1)
        return val, offsets^

    comptime total_offsets = _calc_total_offsets()
    comptime total_twfs = total_offsets[0]
    comptime twf_offsets = total_offsets[1]

    @parameter
    if is_cpu[target]():
        _cpu_fft_kernel_radix_n[
            length=sequence_length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
        ](output, x, cpu_workers=cpu_workers)
        return
    constrained[
        has_accelerator(), "The non-cpu implementation is for GPU only"
    ]()

    comptime gpu_info = ctx.default_device_info
    comptime max_threads_per_block = UInt(gpu_info.max_thread_block_size)
    comptime threads_per_sm = gpu_info.threads_per_multiprocessor
    constrained[
        threads_per_sm > 0,
        "Unknown number of threads per sm for the given device. ",
        "It is needed in order to run the gpu implementation.",
    ]()
    comptime max_threads_available = UInt(threads_per_sm * gpu_info.sm_count)
    comptime num_threads = sequence_length // ordered_bases[
        len(ordered_bases) - 1
    ]
    comptime num_blocks = UInt(
        ceil(num_threads / max_threads_per_block).cast[DType.uint]()
    )
    comptime shared_mem_size = UInt(gpu_info.shared_memory_per_multiprocessor)
    comptime output_size = UInt(size_of[out_dtype]()) * sequence_length * 2
    comptime twf_size = UInt(size_of[out_dtype]()) * total_twfs * 2

    @parameter
    if (
        num_threads <= max_threads_per_block
        and (output_size + twf_size) <= shared_mem_size
    ):
        comptime batch_size = max_threads_available // num_threads
        comptime func[batch_amnt: UInt] = _intra_block_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            x.origin,
            output.origin,
            length=sequence_length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
            warp_exec = UInt(gpu_info.warp_size) >= batch_amnt * num_threads,
        ]

        @parameter
        for i in range(batches // batch_size):
            comptime out_batch_layout = Layout.row_major(
                Int(batch_size), Int(sequence_length), 2
            )
            var out_batch = LayoutTensor[mut=True, out_dtype, out_batch_layout](
                output.ptr + output.stride[0]() * Int(i * batch_size)
            )
            comptime x_batch_layout = Layout.row_major(
                Int(batch_size),
                Int(sequence_length),
                in_layout.shape[2].value(),
            )
            var x_batch = LayoutTensor[mut=True, in_dtype, x_batch_layout](
                x.ptr + x.stride[0]() * Int(i * batch_size)
            )

            ctx.enqueue_function_checked[func[batch_size], func[batch_size]](
                out_batch,
                x_batch,
                grid_dim=(1, batch_size),
                block_dim=Int(num_threads),
            )

        comptime remainder = batches % batch_size

        @parameter
        if remainder > 0:
            comptime offset = (batches - remainder) * batch_size
            comptime out_batch_layout = Layout.row_major(
                Int(remainder), Int(sequence_length), 2
            )
            var out_batch = LayoutTensor[mut=True, out_dtype, out_batch_layout](
                output.ptr + output.stride[0]() * Int(offset)
            )
            comptime x_batch_layout = Layout.row_major(
                Int(remainder), Int(sequence_length), in_layout.shape[2].value()
            )
            var x_batch = LayoutTensor[mut=True, in_dtype, x_batch_layout](
                x.ptr + x.stride[0]() * Int(offset)
            )

            ctx.enqueue_function_checked[func[remainder], func[remainder]](
                out_batch,
                x_batch,
                grid_dim=(1, remainder),
                block_dim=Int(num_threads),
            )
    elif num_threads <= max_threads_available:
        # TODO: implement slicing and iterating over smaller batches
        comptime block_dim = UInt(
            ceil(num_threads / num_blocks).cast[DType.uint]()
        )
        _launch_inter_multiprocessor_fft[
            length=sequence_length,
            processed_list=processed_list,
            ordered_bases=ordered_bases,
            do_rfft=do_rfft,
            inverse=inverse,
            block_dim=block_dim,
            num_blocks=num_blocks,
            batches=batches,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
            max_cluster_size=max_cluster_size,
        ](output, x, ctx)
    else:
        # TODO: Implement for sequences > max_threads_available in the same GPU
        constrained[
            False,
            "fft for sequences longer than max_threads_available",
            "is not implemented yet. max_threads_available: ",
            String(max_threads_available),
        ]()


@always_inline
fn ifft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    target: StaticString = _DEFAULT_DEVICE,
    bases: List[UInt] = _estimate_best_bases[in_layout, out_layout, target](),
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
    *,
    cpu_workers: Optional[UInt] = None,
) raises:
    """Calculate the inverse Fast Fourier Transform.

    Parameters:
        in_dtype: The `DType` of the input tensor.
        out_dtype: The `DType` of the output tensor.
        in_layout: The `Layout` of the input tensor.
        out_layout: The `Layout` of the output tensor.
        target: Target device ("cpu" or "gpu").
        bases: The list of bases for which to build the mixed-radix algorithm.

    Args:
        output: The output tensor.
        x: The input tensor.
        ctx: The `DeviceContext`.
        cpu_workers: The amount of workers to use when running on CPU.

    Notes:
        This function is provided as a wrapper for the `fft` function. The
        documentation is more complete there.
    """
    fft[bases=bases, inverse=True, target=target](
        output, x, ctx, cpu_workers=cpu_workers
    )


# ===-----------------------------------------------------------------------===#
# inter_multiprocessor_fft
# ===-----------------------------------------------------------------------===#


fn _launch_inter_multiprocessor_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    length: UInt,
    processed_list: List[UInt],
    ordered_bases: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    num_blocks: UInt,
    block_dim: UInt,
    batches: UInt,
    total_twfs: UInt,
    twf_offsets: List[UInt],
    max_cluster_size: UInt,
](
    output: LayoutTensor[out_dtype, out_layout, out_origin],
    x: LayoutTensor[in_dtype, in_layout, in_origin],
    ctx: DeviceContext,
) raises:
    comptime twf_layout = Layout.row_major(Int(total_twfs), 2)
    comptime twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    var twfs = ctx.enqueue_create_buffer[out_dtype](twf_layout.size())
    ctx.enqueue_copy(twfs, twfs_array.unsafe_ptr())
    var twiddle_factors = LayoutTensor[mut=False, out_dtype, twf_layout](
        twfs.unsafe_ptr()
    )

    comptime grid_dim = (Int(num_blocks), Int(batches))
    comptime gpu_info = ctx.default_device_info
    comptime is_sm_90_or_newer = (
        gpu_info.vendor == Vendor.NVIDIA_GPU and gpu_info.compute >= 9.0
    )

    @parameter
    if is_sm_90_or_newer and num_blocks * batches <= max_cluster_size:
        # TODO: this should use distributed shared memory for twfs
        comptime func = _inter_block_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            twiddle_factors.layout,
            x.origin,
            output.origin,
            twiddle_factors.origin,
            twiddle_factors.address_space,
            length=length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            twf_offsets=twf_offsets,
        ]
        ctx.enqueue_function_checked[func, func](
            output,
            x,
            twiddle_factors,
            grid_dim=grid_dim,
            block_dim=Int(block_dim),
        )
    else:
        comptime func[b: Int] = _inter_multiprocessor_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            twiddle_factors.layout,
            x.origin,
            output.origin,
            twiddle_factors.origin,
            twiddle_factors.address_space,
            length=length,
            base = ordered_bases[b],
            ordered_bases=ordered_bases,
            processed = processed_list[b],
            do_rfft=do_rfft,
            inverse=inverse,
            twf_offset = twf_offsets[b],
        ]

        @parameter
        for b in range(len(ordered_bases)):
            ctx.enqueue_function_checked[func[b], func[b]](
                output,
                x,
                twiddle_factors,
                grid_dim=grid_dim,
                block_dim=Int(block_dim),
            )


fn _inter_multiprocessor_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    twf_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    *,
    length: UInt,
    base: UInt,
    ordered_bases: List[UInt],
    processed: UInt,
    do_rfft: Bool,
    inverse: Bool,
    twf_offset: UInt,
](
    batch_output: LayoutTensor[out_dtype, out_layout, out_origin],
    batch_x: LayoutTensor[in_dtype, in_layout, in_origin],
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
):
    """A kernel that assumes `sequence_length // smallest_base <=
    max_threads_available`."""
    comptime amnt_threads = length // base
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y
    comptime x_layout = Layout.row_major(
        Int(length), in_layout.shape[2].value()
    )
    var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * Int(block_num)
    )
    comptime block_out_layout = Layout.row_major(Int(length), 2)
    var output = LayoutTensor[
        mut=True, out_dtype, block_out_layout, batch_output.origin
    ](batch_output.ptr + batch_output.stride[0]() * Int(block_num))
    comptime x_out_layout = Layout.row_major(Int(base), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutAnyOrigin
    ].stack_allocation()

    comptime last_base = ordered_bases[len(ordered_bases) - 1]
    comptime total_threads = length // last_base
    comptime func = _radix_n_fft_kernel[
        out_dtype=out_dtype,
        out_layout = output.layout,
        out_address_space = output.address_space,
        twf_layout = twiddle_factors.layout,
        twf_origin = twiddle_factors.origin,
        twf_address_space = twiddle_factors.address_space,
        do_rfft=do_rfft,
        base=base,
        length=length,
        processed=processed,
        inverse=inverse,
        twf_offset=twf_offset,
        ordered_bases=ordered_bases,
    ]

    @parameter
    if amnt_threads == total_threads:
        func(output, x, global_i, twiddle_factors, x_out)
    else:
        if global_i < amnt_threads:  # is execution thread
            func(output, x, global_i, twiddle_factors, x_out)


# ===-----------------------------------------------------------------------===#
# inter_block_fft
# ===-----------------------------------------------------------------------===#


fn _inter_block_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    twf_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    *,
    length: UInt,
    ordered_bases: List[UInt],
    processed_list: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    twf_offsets: List[UInt],
](
    batch_output: LayoutTensor[out_dtype, out_layout, out_origin],
    batch_x: LayoutTensor[in_dtype, in_layout, in_origin],
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
):
    """A kernel that assumes `sequence_length // smallest_base <=
    max_threads_available`."""
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y
    comptime x_layout = Layout.row_major(
        Int(length), in_layout.shape[2].value()
    )
    var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * Int(block_num)
    )
    comptime block_out_layout = Layout.row_major(Int(length), 2)
    # TODO: this should use distributed shared memory for the intermediate output
    var output = LayoutTensor[
        mut=True, out_dtype, block_out_layout, batch_output.origin
    ](batch_output.ptr + batch_output.stride[0]() * Int(block_num))
    comptime last_base = ordered_bases[len(ordered_bases) - 1]
    comptime total_threads = length // last_base
    comptime x_out_layout = Layout.row_major(Int(ordered_bases[0]), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutAnyOrigin
    ].stack_allocation()

    @parameter
    for b in range(len(ordered_bases)):
        comptime base = ordered_bases[b]
        comptime processed = processed_list[b]
        comptime amnt_threads = length // base
        comptime func = _radix_n_fft_kernel[
            out_dtype=out_dtype,
            out_layout = output.layout,
            out_address_space = output.address_space,
            twf_layout = twiddle_factors.layout,
            twf_origin = twiddle_factors.origin,
            twf_address_space = twiddle_factors.address_space,
            do_rfft=do_rfft,
            base=base,
            length=length,
            processed=processed,
            inverse=inverse,
            twf_offset = twf_offsets[b],
            ordered_bases=ordered_bases,
        ]

        @parameter
        if amnt_threads == total_threads:
            func(output, x, global_i, twiddle_factors, x_out)
        else:
            if global_i < amnt_threads:  # is execution thread
                func(output, x, global_i, twiddle_factors, x_out)

        cluster_arrive_relaxed()
        cluster_wait()


# ===-----------------------------------------------------------------------===#
# intra_block_fft
# ===-----------------------------------------------------------------------===#


fn _intra_block_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    in_origin: ImmutOrigin,
    out_origin: MutOrigin,
    *,
    length: UInt,
    ordered_bases: List[UInt],
    processed_list: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    total_twfs: UInt,
    twf_offsets: List[UInt],
    warp_exec: Bool,
](
    batch_output: LayoutTensor[out_dtype, out_layout, out_origin],
    batch_x: LayoutTensor[in_dtype, in_layout, in_origin],
):
    """An FFT that assumes `sequence_length // smallest_base <=
    max_threads_per_block` and that `sequence_length` out_dtype items fit in
    a block's shared memory."""

    var local_i = thread_idx.x
    var block_num = block_dim.y * block_idx.y
    comptime x_layout = Layout.row_major(
        Int(length), in_layout.shape[2].value()
    )
    var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * Int(block_num)
    )
    comptime block_out_layout = Layout.row_major(Int(length), 2)
    var output = LayoutTensor[
        mut=True, out_dtype, block_out_layout, batch_output.origin
    ](batch_output.ptr + batch_output.stride[0]() * Int(block_num))
    comptime shared_f_layout = Layout.row_major(Int(length), 2)
    var shared_f = LayoutTensor[
        out_dtype, shared_f_layout, MutAnyOrigin
    ].stack_allocation()
    comptime twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    comptime twfs_layout = Layout.row_major(Int(total_twfs), 2)
    var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
        twfs_array.unsafe_ptr()
    )
    comptime last_base = ordered_bases[len(ordered_bases) - 1]
    comptime total_threads = length // last_base
    comptime x_out_layout = Layout.row_major(Int(ordered_bases[0]), 2)
    var x_out = LayoutTensor[
        out_dtype, x_out_layout, MutAnyOrigin
    ].stack_allocation()

    @parameter
    for b in range(len(ordered_bases)):
        comptime base = ordered_bases[b]
        comptime processed = processed_list[b]
        comptime amnt_threads = length // base
        comptime func = _radix_n_fft_kernel[
            out_dtype=out_dtype,
            out_layout = shared_f.layout,
            out_address_space = shared_f.address_space,
            twf_layout = twfs.layout,
            twf_address_space = twfs.address_space,
            do_rfft=do_rfft,
            base=base,
            length=length,
            processed=processed,
            inverse=inverse,
            twf_offset = twf_offsets[b],
            ordered_bases=ordered_bases,
        ]

        @parameter
        if amnt_threads == total_threads:
            func(shared_f, x, local_i, twfs, x_out)
        else:
            if local_i < amnt_threads:  # is execution thread
                func(shared_f, x, local_i, twfs, x_out)

        @parameter
        if not warp_exec:
            barrier()

    @parameter
    for i in range(last_base):
        comptime offset = i * total_threads
        var res = shared_f.load[width=2](Int(local_i + offset), 0)
        output.store(Int(local_i + offset), 0, res)

    @parameter
    if not warp_exec:
        barrier()


# ===-----------------------------------------------------------------------===#
# _cpu_fft_kernel_radix_n
# ===-----------------------------------------------------------------------===#


fn _cpu_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    length: UInt,
    ordered_bases: List[UInt],
    processed_list: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    total_twfs: UInt,
    twf_offsets: List[UInt],
](
    batch_output: LayoutTensor[mut=True, out_dtype, out_layout],
    batch_x: LayoutTensor[mut=False, in_dtype, in_layout],
    cpu_workers: Optional[UInt] = None,
):
    """An FFT that runs on the CPU."""

    comptime twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    comptime twfs_layout = Layout.row_major(Int(total_twfs), 2)
    var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
        twfs_array.unsafe_ptr()
    )

    @parameter
    for b in range(len(ordered_bases)):
        comptime base = ordered_bases[b]
        comptime processed = processed_list[b]
        comptime batches = UInt(in_layout.shape[0].value())
        comptime amnt_threads_per_block = length // base
        comptime amnt_threads = batches * amnt_threads_per_block

        @parameter
        fn _inner_kernel(global_i: Int):
            var block_num = UInt(global_i) // amnt_threads_per_block
            var local_i = UInt(global_i) % amnt_threads_per_block
            comptime block_out_layout = Layout.row_major(Int(length), 2)
            var output = LayoutTensor[
                mut=True, out_dtype, block_out_layout, batch_output.origin
            ](batch_output.ptr + batch_output.stride[0]() * Int(block_num))
            comptime x_layout = Layout.row_major(
                Int(length), in_layout.shape[2].value()
            )
            var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
                batch_x.ptr + batch_x.stride[0]() * Int(block_num)
            )
            var x_out_array = InlineArray[Scalar[out_dtype], Int(base * 2)](
                uninitialized=True
            )
            var x_out = LayoutTensor[
                mut=True, out_dtype, Layout.row_major(Int(base), 2)
            ](x_out_array.unsafe_ptr())

            _radix_n_fft_kernel[
                out_dtype=out_dtype,
                out_layout = output.layout,
                out_address_space = output.address_space,
                twf_layout = twfs.layout,
                twf_address_space = twfs.address_space,
                do_rfft=do_rfft,
                base=base,
                length=length,
                processed=processed,
                inverse=inverse,
                twf_offset = twf_offsets[b],
                ordered_bases=ordered_bases,
            ](output, x, local_i, twfs, x_out)

        parallelize[func=_inner_kernel](
            Int(amnt_threads),
            Int(cpu_workers.or_else(UInt(parallelism_level()))),
        )


# ===-----------------------------------------------------------------------===#
# radix implementation
# ===-----------------------------------------------------------------------===#


@always_inline
fn _radix_n_fft_kernel[
    in_dtype: DType,
    out_dtype: DType,
    out_layout: Layout,
    in_layout: Layout,
    out_origin: MutOrigin,
    out_address_space: AddressSpace,
    twf_layout: Layout,
    twf_origin: ImmutOrigin,
    twf_address_space: AddressSpace,
    x_out_layout: Layout,
    *,
    length: UInt,
    do_rfft: Bool,
    base: UInt,
    processed: UInt,
    inverse: Bool,
    twf_offset: UInt,
    ordered_bases: List[UInt],
](
    output: LayoutTensor[
        out_dtype, out_layout, out_origin, address_space=out_address_space
    ],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    local_i: UInt,
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
    x_out: LayoutTensor[mut=True, out_dtype, x_out_layout],
):
    """A generic Cooley-Tukey algorithm. It has most of the generalizable radix
    optimizations."""
    constrained[length >= base, "length must be >= base"]()
    comptime Sc = Scalar[_get_dtype[length]()]
    comptime offset = Sc(processed)
    var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (offset * Sc(base))

    comptime Co = ComplexScalar[out_dtype]
    comptime CoV = SIMD[out_dtype, 2]
    comptime is_even = length % 2 == 0

    @always_inline
    fn to_Co(v: CoV) -> Co:
        return UnsafePointer(to=v).bitcast[Co]()[]

    @always_inline
    fn to_CoV(c: Co) -> CoV:
        return UnsafePointer(to=c).bitcast[CoV]()[]

    @parameter
    fn _scatter_offsets(out res: SIMD[Sc.dtype, Int(base * 2)]):
        res = {}
        var idx = 0
        for i in range(base):
            res[idx] = i * offset * 2
            idx += 1
            res[idx] = i * offset * 2 + 1
            idx += 1

    @parameter
    fn _base_phasor[i: UInt, j: UInt](out res: Co):
        comptime base_twf = _get_twiddle_factors[base, out_dtype, inverse]()
        res = {1, 0}

        @parameter
        for _ in range(i):
            res *= base_twf[j - 1]

    @parameter
    @always_inline
    fn _twf_fma[twf: Co, is_j1: Bool](x_j_v: CoV, acc_v: CoV) -> CoV:
        var x_j = to_Co(x_j_v)
        var acc = to_Co(acc_v)
        var x_i: Co

        @parameter
        if do_rfft and twf.re == 1 and is_j1:  # Co(1, 0)
            x_i = Co(acc.re + x_j.re, 0)
        elif do_rfft and twf.re == 1:  # Co(1, 0)
            x_i = Co(acc.re + x_j.re, acc.im)
        elif twf.re == 1:  # Co(1, 0)
            x_i = x_j + acc
        elif is_even and do_rfft and twf.im == -1 and is_j1:  # Co(0, -1)
            x_i = Co(acc.re, -x_j.re)
        elif is_even and do_rfft and twf.im == -1:  # Co(0, -1)
            x_i = Co(acc.re, acc.im - x_j.re)
        elif is_even and twf.im == -1:  # Co(0, -1)
            x_i = Co(acc.re + x_j.im, acc.im - x_j.re)
        elif is_even and do_rfft and twf.re == -1 and is_j1:  # Co(-1, 0)
            x_i = Co(acc.re - x_j.re, 0)
        elif is_even and do_rfft and twf.re == -1:  # Co(-1, 0)
            x_i = Co(acc.re - x_j.re, acc.im)
        elif is_even and twf.re == -1:  # Co(-1, 0)
            x_i = Co(acc.re - x_j.re, acc.im - x_j.im)
        elif is_even and do_rfft and twf.im == 1 and is_j1:  # Co(0, 1)
            x_i = Co(acc.re, x_j.re)
        elif is_even and do_rfft and twf.im == 1:  # Co(0, 1)
            x_i = Co(acc.re, acc.im + x_j.re)
        elif is_even and twf.im == 1:  # Co(0, 1)
            x_i = Co(acc.re - x_j.im, acc.im + x_j.re)
        elif do_rfft:
            x_i = Co(twf.re.fma(x_j.re, acc.re), twf.im.fma(x_j.re, acc.im))
        else:
            x_i = twf.fma(x_j, acc)
        return to_CoV(x_i)

    @parameter
    fn _get_x[i: UInt]() -> SIMD[out_dtype, 2]:
        @parameter
        if processed == 1:
            # reorder input x(local_i) items to match F(current_item) layout
            var idx = Sc(local_i) * Sc(base) + Sc(i)

            var copy_from: Sc

            @parameter
            if base == length:  # do a DFT on the inputs
                copy_from = idx
            else:
                copy_from = _mixed_radix_digit_reverse[length, ordered_bases](
                    idx
                )

            @parameter
            if do_rfft:
                return {x.load[1](Int(copy_from), 0).cast[out_dtype](), 0}
            else:
                return x.load[2](Int(copy_from), 0).cast[out_dtype]()
        else:
            comptime step = Sc(i * offset)
            return output.load[2](Int(n + step), 0)

    var x_0 = _get_x[0]()

    @parameter
    for j in range(UInt(1), base):
        var x_j = _get_x[j]()

        @parameter
        if processed == 1:

            @parameter
            for i in range(base):
                comptime base_phasor = _base_phasor[i, j]()
                var acc: CoV

                @parameter
                if j == 1:
                    acc = x_0
                else:
                    acc = x_out.load[2](Int(i), 0)
                x_out.store(Int(i), 0, _twf_fma[base_phasor, j == 1](x_j, acc))
            continue

        var i0_j_twf_vec = twiddle_factors.load[2](
            Int(twf_offset + local_i * (base - 1) + (j - 1)), 0
        )
        var i0_j_twf = to_Co(i0_j_twf_vec)

        @parameter
        for i in range(base):
            comptime base_phasor = _base_phasor[i, j]()
            var twf: Co

            @parameter
            if base_phasor.re == 1:  # Co(1, 0)
                twf = i0_j_twf
            elif base_phasor.im == -1:  # Co(0, -1)
                twf = Co(i0_j_twf.im, -i0_j_twf.re)
            elif base_phasor.re == -1:  # Co(-1, 0)
                twf = -i0_j_twf
            elif base_phasor.im == 1:  # Co(0, 1)
                twf = Co(-i0_j_twf.im, i0_j_twf.re)
            else:
                twf = i0_j_twf * base_phasor

            var acc: CoV

            @parameter
            if j == 1:
                acc = x_0
            else:
                acc = x_out.load[2](Int(i), 0)

            x_out.store(Int(i), 0, to_CoV(twf.fma(to_Co(x_j), to_Co(acc))))

    @parameter
    if inverse and processed * base == length:  # last ifft stage
        comptime factor = (Float64(1) / Float64(length)).cast[out_dtype]()

        @parameter
        if base.is_power_of_two():
            x_out.ptr.store(x_out.load[Int(base * 2)](0, 0) * factor)
        else:

            @parameter
            for i in range(base):
                x_out.store(Int(i), 0, x_out.load[2](Int(i), 0) * factor)

    @parameter
    if base.is_power_of_two() and processed == 1:
        output.store(Int(n), 0, x_out.load[Int(base * 2)](0, 0))
    elif base.is_power_of_two() and out_address_space is AddressSpace.GENERIC:
        comptime offsets = _scatter_offsets()
        var v = x_out.load[Int(base * 2)](0, 0)
        output.ptr.offset(n * 2).scatter(offsets, v)
    else:

        @parameter
        for i in range(base):
            comptime step = Sc(i * offset)
            output.store(Int(n + step), 0, x_out.load[2](Int(i), 0))
