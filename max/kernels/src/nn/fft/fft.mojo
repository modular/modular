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
from layout.tensor_builder import LayoutTensorBuild as tb
from bit import next_power_of_two
from math import ceil
from runtime.asyncrt import parallelism_level
from sys.info import has_accelerator, num_logical_cores


from .utils import (
    _get_dtype,
    _get_twiddle_factors,
    _get_ordered_bases_processed_list,
    _get_flat_twfs,
    _mixed_radix_digit_reverse,
)

alias _DEFAULT_DEVICE = "cpu" if not has_accelerator() else "gpu"


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
    alias length = out_layout.shape[1].value()
    alias max_radix_number = 64

    @parameter
    if not is_cpu[target]():
        # NOTE: The more threads the better, but estimate the best ranges such
        # that the thread number preferably fits in a block.
        alias common_thread_block_size = 1024
        alias min_radix_for_block = length // common_thread_block_size

        @parameter
        if length // max_radix_number <= common_thread_block_size:
            var radixes = range(min_radix_for_block, max_radix_number)
            # TODO: replace this with inline for generators once they properly
            # preallocate the sequence length (`[i for i in range(...)]`)
            bases = List[UInt](capacity=len(radixes))
            for i in range(len(radixes)):
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
            bases.append(i)


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
    alias batches = in_layout.shape[0].value()
    alias sequence_length = in_layout.shape[1].value()
    alias do_rfft = in_layout.shape[2].value() == 1
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()

    alias bases_processed = _get_ordered_bases_processed_list[
        sequence_length, bases, target
    ]()
    alias ordered_bases = bases_processed[0]
    alias processed_list = bases_processed[1]

    @parameter
    fn _calc_total_offsets() -> (UInt, List[UInt]):
        alias last_base = ordered_bases[len(ordered_bases) - 1]
        var bases = materialize[ordered_bases]()
        var c = (sequence_length // last_base) * (last_base - 1) * len(bases)
        var offsets = List[UInt](capacity=c)
        var val = UInt(0)
        for base in bases:
            offsets.append(val)
            val += (sequence_length // base) * (base - 1)
        return val, offsets^

    alias total_offsets = _calc_total_offsets()
    alias total_twfs = total_offsets[0]
    alias twf_offsets = total_offsets[1]

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

    alias gpu_info = ctx.default_device_info
    alias max_threads_per_block = gpu_info.max_thread_block_size
    alias threads_per_sm = gpu_info.threads_per_sm
    constrained[
        threads_per_sm > 0,
        "Unknown number of threads per sm for the given device. ",
        "It is needed in order to run the gpu implementation.",
    ]()
    alias max_threads_available = threads_per_sm * gpu_info.sm_count
    alias num_threads = sequence_length // ordered_bases[len(ordered_bases) - 1]
    alias num_blocks = UInt(
        ceil(num_threads / max_threads_per_block).cast[DType.uint]()
    )
    alias shared_mem_size = gpu_info.shared_memory_per_multiprocessor
    alias output_size = out_dtype.size_of() * sequence_length * 2
    alias twf_size = out_dtype.size_of() * total_twfs * 2

    @parameter
    if (
        num_threads <= max_threads_per_block
        and (output_size + twf_size) <= shared_mem_size
    ):
        alias batch_size = max_threads_available // num_threads
        alias func[batch_amnt: UInt] = _intra_block_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
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
            alias out_batch_layout = Layout.row_major(
                batch_size, sequence_length, 2
            )
            var out_batch = LayoutTensor[mut=True, out_dtype, out_batch_layout](
                output.ptr + output.stride[0]() * i * batch_size
            )
            alias x_batch_layout = Layout.row_major(
                batch_size, sequence_length, in_layout.shape[2].value()
            )
            var x_batch = LayoutTensor[mut=True, in_dtype, x_batch_layout](
                x.ptr + x.stride[0]() * i * batch_size
            )

            ctx.enqueue_function[func[batch_size]](
                out_batch,
                x_batch,
                grid_dim=(1, batch_size),
                block_dim=num_threads,
            )

        alias remainder = batches % batch_size

        @parameter
        if remainder > 0:
            alias offset = (batches - remainder) * batch_size
            alias out_batch_layout = Layout.row_major(
                remainder, sequence_length, 2
            )
            var out_batch = LayoutTensor[mut=True, out_dtype, out_batch_layout](
                output.ptr + output.stride[0]() * offset
            )
            alias x_batch_layout = Layout.row_major(
                remainder, sequence_length, in_layout.shape[2].value()
            )
            var x_batch = LayoutTensor[mut=True, in_dtype, x_batch_layout](
                x.ptr + x.stride[0]() * offset
            )

            ctx.enqueue_function[func[remainder]](
                out_batch,
                x_batch,
                grid_dim=(1, remainder),
                block_dim=num_threads,
            )
    elif num_threads <= max_threads_available:
        alias block_dim = UInt(
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
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
) raises:
    alias twf_layout = Layout.row_major(total_twfs, 2)
    alias twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    var twfs = ctx.enqueue_create_buffer[out_dtype](twf_layout.size())
    ctx.enqueue_copy(twfs, twfs_array.unsafe_ptr())
    var twiddle_factors = LayoutTensor[mut=False, out_dtype, twf_layout](
        twfs.unsafe_ptr()
    )

    alias grid_dim = (num_blocks, batches)
    alias gpu_info = ctx.default_device_info
    alias is_sm_90_or_newer = (
        gpu_info.vendor == Vendor.NVIDIA_GPU and gpu_info.compute >= 9.0
    )

    @parameter
    if is_sm_90_or_newer and num_blocks * batches <= max_cluster_size:
        # TODO: this should use distributed shared memory for twfs
        ctx.enqueue_function[
            _inter_block_fft_kernel_radix_n[
                in_dtype,
                out_dtype,
                in_layout,
                out_layout,
                twiddle_factors.layout,
                twiddle_factors.origin,
                twiddle_factors.address_space,
                length=length,
                ordered_bases=ordered_bases,
                processed_list=processed_list,
                do_rfft=do_rfft,
                inverse=inverse,
                twf_offsets=twf_offsets,
            ]
        ](output, x, twiddle_factors, grid_dim=grid_dim, block_dim=block_dim)
    else:
        alias func[b: Int] = _inter_multiprocessor_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            twiddle_factors.layout,
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
            ctx.enqueue_function[func[b]](
                output,
                x,
                twiddle_factors,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )


fn _inter_multiprocessor_fft_kernel_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    twf_layout: Layout,
    twf_origin: ImmutableOrigin,
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
    batch_output: LayoutTensor[mut=True, out_dtype, out_layout],
    batch_x: LayoutTensor[mut=False, in_dtype, in_layout],
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
):
    """A kernel that assumes `sequence_length // smallest_base <=
    max_threads_available`."""
    alias amnt_threads = length // base
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y
    alias x_layout = Layout.row_major(length, in_layout.shape[2].value())
    var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * block_num
    )
    alias block_out_layout = Layout.row_major(length, 2)
    var output = LayoutTensor[
        mut=True, out_dtype, block_out_layout, batch_output.origin
    ](batch_output.ptr + batch_output.stride[0]() * block_num)
    var x_out = tb[out_dtype]().row_major[base, 2]().alloc()

    alias last_base = ordered_bases[len(ordered_bases) - 1]
    alias total_threads = length // last_base
    alias func = _radix_n_fft_kernel[
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
    twf_origin: ImmutableOrigin,
    twf_address_space: AddressSpace,
    *,
    length: UInt,
    ordered_bases: List[UInt],
    processed_list: List[UInt],
    do_rfft: Bool,
    inverse: Bool,
    twf_offsets: List[UInt],
](
    batch_output: LayoutTensor[mut=True, out_dtype, out_layout],
    batch_x: LayoutTensor[mut=False, in_dtype, in_layout],
    twiddle_factors: LayoutTensor[
        out_dtype, twf_layout, twf_origin, address_space=twf_address_space
    ],
):
    """A kernel that assumes `sequence_length // smallest_base <=
    max_threads_available`."""
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var block_num = block_dim.y * block_idx.y
    alias x_layout = Layout.row_major(length, in_layout.shape[2].value())
    var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * block_num
    )
    alias block_out_layout = Layout.row_major(length, 2)
    # TODO: this should use distributed shared memory for the intermediate output
    var output = LayoutTensor[
        mut=True, out_dtype, block_out_layout, batch_output.origin
    ](batch_output.ptr + batch_output.stride[0]() * block_num)
    alias last_base = ordered_bases[len(ordered_bases) - 1]
    alias total_threads = length // last_base
    var x_out = tb[out_dtype]().row_major[ordered_bases[0], 2]().alloc()

    @parameter
    for b in range(len(ordered_bases)):
        alias base = ordered_bases[b]
        alias processed = processed_list[b]
        alias amnt_threads = length // base
        alias func = _radix_n_fft_kernel[
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
    batch_output: LayoutTensor[mut=True, out_dtype, out_layout],
    batch_x: LayoutTensor[mut=False, in_dtype, in_layout],
):
    """An FFT that assumes `sequence_length // smallest_base <=
    max_threads_per_block` and that `sequence_length` out_dtype items fit in
    a block's shared memory."""

    var local_i = thread_idx.x
    var block_num = block_dim.y * block_idx.y
    alias x_layout = Layout.row_major(length, in_layout.shape[2].value())
    var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
        batch_x.ptr + batch_x.stride[0]() * block_num
    )
    alias block_out_layout = Layout.row_major(length, 2)
    var output = LayoutTensor[
        mut=True, out_dtype, block_out_layout, batch_output.origin
    ](batch_output.ptr + batch_output.stride[0]() * block_num)
    var shared_f = tb[out_dtype]().row_major[length, 2]().shared().alloc()
    alias twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    alias twfs_layout = Layout.row_major(total_twfs, 2)
    var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
        twfs_array.unsafe_ptr()
    )
    alias last_base = ordered_bases[len(ordered_bases) - 1]
    alias total_threads = length // last_base
    var x_out = tb[out_dtype]().row_major[ordered_bases[0], 2]().alloc()

    @parameter
    for b in range(len(ordered_bases)):
        alias base = ordered_bases[b]
        alias processed = processed_list[b]
        alias amnt_threads = length // base
        alias func = _radix_n_fft_kernel[
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
        alias offset = i * total_threads
        var res = shared_f.load[width=2](local_i + offset, 0)
        output.store(local_i + offset, 0, res)

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

    alias twfs_array = _get_flat_twfs[
        out_dtype, length, total_twfs, ordered_bases, processed_list, inverse
    ]()
    alias twfs_layout = Layout.row_major(total_twfs, 2)
    var twfs = LayoutTensor[mut=False, out_dtype, twfs_layout](
        twfs_array.unsafe_ptr()
    )

    @parameter
    for b in range(len(ordered_bases)):
        alias base = ordered_bases[b]
        alias processed = processed_list[b]
        alias batches = in_layout.shape[0].value()
        alias amnt_threads_per_block = length // base
        alias amnt_threads = batches * amnt_threads_per_block

        @parameter
        fn _inner_kernel(global_i: Int):
            var block_num = global_i // amnt_threads_per_block
            var local_i = global_i % amnt_threads_per_block
            alias block_out_layout = Layout.row_major(length, 2)
            var output = LayoutTensor[
                mut=True, out_dtype, block_out_layout, batch_output.origin
            ](batch_output.ptr + batch_output.stride[0]() * block_num)
            alias x_layout = Layout.row_major(
                length, in_layout.shape[2].value()
            )
            var x = LayoutTensor[mut=False, in_dtype, x_layout, batch_x.origin](
                batch_x.ptr + batch_x.stride[0]() * block_num
            )
            var x_out_array = InlineArray[Scalar[out_dtype], base * 2](
                uninitialized=True
            )
            var x_out = LayoutTensor[
                mut=True, out_dtype, Layout.row_major(base, 2)
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
            amnt_threads, cpu_workers.or_else(parallelism_level())
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
    out_origin: MutableOrigin,
    out_address_space: AddressSpace,
    twf_layout: Layout,
    twf_origin: ImmutableOrigin,
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
    alias Sc = Scalar[_get_dtype[length]()]
    alias offset = Sc(processed)
    var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (offset * Sc(base))

    alias Co = ComplexScalar[out_dtype]
    alias CoV = SIMD[out_dtype, 2]
    alias is_even = length % 2 == 0

    @always_inline
    fn to_Co(v: CoV) -> Co:
        return UnsafePointer(to=v).bitcast[Co]()[]

    @always_inline
    fn to_CoV(c: Co) -> CoV:
        return UnsafePointer(to=c).bitcast[CoV]()[]

    @parameter
    fn _scatter_offsets(out res: SIMD[Sc.dtype, base * 2]):
        res = {}
        var idx = 0
        for i in range(base):
            res[idx] = i * offset * 2
            idx += 1
            res[idx] = i * offset * 2 + 1
            idx += 1

    @parameter
    fn _base_phasor[i: UInt, j: UInt](out res: Co):
        alias base_twf = _get_twiddle_factors[base, out_dtype, inverse]()
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
            alias step = Sc(i * offset)
            return output.load[2](Int(n + step), 0)

    var x_0 = _get_x[0]()

    @parameter
    for j in range(UInt(1), base):
        var x_j = _get_x[j]()

        @parameter
        if processed == 1:

            @parameter
            for i in range(base):
                alias base_phasor = _base_phasor[i, j]()
                var acc: CoV

                @parameter
                if j == 1:
                    acc = x_0
                else:
                    acc = x_out.load[2](i, 0)
                x_out.store(i, 0, _twf_fma[base_phasor, j == 1](x_j, acc))
            continue

        var i0_j_twf_vec = twiddle_factors.load[2](
            twf_offset + local_i * (base - 1) + (j - 1), 0
        )
        var i0_j_twf = to_Co(i0_j_twf_vec)

        @parameter
        for i in range(base):
            alias base_phasor = _base_phasor[i, j]()
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
                acc = x_out.load[2](i, 0)

            x_out.store(i, 0, to_CoV(twf.fma(to_Co(x_j), to_Co(acc))))

    @parameter
    if inverse and processed * base == length:  # last ifft stage
        alias factor = (Float64(1) / Float64(length)).cast[out_dtype]()

        @parameter
        if base.is_power_of_two():
            x_out.ptr.store(x_out.load[base * 2](0, 0) * factor)
        else:

            @parameter
            for i in range(base):
                x_out.store(i, 0, x_out.load[2](i, 0) * factor)

    @parameter
    if base.is_power_of_two() and processed == 1:
        output.store(Int(n), 0, x_out.load[base * 2](0, 0))
    elif base.is_power_of_two() and out_address_space is AddressSpace.GENERIC:
        alias offsets = _scatter_offsets()
        var v = x_out.load[base * 2](0, 0)
        output.ptr.offset(n * 2).scatter(offsets, v)
    else:

        @parameter
        for i in range(base):
            alias step = Sc(i * offset)
            output.store(Int(n + step), 0, x_out.load[2](i, 0))
