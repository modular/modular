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
"""NaN/Inf detection kernels for the max-debug.nan-check feature.

These kernels are registered as custom ops in MOGGKernelAPI and inserted
by the NanCheckPass compiler pass. The architecture is:

1. nan_check_count: read-only reduction that counts NaN/Inf values in a
   floating-point tensor. Outputs two single-element int32 tensors with
   the counts. CPU: vectorized scan with atomics. GPU: parallel reduction.

2. nan_check_raise: host-side kernel that reads the NaN/Inf counts
   (transferred to host via mo.transfer for GPU graphs) and raises a
   diagnostic error if any are non-zero.
"""

from std.algorithm import elementwise
from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.host.info import is_cpu
from std.memory import alloc, stack_allocation
from std.atomic import Atomic
from std.sys import simd_width_of
from std.utils.numerics import isinf, isnan

from std.math import ceildiv
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList
from tensor import InputTensor, OutputTensor


@__name(t"nan_check_gpu_{dtype}", mangle=True)
def _nan_check_gpu_kernel[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    total_elements: Int,
    out_nan: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    out_inf: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
):
    """GPU kernel: count NaN/Inf values via parallel reduction."""
    var nan_local = stack_allocation[
        1, Int32, address_space=AddressSpace.SHARED
    ]()
    var inf_local = stack_allocation[
        1, Int32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        nan_local[0] = Int32(0)
        inf_local[0] = Int32(0)
    barrier()

    var tid = block_idx.x * block_dim.x + thread_idx.x
    var my_nan = Int32(0)
    var my_inf = Int32(0)
    if tid < total_elements:
        var val = src_ptr.load(tid)
        if isnan(val):
            my_nan = Int32(1)
        elif isinf(val):
            my_inf = Int32(1)
    if my_nan > 0:
        _ = Atomic.fetch_add(nan_local, my_nan)
    if my_inf > 0:
        _ = Atomic.fetch_add(inf_local, my_inf)

    barrier()

    if thread_idx.x == 0:
        if nan_local[0] > 0:
            _ = Atomic.fetch_add(out_nan, nan_local[0])
        if inf_local[0] > 0:
            _ = Atomic.fetch_add(out_inf, inf_local[0])


def nan_check_count[
    dtype: DType,
    rank: Int,
    target: StaticString,
](
    nan_count_out: OutputTensor[dtype=DType.int32, rank=1, ...],
    inf_count_out: OutputTensor[dtype=DType.int32, rank=1, ...],
    input: InputTensor[dtype=dtype, rank=rank, ...],
    ctx: DeviceContextPtr,
) raises:
    """Counts NaN/Inf values in a floating-point tensor.

    Read-only: does not modify the input tensor. Outputs two single-element
    int32 tensors with the NaN and Inf counts. No D2H transfer or
    synchronization is performed — the counts stay on the same device as
    the input.
    """
    var total = input.size()

    comptime if is_cpu[target]():
        # CPU path: vectorized scan using elementwise with atomic accumulators.
        var nan_acc = alloc[Scalar[DType.int32]](1)
        var inf_acc = alloc[Scalar[DType.int32]](1)
        nan_acc[] = Int32(0)
        inf_acc[] = Int32(0)

        @parameter
        @always_inline
        def scan[
            width: Int, _rank: Int, alignment: Int = 1
        ](idx: IndexList[_rank]) capturing:
            var flat = idx[0]
            var ptr = input.unsafe_ptr()
            var val = ptr.load[width=width](flat)
            var nans = isnan(val).cast[DType.int32]().reduce_add()
            var infs = isinf(val).cast[DType.int32]().reduce_add()
            if nans > 0:
                _ = Atomic.fetch_add(nan_acc, nans)
            if infs > 0:
                _ = Atomic.fetch_add(inf_acc, infs)

        elementwise[scan, simd_width_of[dtype]()](
            total, ctx.get_optional_device_context()
        )

        nan_count_out.unsafe_ptr()[] = nan_acc[]
        inf_count_out.unsafe_ptr()[] = inf_acc[]
        nan_acc.free()
        inf_acc.free()
    else:
        # GPU path: parallel reduction writing directly to output tensors.
        # Zero the output counts first via a single-thread init kernel,
        # then run the reduction that atomically accumulates into them.
        var gpu_ctx = ctx.get_device_context()
        var out_nan_ptr = rebind[
            UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
        ](nan_count_out.unsafe_ptr())
        var out_inf_ptr = rebind[
            UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
        ](inf_count_out.unsafe_ptr())

        @parameter
        @__name(t"nan_check_zero_counts", mangle=True)
        def zero_counts(
            nan_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
            inf_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
        ):
            nan_ptr[] = Int32(0)
            inf_ptr[] = Int32(0)

        gpu_ctx.enqueue_function[zero_counts, zero_counts](
            out_nan_ptr, out_inf_ptr, grid_dim=1, block_dim=1
        )

        comptime BLOCK = 256
        var grid = ceildiv(total, BLOCK)

        comptime kernel = _nan_check_gpu_kernel[dtype]
        gpu_ctx.enqueue_function[kernel, kernel](
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                input.unsafe_ptr()
            ),
            total,
            out_nan_ptr,
            out_inf_ptr,
            grid_dim=grid,
            block_dim=BLOCK,
        )


def nan_check_raise[
    label: StaticString,
    type_str: StaticString,
](
    nan_count: InputTensor[dtype=DType.int32, rank=1, ...],
    inf_count: InputTensor[dtype=DType.int32, rank=1, ...],
) raises:
    """Raises an error if NaN or Inf counts are non-zero.

    Reads two single-element int32 tensors on the host and raises a
    diagnostic error if either is > 0.
    """
    var nans = Int(nan_count.unsafe_ptr()[])
    var infs = Int(inf_count.unsafe_ptr()[])
    if nans > 0 or infs > 0:
        raise Error(
            "NaN/Inf detected in '"
            + String(label)
            + "' ("
            + String(type_str)
            + "): "
            + String(nans)
            + " NaN, "
            + String(infs)
            + " Inf"
        )
