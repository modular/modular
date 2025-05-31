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

from math import align_up
from sys import env_get_bool, env_get_dtype, env_get_int, sizeof

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from buffer.dimlist import _make_tuple
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host.info import DEFAULT_GPU_ARCH
from internal_utils import DeviceNDBuffer, HostNDBuffer, arg_parse
import random
from internal_utils._utils import (
    InitializationType,
    ValOrDim,
    dynamic,
    initialize,
    static,
)
from utils import IndexList, StaticTuple
from linalg.matmul_gpu import _matmul_gpu
from memory import UnsafePointer
from gpu.comm.allreduce import (
    MAX_GPUS,
    Signal,
    elementwise_epilogue_type,
)

from linalg.distributed_matmul import matmul_allreduce

from utils import IndexList


fn _get_run_name[
    type: DType,
    ngpus: Int,
    partition_dim: Int,
    num_partitions: Int,
    overlap_with_dpl: Bool,
](m: ValOrDim, n: ValOrDim, k: ValOrDim,) -> String:
    var vendor_str = "matmul_allreduce"
    var type_str = String("(", type, ") : ")
    var m_str = String(m.value, "_dynamic") if m.dim.is_dynamic() else String(
        m.dim
    )
    var n_str = String(n.value, "_dynamic") if n.dim.is_dynamic() else String(
        n.dim
    )
    var k_str = String(k.value, "_dynamic") if k.dim.is_dynamic() else String(
        k.dim
    )

    var ngpus_str = String("/ngpus=", ngpus)
    var num_partitions_str = String(
        "/num_partitions=", num_partitions
    ) if num_partitions > 1 else String()
    var partition_dim_str = String(
        "/partition_dim=", partition_dim
    ) if num_partitions > 1 else String()
    var overlap_str = String("/overlap") if overlap_with_dpl else String()
    return String(
        vendor_str,
        type_str,
        m_str,
        " x ",
        n_str,
        " x ",
        k_str,
        ngpus_str,
        num_partitions_str,
        partition_dim_str,
        overlap_str,
    )


fn bench_matmul_all_reduce[
    type: DType,
    ngpus: Int,
    partition_dim: Int,
    num_partitions: Int,
    overlap_with_dpl: Bool,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
) raises:
    # Create matmul input and output buffers.
    var A_list = List[DeviceBuffer[type]](capacity=ngpus)
    var B_list = List[DeviceBuffer[type]](capacity=ngpus)
    var C_list = List[DeviceBuffer[type]](capacity=ngpus)
    var C_reduced_list = List[DeviceBuffer[type]](capacity=ngpus)
    var A_host_list = List[UnsafePointer[Scalar[type]]](capacity=ngpus)
    var B_host_list = List[UnsafePointer[Scalar[type]]](capacity=ngpus)
    var C_reduced_host_list = List[UnsafePointer[Scalar[type]]](capacity=ngpus)

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal], MAX_GPUS](
        UnsafePointer[Signal]()
    )

    var mn = m.value * n.value
    var mk = m.value * k.value
    var nk = n.value * k.value

    # Set up temp buffers for GPUs to reduce-scatter into / all-gather from.
    # Partitioned matmul has size M * N / num_partitions
    var temp_length = m.value * n.value // num_partitions
    var temp_buffer_num_bytes = ngpus * sizeof[type]() * temp_length

    # Initialize buffers for each GPU
    @parameter
    for i in range(ngpus):
        # Allocate A. B, C on device for matmul.
        A_list.append(list_of_ctx[i].enqueue_create_buffer[type](mk))
        B_list.append(list_of_ctx[i].enqueue_create_buffer[type](nk))
        C_list.append(list_of_ctx[i].enqueue_create_buffer[type](mn))
        C_reduced_list.append(list_of_ctx[i].enqueue_create_buffer[type](mn))

        # Allocate matmul inputs A B and final output C_reduced on host
        A_host_list.append(UnsafePointer[Scalar[type]].alloc(mk))
        B_host_list.append(UnsafePointer[Scalar[type]].alloc(nk))
        C_reduced_host_list.append(UnsafePointer[Scalar[type]].alloc(mn))

        # Initialize randomdly A and b
        random.randn(A_host_list[i], mk)
        random.randn(B_host_list[i], nk)

        # Copy A and B to device
        list_of_ctx[i].enqueue_copy(A_list[i], A_host_list[i])
        list_of_ctx[i].enqueue_copy(B_list[i], B_host_list[i])

        # Create and initialize signal buffers
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](
                sizeof[Signal]() + temp_buffer_num_bytes
            )
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = signal_buffers[i].unsafe_ptr().bitcast[Signal]()

    # Create the list of NDBuffers.
    alias A_static_shape = DimList(m.dim, k.dim)
    alias B_static_shape = DimList(n.dim, k.dim)
    alias C_static_shape = DimList(m.dim, n.dim)
    var As = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, A_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, A_static_shape]())
    var Bs = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, B_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, B_static_shape]())
    var Cs = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, C_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, C_static_shape]())
    var out_bufs = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, C_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, C_static_shape]())

    # Setup the kernel NDBuffers
    @parameter
    for i in range(ngpus):
        As[i] = NDBuffer[type, 2, MutableAnyOrigin, A_static_shape](
            A_list[i].unsafe_ptr(), DimList(m.value, k.value)
        )
        Bs[i] = NDBuffer[type, 2, MutableAnyOrigin, B_static_shape](
            B_list[i].unsafe_ptr(), DimList(n.value, k.value)
        )
        Cs[i] = NDBuffer[type, 2, MutableAnyOrigin, C_static_shape](
            C_list[i].unsafe_ptr(), DimList(m.value, n.value)
        )
        out_bufs[i] = NDBuffer[type, 2, MutableAnyOrigin, C_static_shape](
            C_reduced_list[i].unsafe_ptr(), DimList(m.value, n.value)
        )

    # Copy-capture in registers since the lambda will be used on GPU.
    var out_bufs_capture = StaticTuple[
        NDBuffer[type, 2, MutableAnyOrigin], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin]())

    @parameter
    for i in range(ngpus):
        out_bufs_capture[i] = NDBuffer[type, 2](
            C_reduced_list[i].unsafe_ptr(), DimList(m.value, n.value)
        )

    # Prepare the output lambda
    @always_inline
    @parameter
    @__copy_capture(out_bufs_capture)
    fn outputs_lambda[
        input_index: Int,
        _type: DType,
        _rank: Int,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: IndexList[_rank], val: SIMD[_type, _width]) -> None:
        out_bufs_capture[input_index].store[width=_width, alignment=_alignment](
            rebind[IndexList[2]](coords), rebind[SIMD[type, _width]](val)
        )

    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch() raises:
            matmul_allreduce[
                type=type,
                ngpus=ngpus,
                partition_dim=partition_dim,
                num_partitions=num_partitions,
                overlap_with_dpl=overlap_with_dpl,
                outputs_lambda=outputs_lambda,
            ](As, Bs, Cs, out_bufs, rank_sigs, list_of_ctx)

        b.iter_custom_multicontext[kernel_launch](list_of_ctx)

    b.bench_function[bench_func](
        BenchId(
            _get_run_name[
                type,
                ngpus,
                partition_dim,
                num_partitions,
                overlap_with_dpl,
            ](m, n, k)
        ),
        ThroughputMeasure(
            BenchMetric.flops,
            (2 * m.value * m.value * k.value + m.value * m.value) * ngpus,
        ),
    )

    _ = signal_buffers^
    _ = A_list^
    _ = B_list^
    _ = C_list^
    _ = C_reduced_list^


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()

    var M = Int(arg_parse("M", 8192))
    alias N = env_get_int["N", 8192]()
    alias K = env_get_int["K", 2048]()
    alias num_gpus = env_get_int["NUM_GPUS", 4]()
    alias partition_dim = env_get_int["DIM", 1]()
    alias num_partitions = env_get_int["PARTITIONS", 1]()
    alias overlap_with_dpl = env_get_bool["OVERLAP", False]()

    var m = Bench()

    if DeviceContext.number_of_devices() < num_gpus:
        raise Error("Not enough GPUs available")

    # Create GPU context.
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    bench_matmul_all_reduce[
        dtype,
        ngpus=num_gpus,
        partition_dim=partition_dim,
        num_partitions=num_partitions,
        overlap_with_dpl=overlap_with_dpl,
    ](
        m,
        ctx,
        dynamic(M),
        static[N](),
        static[K](),
    )

    m.dump_report()
