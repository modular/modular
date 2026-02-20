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
"""P2P link bandwidth benchmark between 2 GPUs.

Measures raw NVLink/xGMI bandwidth between a pair of GPUs using a simple
grid-strided copy kernel at configurable SIMD widths.

Modes:
    0: unidir push - GPU 0 reads local, writes to GPU 1's buffer
    1: unidir pull - GPU 0 reads from GPU 1's buffer, writes local
    2: bidir push  - both GPUs write to each other simultaneously
    3: bidir pull  - both GPUs read from each other simultaneously

Usage:
    mojo -D direction=0 bench_p2p_copy.mojo                    # unidir push
    mojo -D direction=1 bench_p2p_copy.mojo                    # unidir pull
    mojo -D direction=2 bench_p2p_copy.mojo                    # bidir push
    mojo -D direction=3 bench_p2p_copy.mojo                    # bidir pull
    mojo -D store_width=8 -D direction=0 bench_p2p_copy.mojo   # custom width
"""

from math import ceildiv
from sys import env_get_int, env_get_dtype, is_amd_gpu, size_of, simd_width_of

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from comm.sync import can_enable_p2p
from gpu import global_idx, grid_dim, MAX_THREADS_PER_BLOCK_METADATA
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import arg_parse, human_readable_size
from utils import StaticTuple

comptime BLOCK_SIZE = 256
comptime store_width = env_get_int["store_width", 0]()
# direction: 0 = unidir push, 1 = unidir pull, 2 = bidir push, 3 = bidir pull
comptime direction = env_get_int["direction", 0]()
comptime _target_address_space = (
    AddressSpace.GLOBAL if is_amd_gpu() else AddressSpace.GENERIC
)


fn _mode_name[direction: Int]() -> String:
    comptime if direction == 0:
        return "unidir-push"
    elif direction == 1:
        return "unidir-pull"
    elif direction == 2:
        return "bidir-push"
    else:
        return "bidir-pull"


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
fn p2p_copy_kernel[
    dtype: DType,
    width: Int,
](
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int,
):
    var global_tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK_SIZE
    var num_vectors = num_elements // width

    comptime vec_align = width * size_of[dtype]()

    for idx in range(global_tid, num_vectors, stride):
        var elem_idx = idx * width
        var data = src.address_space_cast[_target_address_space]().load[
            width=width, alignment=vec_align
        ](elem_idx)
        dst.address_space_cast[_target_address_space]().store[
            width=width, alignment=vec_align
        ](elem_idx, data)


fn bench_p2p[
    dtype: DType,
    simd_width: Int,
    direction: Int,
](mut b: Bench, num_bytes: Int) raises:
    comptime copy_kernel = p2p_copy_kernel[dtype, simd_width]
    comptime is_push = (direction == 0 or direction == 2)
    comptime is_bidir = (direction == 2 or direction == 3)

    var num_elements = num_bytes // size_of[dtype]()
    debug_assert(
        num_bytes % (size_of[dtype]() * simd_width) == 0,
        "Ragged sizes unsupported by bench_p2p.",
    )
    var grid_size = ceildiv(num_elements // simd_width, BLOCK_SIZE)

    # Create contexts and buffers on both GPUs.
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)

    # For bidir, each GPU needs separate read and write buffers to avoid
    # concurrent read/write conflicts during verification.
    var buf0_write = ctx0.enqueue_create_buffer[dtype](num_elements)
    var buf1_write = ctx1.enqueue_create_buffer[dtype](num_elements)

    # Bidir uses separate read buffers; unidir reads from write buffers.
    var buf0_read = ctx0.enqueue_create_buffer[dtype](num_elements)
    var buf1_read = ctx1.enqueue_create_buffer[dtype](num_elements)

    comptime if is_bidir:
        ctx0.enqueue_memset(buf0_read, Scalar[dtype](10))
        ctx1.enqueue_memset(buf1_read, Scalar[dtype](20))

    ctx0.enqueue_memset(buf0_write, Scalar[dtype](1))
    ctx1.enqueue_memset(buf1_write, Scalar[dtype](2))
    ctx0.synchronize()
    ctx1.synchronize()

    # bench_multicontext requires >= 2 contexts, so always pass both.
    # For unidir only GPU 0 does work; GPU 1 is idle (see bench_iter).
    var ctxs = List[DeviceContext]()
    ctxs.append(DeviceContext(device_id=0))
    ctxs.append(DeviceContext(device_id=1))

    var name = String(
        "p2p-",
        _mode_name[direction](),
        "-",
        dtype,
        "-w",
        simd_width,
        "-",
        human_readable_size(num_bytes),
    )

    @parameter
    @always_inline
    fn bench_iter(
        mut bencher: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            # In unidir mode only GPU 0 does work; GPU 1 is idle.
            comptime if not is_bidir:
                if ctx_idx != 0:
                    return

            # Determine src/dst based on direction and which GPU we are.
            # Push: each GPU reads local, writes remote.
            # Pull: each GPU reads remote, writes local.
            var dst: UnsafePointer[Scalar[dtype], MutAnyOrigin]
            var src: UnsafePointer[Scalar[dtype], MutAnyOrigin]

            comptime if is_bidir:
                comptime if is_push:
                    if ctx_idx == 0:
                        dst = buf1_write.unsafe_ptr()
                        src = buf0_read.unsafe_ptr()
                    else:
                        dst = buf0_write.unsafe_ptr()
                        src = buf1_read.unsafe_ptr()
                else:
                    if ctx_idx == 0:
                        dst = buf0_write.unsafe_ptr()
                        src = buf1_read.unsafe_ptr()
                    else:
                        dst = buf1_write.unsafe_ptr()
                        src = buf0_read.unsafe_ptr()
            else:
                comptime if is_push:
                    dst = buf1_write.unsafe_ptr()
                    src = buf0_write.unsafe_ptr()
                else:
                    dst = buf0_write.unsafe_ptr()
                    src = buf1_write.unsafe_ptr()

            ctx_inner.enqueue_function[copy_kernel, copy_kernel](
                dst,
                src,
                num_elements,
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
            )

        bencher.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_iter](
        ctxs,
        BenchId(name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    b.dump_report()

    # --- Verification ---
    # Run one more copy and check that the data arrived correctly.
    # Reset destination buffers to 0 so we can verify fresh writes.
    comptime if is_bidir:
        ctx0.enqueue_memset(buf0_write, Scalar[dtype](0))
        ctx1.enqueue_memset(buf1_write, Scalar[dtype](0))
        ctx0.synchronize()
        ctx1.synchronize()

        comptime if is_push:
            # GPU 0: buf0_read(10) -> buf1_write, GPU 1: buf1_read(20) -> buf0_write
            ctx0.enqueue_function[copy_kernel, copy_kernel](
                buf1_write.unsafe_ptr(),
                buf0_read.unsafe_ptr(),
                num_elements,
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
            )
            ctx1.enqueue_function[copy_kernel, copy_kernel](
                buf0_write.unsafe_ptr(),
                buf1_read.unsafe_ptr(),
                num_elements,
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
            )
        else:
            # GPU 0: buf1_read(20) -> buf0_write, GPU 1: buf0_read(10) -> buf1_write
            ctx0.enqueue_function[copy_kernel, copy_kernel](
                buf0_write.unsafe_ptr(),
                buf1_read.unsafe_ptr(),
                num_elements,
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
            )
            ctx1.enqueue_function[copy_kernel, copy_kernel](
                buf1_write.unsafe_ptr(),
                buf0_read.unsafe_ptr(),
                num_elements,
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
            )

        ctx0.synchronize()
        ctx1.synchronize()

        # Verify: copy back to host and check
        var host0 = alloc[Scalar[dtype]](num_elements)
        var host1 = alloc[Scalar[dtype]](num_elements)
        ctx0.enqueue_copy(host0, buf0_write)
        ctx1.enqueue_copy(host1, buf1_write)
        ctx0.synchronize()
        ctx1.synchronize()

        # buf0_write should have buf1_read's value (20)
        # buf1_write should have buf0_read's value (10)
        _verify(host0, Scalar[dtype](20), num_elements, 0)
        _verify(host1, Scalar[dtype](10), num_elements, 1)

        host0.free()
        host1.free()
    else:
        # Unidir: reset dst, run one copy, verify.
        comptime if is_push:
            # src=buf0_write(1) -> dst=buf1_write
            ctx1.enqueue_memset(buf1_write, Scalar[dtype](0))
            ctx1.synchronize()
            ctx0.enqueue_function[copy_kernel, copy_kernel](
                buf1_write.unsafe_ptr(),
                buf0_write.unsafe_ptr(),
                num_elements,
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
            )
            ctx0.synchronize()
            var host = alloc[Scalar[dtype]](num_elements)
            ctx1.enqueue_copy(host, buf1_write)
            ctx1.synchronize()
            _verify(host, Scalar[dtype](1), num_elements, 1)
            host.free()
        else:
            # src=buf1_write(2) -> dst=buf0_write
            ctx0.enqueue_memset(buf0_write, Scalar[dtype](0))
            ctx0.synchronize()
            ctx0.enqueue_function[copy_kernel, copy_kernel](
                buf0_write.unsafe_ptr(),
                buf1_write.unsafe_ptr(),
                num_elements,
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
            )
            ctx0.synchronize()
            var host = alloc[Scalar[dtype]](num_elements)
            ctx0.enqueue_copy(host, buf0_write)
            ctx0.synchronize()
            _verify(host, Scalar[dtype](2), num_elements, 0)
            host.free()

    _ = buf0_write^
    _ = buf1_write^
    _ = buf0_read^
    _ = buf1_read^
    _ = ctx0^
    _ = ctx1^
    _ = ctxs^


fn _verify[
    dtype: DType
](
    host: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    expected: Scalar[dtype],
    num_elements: Int,
    gpu: Int,
) raises:
    for i in range(num_elements):
        if host[i] != expected:
            raise Error(
                String(
                    "Verification failed at GPU ",
                    gpu,
                    " index ",
                    i,
                    ": got ",
                    host[i],
                    " expected ",
                    expected,
                )
            )
    print("Verification passed for GPU", gpu)


fn main() raises:
    var num_bytes = arg_parse("num_bytes", 64 * 1024 * 1024)
    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()
    comptime simd_width = (
        simd_width_of[dtype, target = get_gpu_target()]() if store_width
        == 0 else store_width
    )

    if DeviceContext.number_of_devices() < 2:
        raise Error("At least 2 GPUs required")

    if not can_enable_p2p():
        raise Error("P2P access not available between GPUs")

    var num_elements = num_bytes // size_of[dtype]()
    if num_elements % simd_width != 0:
        raise Error("num_elements must be a multiple of store_width")

    var b = Bench()
    bench_p2p[dtype, simd_width, direction](b, num_bytes)
