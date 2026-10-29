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
"""PDL correctness for the SM100 blockwise-scaled FP8 matmul (B200).

The matmul reads A and the A-scales from a predecessor grid (the activation
quantizer) and its output is read by a successor. This test builds both ends
adversarially:

- `late_producer` releases its dependents immediately, then sleeps before it
  copies A and the A-scales into the buffers the matmul consumes (which start
  zeroed). A matmul that reads A or the A-scales before its PDL wait sees zeros
  and produces a different result.
- `consumer` waits on the matmul and copies its output. A matmul that releases
  its dependents before its stores are visible is caught the same way.

The PDL-on chain must be bit-identical to the PDL-off matmul on the same
inputs, since PDL reorders loads and never arithmetic. M covers the three
tile-selection regimes of the dispatcher (<= 128, <= 512, larger) and values
that are not tile multiples; N = 576 leaves a partial N tile and K = 2048 /
7168 exercise short and long main loops.
"""

from std.math import ceildiv
from std.random import rand
from std.time import sleep

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from layout import Idx, TileTensor, row_major
from linalg.fp8_quantization import matmul_dynamic_scaled_fp8

comptime fp8_type = DType.float8_e4m3fn
comptime out_type = DType.bfloat16
comptime scales_type = DType.float32
comptime BLOCK = 128


def late_producer(
    src_a: MutPointer[Scalar[fp8_type], MutAnyOrigin],
    dst_a: MutPointer[Scalar[fp8_type], MutAnyOrigin],
    len_a: Int32,
    src_s: MutPointer[Scalar[scales_type], MutAnyOrigin],
    dst_s: MutPointer[Scalar[scales_type], MutAnyOrigin],
    len_s: Int32,
):
    """Releases dependents at once, then publishes A and the A-scales late."""
    launch_dependent_grids()
    sleep(0.00005)

    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(len_a), stride):
        dst_a[i] = src_a[i]
    for i in range(tid, Int(len_s), stride):
        dst_s[i] = src_s[i]


def consumer(
    src: MutPointer[Scalar[out_type], MutAnyOrigin],
    dst: MutPointer[Scalar[out_type], MutAnyOrigin],
    length: Int32,
):
    """Copies the matmul output after waiting on it."""
    wait_on_dependent_grids()

    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(length), stride):
        dst[i] = src[i]


def run_case[N: Int, K: Int](ctx: DeviceContext, m: Int, num_iters: Int) raises:
    comptime k_blocks = K // BLOCK
    comptime n_blocks = ceildiv(N, BLOCK)

    print("blockwise fp8 PDL: M=", m, " N=", N, " K=", K, sep="")

    var a_elems = m * K
    var c_elems = m * N
    var as_elems = k_blocks * m
    var bs_elems = n_blocks * k_blocks
    var b_elems = N * K

    var a_f32 = List[Float32](unsafe_uninit_length=a_elems)
    rand(a_f32.unsafe_ptr(), a_elems)
    var a_host = List[Scalar[fp8_type]](unsafe_uninit_length=a_elems)
    for i in range(a_elems):
        a_host[i] = (a_f32[i] * 2.0 - 1.0).cast[fp8_type]()
    var b_f32 = List[Float32](unsafe_uninit_length=b_elems)
    rand(b_f32.unsafe_ptr(), b_elems)
    var b_host = List[Scalar[fp8_type]](unsafe_uninit_length=b_elems)
    for i in range(b_elems):
        b_host[i] = (b_f32[i] * 2.0 - 1.0).cast[fp8_type]()
    var as_host = List[Scalar[scales_type]](unsafe_uninit_length=as_elems)
    rand(as_host.unsafe_ptr(), as_elems)
    var bs_host = List[Scalar[scales_type]](unsafe_uninit_length=bs_elems)
    rand(bs_host.unsafe_ptr(), bs_elems)

    var a_src = ctx.enqueue_create_buffer[fp8_type](a_elems)
    var a_dst = ctx.enqueue_create_buffer[fp8_type](a_elems)
    var b_dev = ctx.enqueue_create_buffer[fp8_type](b_elems)
    var as_src = ctx.enqueue_create_buffer[scales_type](as_elems)
    var as_dst = ctx.enqueue_create_buffer[scales_type](as_elems)
    var bs_dev = ctx.enqueue_create_buffer[scales_type](bs_elems)
    var c_ref_dev = ctx.enqueue_create_buffer[out_type](c_elems)
    var c_dev = ctx.enqueue_create_buffer[out_type](c_elems)
    var c_out_dev = ctx.enqueue_create_buffer[out_type](c_elems)
    ctx.enqueue_copy(a_src, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(as_src, as_host)
    ctx.enqueue_copy(bs_dev, bs_host)

    var a_src_t = TileTensor(a_src, row_major(m, Idx[K]))
    var a_dst_t = TileTensor(a_dst, row_major(m, Idx[K]))
    var b_t = TileTensor(b_dev, row_major(Idx[N], Idx[K]))
    var as_src_t = TileTensor(as_src, row_major(Idx[k_blocks], m))
    var as_dst_t = TileTensor(as_dst, row_major(Idx[k_blocks], m))
    var bs_t = TileTensor(bs_dev, row_major(Idx[n_blocks], Idx[k_blocks]))
    var c_ref_t = TileTensor(c_ref_dev, row_major(m, Idx[N]))
    var c_t = TileTensor(c_dev, row_major(m, Idx[N]))

    # Reference: the same kernel with PDL off, straight from the sources.
    matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=1,
        n_scale_granularity=BLOCK,
        k_scale_granularity=BLOCK,
        transpose_b=True,
        target="gpu",
    ](c_ref_t, a_src_t, b_t, as_src_t, bs_t, ctx)
    ctx.synchronize()
    var want = List[Scalar[out_type]](unsafe_uninit_length=c_elems)
    ctx.enqueue_copy(want, c_ref_dev)
    ctx.synchronize()

    var got = List[Scalar[out_type]](unsafe_uninit_length=c_elems)
    comptime pdl_level = PDLLevel.ON
    for it in range(num_iters):
        ctx.enqueue_memset[fp8_type](a_dst, 0)
        ctx.enqueue_memset[scales_type](as_dst, 0)
        ctx.enqueue_memset[out_type](c_dev, 0)
        ctx.enqueue_memset[out_type](c_out_dev, 0)

        ctx.enqueue_function[late_producer](
            a_src,
            a_dst,
            Int32(a_elems),
            as_src,
            as_dst,
            Int32(as_elems),
            grid_dim=2,
            block_dim=128,
        )
        matmul_dynamic_scaled_fp8[
            input_scale_granularity="block",
            weight_scale_granularity="block",
            m_scale_granularity=1,
            n_scale_granularity=BLOCK,
            k_scale_granularity=BLOCK,
            transpose_b=True,
            target="gpu",
            pdl_level=pdl_level,
        ](c_t, a_dst_t, b_t, as_dst_t, bs_t, ctx)
        ctx.enqueue_function[consumer](
            c_dev,
            c_out_dev,
            Int32(c_elems),
            grid_dim=ceildiv(c_elems, 256),
            block_dim=256,
            attributes=pdl_launch_attributes(pdl_level),
        )
        ctx.synchronize()
        ctx.enqueue_copy(got, c_out_dev)
        ctx.synchronize()

        var mismatch = 0
        for j in range(c_elems):
            if got[j] != want[j]:
                mismatch += 1
        if mismatch != 0:
            raise Error(
                String(
                    "PDL race: iter ",
                    it,
                    " has ",
                    mismatch,
                    " of ",
                    c_elems,
                    " outputs differing from the PDL-off matmul",
                )
            )

    _ = a_f32^
    _ = a_host^
    _ = b_f32^
    _ = b_host^
    _ = as_host^
    _ = bs_host^
    _ = want^
    _ = got^
    _ = a_src^
    _ = a_dst^
    _ = b_dev^
    _ = as_src^
    _ = as_dst^
    _ = bs_dev^
    _ = c_ref_dev^
    _ = c_dev^
    _ = c_out_dev^


def run_k[K: Int](ctx: DeviceContext) raises:
    # The dispatcher picks a different tile and CTA group at M <= 128,
    # M <= 512 and above; the ragged values leave partial M tiles.
    run_case[576, K](ctx, 4, 20)
    run_case[576, K](ctx, 36, 20)
    run_case[576, K](ctx, 132, 20)
    run_case[576, K](ctx, 516, 20)
    run_case[1536, K](ctx, 36, 20)
    run_case[1536, K](ctx, 516, 20)
    run_case[1536, K](ctx, 2052, 5)


def main() raises:
    with DeviceContext() as ctx:
        run_k[2048](ctx)
        run_k[7168](ctx)
        print("PASSED")
