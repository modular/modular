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

"""Benchmarks `--ssd-layers` chunked SSD chain launches back to back.

A Mamba-2 mixer runs the chunked scan (`mamba2_ssd_chunked`) once per layer in
one stream: map, cumsum, chunk state, state passing and output, five grid
boundaries and five scratch tensors per layer. This driver launches that chain
`--ssd-layers` times with no host work in between, which is what 16 or 24 Mamba
layers do in a prefill step, and reports the wall time of the whole stack plus
the host time it took to enqueue it.

Required runtime options are --ssd-heads, --ssd-head-dim, --ssd-groups,
--sequence-lengths and --ssd-layers. Sequence lengths are a quoted
comma-separated vector, for example '[2048]'. Wall time comes from stream
events around the enqueue; host time is a CPU clock around the same enqueue
with the queue drained, so it is the part of a layer the GPU cannot hide. The
chain also runs twice from identically initialized pools on two independently
allocated sets of buffers and the two results must be bit-identical, so a
kernel that reads scratch it does not write fails the benchmark instead of
reporting a number.
"""

from std.benchmark import Batch, Bench, BenchmarkInfo, Report
from std.builtin.sort import sort
from std.random import rand, seed
from std.time import perf_counter_ns

from internal_utils import arg_parse
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.mamba2_ssd_chunked import mamba2_ssd_chunked_fwd_inplace_gpu

comptime WARMUP = 5
comptime SAMPLES = 20


def _median_ns(times: List[Int]) -> Float64:
    var ordered = times.copy()
    sort(Span(ordered))
    var middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (Float64(ordered[middle - 1]) + Float64(ordered[middle])) / 2
    return Float64(ordered[middle])


def _bf16_rand(n: Int, scale: Float32) -> List[Scalar[DType.bfloat16]]:
    var f = List[Float32](unsafe_uninit_length=n)
    rand(f.unsafe_ptr(), n)
    var out = List[Scalar[DType.bfloat16]](unsafe_uninit_length=n)
    for i in range(n):
        out[i] = ((f[i] - 0.5) * scale).cast[DType.bfloat16]()
    return out^


def _bench_shape(
    ctx: DeviceContext,
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    seq_lengths: List[Int],
    layers: Int,
) raises:
    comptime dtype = DType.bfloat16
    comptime DSTATE = 128
    var batch = len(seq_lengths)
    var total_len = 0
    for i in range(len(seq_lengths)):
        total_len += seq_lengths[i]
    var pool_n = batch * nheads * head_dim * DSTATE
    var y_n = total_len * nheads * head_dim
    var bc_n = total_len * ngroups * DSTATE
    var dt_n = total_len * nheads

    var a_h = _bf16_rand(nheads, 1.0)
    for i in range(nheads):
        a_h[i] = -abs(a_h[i]) - 0.1
    var x_h = _bf16_rand(total_len * nheads * head_dim, 2.0)
    var dt_h = _bf16_rand(dt_n, 2.0)
    var b_h = _bf16_rand(bc_n, 2.0)
    var c_h = _bf16_rand(bc_n, 2.0)
    var d_h = _bf16_rand(nheads, 1.0)
    var dtb_h = _bf16_rand(nheads, 1.0)
    var pool_h = List[Float32](unsafe_uninit_length=pool_n)
    rand(pool_h.unsafe_ptr(), pool_n)
    var qsl_h = List[Int32](unsafe_uninit_length=batch + 1)
    var slot_h = List[UInt32](unsafe_uninit_length=batch)
    var his_h = List[Scalar[DType.bool]](unsafe_uninit_length=batch)
    qsl_h[0] = 0
    var offset = 0
    for b in range(batch):
        offset += seq_lengths[b]
        qsl_h[b + 1] = Int32(offset)
        slot_h[b] = UInt32(b)
        his_h[b] = Scalar[DType.bool](True)

    var a_d = ctx.enqueue_create_buffer[dtype](nheads)
    var d_d = ctx.enqueue_create_buffer[dtype](nheads)
    var dtb_d = ctx.enqueue_create_buffer[dtype](nheads)
    var x_d = ctx.enqueue_create_buffer[dtype](len(x_h))
    var dt_d = ctx.enqueue_create_buffer[dtype](dt_n)
    var b_d = ctx.enqueue_create_buffer[dtype](bc_n)
    var c_d = ctx.enqueue_create_buffer[dtype](bc_n)
    var qsl_d = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var slot_d = ctx.enqueue_create_buffer[DType.uint32](batch)
    var his_d = ctx.enqueue_create_buffer[DType.bool](batch)
    var upd_d = ctx.enqueue_create_buffer[DType.bool](batch)
    var y_split = ctx.enqueue_create_buffer[dtype](y_n)
    var y_joined = ctx.enqueue_create_buffer[dtype](y_n)
    var pool_split = ctx.enqueue_create_buffer[DType.float32](pool_n)
    var pool_joined = ctx.enqueue_create_buffer[DType.float32](pool_n)
    var y_split_back = ctx.enqueue_create_host_buffer[dtype](y_n)
    var y_joined_back = ctx.enqueue_create_host_buffer[dtype](y_n)
    var pool_split_back = ctx.enqueue_create_host_buffer[DType.float32](pool_n)
    var pool_joined_back = ctx.enqueue_create_host_buffer[DType.float32](pool_n)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(d_d, d_h)
    ctx.enqueue_copy(dtb_d, dtb_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(dt_d, dt_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(c_d, c_h)
    ctx.enqueue_copy(qsl_d, qsl_h)
    ctx.enqueue_copy(slot_d, slot_h)
    ctx.enqueue_copy(his_d, his_h)
    ctx.enqueue_copy(upd_d, his_h)

    var a_t = TileTensor(a_d, row_major(nheads))
    var d_t = TileTensor(d_d, row_major(nheads))
    var dtb_t = TileTensor(dtb_d, row_major(nheads))
    var slot_t = TileTensor(slot_d, row_major(batch))
    var his_t = TileTensor(his_d, row_major(batch))
    var upd_t = TileTensor(upd_d, row_major(batch))
    var qsl_t = TileTensor(qsl_d, row_major(batch + 1))
    var x_t = TileTensor(x_d, row_major(total_len, nheads, head_dim))
    var dt_t = TileTensor(dt_d, row_major(total_len, nheads))
    var b_t = TileTensor(b_d, row_major(total_len, ngroups, DSTATE))
    var c_t = TileTensor(c_d, row_major(total_len, ngroups, DSTATE))
    var y_split_t = TileTensor(y_split, row_major(total_len, nheads, head_dim))
    var y_joined_t = TileTensor(
        y_joined, row_major(total_len, nheads, head_dim)
    )
    var pool_split_t = TileTensor(
        pool_split, row_major(batch, nheads, head_dim, DSTATE)
    )
    var pool_joined_t = TileTensor(
        pool_joined, row_major(batch, nheads, head_dim, DSTATE)
    )

    def x_fn[
        width: Int, alignment: Int
    ](t: Int, h: Int, p: Int) {var x_t} -> SIMD[dtype, width]:
        return x_t.load[width=width, alignment=alignment * 2]((t, h, p))

    def dt_fn[
        width: Int, alignment: Int
    ](t: Int, h: Int) {var dt_t} -> SIMD[dtype, width]:
        return dt_t.load[width=width, alignment=alignment * 2]((t, h))

    def b_fn[
        width: Int, alignment: Int
    ](t: Int, g: Int, n: Int) {var b_t} -> SIMD[dtype, width]:
        return b_t.load[width=width, alignment=alignment * 2]((t, g, n))

    def c_fn[
        width: Int, alignment: Int
    ](t: Int, g: Int, n: Int) {var c_t} -> SIMD[dtype, width]:
        return c_t.load[width=width, alignment=alignment * 2]((t, g, n))

    def slot_fn[
        width: Int, alignment: Int
    ](b: Int) {var slot_t} -> SIMD[DType.uint32, width]:
        return slot_t.load[width=width]((b,))

    var x_addr = Int(x_d.unsafe_ptr())
    var x_ld = nheads * head_dim
    var b_addr = Int(b_d.unsafe_ptr())
    var c_addr = Int(c_d.unsafe_ptr())
    var bc_ld = ngroups * DSTATE

    # The same `n`-deep chain twice, into two independently allocated pairs of
    # y and pool, so the caller can compare the two.
    def chain_split(lctx: DeviceContext, n: Int) raises {imm}:
        for _ in range(n):
            mamba2_ssd_chunked_fwd_inplace_gpu[DSTATE, dt_softplus=True](
                lctx,
                total_len,
                batch,
                ngroups,
                a_t,
                d_t,
                dtb_t,
                y_split_t,
                pool_split_t,
                qsl_t,
                his_t,
                upd_t,
                x_fn,
                dt_fn,
                b_fn,
                c_fn,
                slot_fn,
                x_addr=x_addr,
                x_ld=x_ld,
                b_addr=b_addr,
                c_addr=c_addr,
                bc_ld=bc_ld,
            )

    def chain_joined(lctx: DeviceContext, n: Int) raises {imm}:
        for _ in range(n):
            mamba2_ssd_chunked_fwd_inplace_gpu[DSTATE, dt_softplus=True](
                lctx,
                total_len,
                batch,
                ngroups,
                a_t,
                d_t,
                dtb_t,
                y_joined_t,
                pool_joined_t,
                qsl_t,
                his_t,
                upd_t,
                x_fn,
                dt_fn,
                b_fn,
                c_fn,
                slot_fn,
                x_addr=x_addr,
                x_ld=x_ld,
                b_addr=b_addr,
                c_addr=c_addr,
                bc_ld=bc_ld,
            )

    def chain_layers(lctx: DeviceContext) raises {imm}:
        chain_joined(lctx, layers)

    # The two runs take identical inputs and their scratch comes from
    # different memory, so they have to agree bit for bit: a value that depends
    # on scratch the kernels only read, or on where the allocator put it, shows
    # up here as a mismatch.
    ctx.enqueue_copy(pool_split, pool_h)
    ctx.enqueue_copy(pool_joined, pool_h)
    chain_split(ctx, layers)
    chain_joined(ctx, layers)
    ctx.enqueue_copy(y_split_back, y_split)
    ctx.enqueue_copy(y_joined_back, y_joined)
    ctx.enqueue_copy(pool_split_back, pool_split)
    ctx.enqueue_copy(pool_joined_back, pool_joined)
    ctx.synchronize()
    for i in range(y_n):
        if y_split_back[i] != y_joined_back[i]:
            raise Error(String("chained SSD layer y mismatch at index ", i))
    for i in range(pool_n):
        if pool_split_back[i] != pool_joined_back[i]:
            raise Error(String("chained SSD layer pool mismatch at index ", i))
    print(t"SSD_LAYERS_PARITY,ok,tokens={total_len},layers={layers}")

    for _ in range(WARMUP):
        chain_layers(ctx)
    var wall_times = List[Int](capacity=SAMPLES)
    for _ in range(SAMPLES):
        wall_times.append(ctx.execution_time(chain_layers, 1))
    var host_times = List[Int](capacity=SAMPLES)
    for _ in range(SAMPLES):
        var start = perf_counter_ns()
        chain_layers(ctx)
        var stop = perf_counter_ns()
        ctx.synchronize()
        host_times.append(stop - start)

    var description = String(
        t"kernel=mamba2_ssd_chunked token_len={total_len} heads={nheads} "
    ) + String(t"head_dim={head_dim} groups={ngroups} layers={layers}")
    for sample in range(SAMPLES):
        print(t"sample {description} index={sample} ns={wall_times[sample]}")
    print(
        t"summary {description} samples={SAMPLES}"
        t" median_ns={_median_ns(wall_times)}"
        t" host_median_ns={_median_ns(host_times)}"
    )
    var benchmark = Bench()
    var report = Report()
    for elapsed in wall_times:
        report.runs.append(Batch(Int(elapsed), 1, True))
    benchmark.info_vec.append(BenchmarkInfo(description, report^))
    benchmark.dump_report()
    _ = a_d^
    _ = d_d^
    _ = dtb_d^
    _ = x_d^
    _ = dt_d^
    _ = b_d^
    _ = c_d^
    _ = qsl_d^
    _ = slot_d^
    _ = his_d^
    _ = upd_d^
    _ = y_split^
    _ = y_joined^
    _ = pool_split^
    _ = pool_joined^


def main() raises:
    var nheads = arg_parse("ssd-heads", 0)
    var head_dim = arg_parse("ssd-head-dim", 0)
    var ngroups = arg_parse("ssd-groups", 0)
    var layers = arg_parse("ssd-layers", 1)
    var lengths_arg = arg_parse("sequence_lengths", String(""))
    if nheads <= 0 or head_dim <= 0 or ngroups <= 0:
        raise Error(
            "positive --ssd-heads, --ssd-head-dim and --ssd-groups are required"
        )
    if nheads % ngroups != 0:
        raise Error("groups must divide the head count")
    if layers <= 0:
        raise Error("--ssd-layers must be positive")
    if not lengths_arg:
        raise Error(
            "--sequence_lengths is required; see bench_mamba2_ssd_layers.yaml"
        )
    var seq_lengths = List[Int]()
    for token in lengths_arg.strip("[]").split(","):
        var length = Int(token.strip())
        if length <= 0:
            raise Error("sequence lengths must be positive")
        seq_lengths.append(length)
    with DeviceContext() as ctx:
        comptime if not (ctx.target.is_nvidia_gpu() or ctx.target.is_amd_gpu()):
            raise Error("the chunked SSD benchmark needs a CUDA or HIP GPU")
        seed(1234)
        print(t"SSD_LAYERS_BENCHMARK device={ctx.device().get_name()}")
        _bench_shape(ctx, nheads, head_dim, ngroups, seq_lengths, layers)
