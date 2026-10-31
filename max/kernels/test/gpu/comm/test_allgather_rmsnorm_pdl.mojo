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

"""PDL ordering and hang test for the fused all-gather + RMSNorm kernel (bf16).

Each round runs, per GPU, `producer -> allgather_rmsnorm x2 -> consumer` in one
stream:

  * `producer` releases its dependents on ENTRY, then sleeps (a different time
    per rank and round) and only then writes the shard. The all-gather is thus
    resident long before its input exists, so a missing or misplaced
    `wait_on_dependent_grids` reads a stale shard and fails the bitwise check.
    It writes gamma too, with a different value each round, because callers
    can build gamma in the graph each step (a dtype cast of the weight): a
    gamma load above the wait reads the previous round's gamma and changes
    `normed_out`.
  * `allgather_rmsnorm` runs at each `PDLLevel`.
  * `consumer` is a PDL-launched copy of `normed_out` that waits before reading,
    so an early release by the all-gather cannot expose unwritten output.

Checked every round, per GPU: `sum_out` is bit-identical to the shards the
producers wrote (a gather is a verbatim copy), `normed_out` is bit-identical to
the `PDLLevel.OFF` arm's (same arithmetic) and to rank 0's, and the consumer's
copy equals `normed_out`. The "fat" variant uses multi-wave, shared-memory-heavy
producer/consumer grids to check that releasing dependents early cannot starve
a resident grid of the resources it needs to finish: a hang fails this test by
timeout.
"""

from std.sys import size_of
from std.sys.intrinsics import llvm_intrinsic
from std.testing import assert_equal, assert_true

from comm import Signal
from comm.allgather_rmsnorm import allgather_rmsnorm
from comm.reducescatter import ReduceScatterConfig
from comm.sync import (
    _multi_gpu_barrier,
    enable_p2p,
    init_signal_buffer,
    is_p2p_enabled,
)
from layout import Coord, TileTensor, row_major
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.utils.index import Index

comptime T = DType.bfloat16
comptime COLS = 6144
# Distinct small integers, so every bf16 value is exact and a stale shard (a
# different round, rank or row) never aliases the expected one.
comptime ROUND_PERIOD = 5
comptime FAT_SMEM_BYTES = 96 * 1024
comptime FAT_BLOCK = 512


@always_inline
def _shard_value(rnd: Int, rank: Int, row: Int, col: Int) -> Scalar[T]:
    return Float32(
        1 + (rnd % ROUND_PERIOD) + rank + (row % 13) + (col % 61)
    ).cast[T]()


@always_inline
def _gamma_value(rnd: Int, col: Int) -> Scalar[T]:
    return Float32(1 + (rnd % ROUND_PERIOD) + (col % 7)).cast[T]()


@always_inline
def _write_gamma(gamma: MutPointer[Scalar[T], MutAnyOrigin], rnd: Int32):
    """Publishes this round's gamma, as a graph-computed cast would."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, COLS, stride):
        gamma[i] = _gamma_value(Int(rnd), i)


def _producer(
    dst: MutPointer[Scalar[T], MutAnyOrigin],
    gamma: MutPointer[Scalar[T], MutAnyOrigin],
    n: Int32,
    rank: Int32,
    rnd: Int32,
    delay_ns: Int32,
):
    launch_dependent_grids()
    var left = Int(delay_ns)
    while left > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(20_000))
        left -= 20_000
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(n), stride):
        dst[i] = _shard_value(Int(rnd), Int(rank), i // COLS, i % COLS)
    _write_gamma(gamma, rnd)


def _signal_producer[
    ngpus: Int
](
    dst: MutPointer[Scalar[T], MutAnyOrigin],
    gamma: MutPointer[Scalar[T], MutAnyOrigin],
    n: Int32,
    rank: Int32,
    rnd: Int32,
    delay_ns: Int32,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
):
    """A producer that is itself a collective: it advances the barrier
    counters of the `Signal` buffers the all-gather will use, so an all-gather
    barrier that runs before its wait corrupts them."""
    launch_dependent_grids()
    var left = Int(delay_ns)
    while left > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(20_000))
        left -= 20_000
    _multi_gpu_barrier[ngpus, is_start=True](
        rank_sigs, rank_sigs[Int(rank)], Int(rank)
    )
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(n), stride):
        dst[i] = _shard_value(Int(rnd), Int(rank), i // COLS, i % COLS)
    _write_gamma(gamma, rnd)
    _multi_gpu_barrier[ngpus, is_start=False](
        rank_sigs, rank_sigs[Int(rank)], Int(rank)
    )


def _consumer(
    dst: MutPointer[Scalar[T], MutAnyOrigin],
    src: MutPointer[Scalar[T], MutAnyOrigin],
    n: Int32,
):
    wait_on_dependent_grids()
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(n), stride):
        dst[i] = src[i]


def _enqueue_producer(
    ctx: DeviceContext,
    dst: DeviceBuffer[T],
    gamma: DeviceBuffer[T],
    n: Int,
    rank: Int,
    rnd: Int,
    delay_ns: Int,
    fat: Bool,
) raises:
    if fat:
        ctx.enqueue_function[_producer](
            dst,
            gamma,
            Int32(n),
            Int32(rank),
            Int32(rnd),
            Int32(delay_ns),
            grid_dim=3 * ctx.default_device_info.sm_count,
            block_dim=FAT_BLOCK,
            shared_mem_bytes=FAT_SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                FAT_SMEM_BYTES
            ),
        )
    else:
        ctx.enqueue_function[_producer](
            dst,
            gamma,
            Int32(n),
            Int32(rank),
            Int32(rnd),
            Int32(delay_ns),
            grid_dim=64,
            block_dim=256,
        )


def _enqueue_consumer(
    ctx: DeviceContext,
    dst: DeviceBuffer[T],
    src: DeviceBuffer[T],
    n: Int,
    fat: Bool,
) raises:
    if fat:
        ctx.enqueue_function[_consumer](
            dst,
            src,
            Int32(n),
            grid_dim=3 * ctx.default_device_info.sm_count,
            block_dim=FAT_BLOCK,
            shared_mem_bytes=FAT_SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                FAT_SMEM_BYTES
            ),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
    else:
        ctx.enqueue_function[_consumer](
            dst,
            src,
            Int32(n),
            grid_dim=64,
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


def _warm_up(ctxs: List[DeviceContext]) raises:
    """Loads every non-collective kernel while no collective is spinning.
    The all-gather loads in `_warm_up_all_gather`.

    A first-time load with a shared-memory opt-in waits for the device to
    quiesce, which would deadlock against an all-gather spinning on a peer whose
    launches the single host thread has not issued yet.
    """
    for fat in range(2):
        for i in range(len(ctxs)):
            var a = ctxs[i].enqueue_create_buffer[T](COLS)
            var b = ctxs[i].enqueue_create_buffer[T](COLS)
            var g = ctxs[i].enqueue_create_buffer[T](COLS)
            _enqueue_producer(ctxs[i], a, g, COLS, i, 0, 0, fat == 1)
            _enqueue_consumer(ctxs[i], b, a, COLS, fat == 1)
            ctxs[i].synchronize()
            _ = a^
            _ = b^
            _ = g^


def _run_chain[
    ngpus: Int, pdl_level: PDLLevel
](
    num_rows: Int,
    rounds: Int,
    fat: Bool,
    ctxs: List[DeviceContext],
    mut ref_normed: List[List[Scalar[T]]],
    record_ref: Bool,
    sig_pred: Bool = False,
    launches: Int = 2,
) raises:
    var config = ReduceScatterConfig[T, ngpus](
        axis_size=num_rows, unit_numel=COLS, threads_per_gpu=0
    )
    comptime epsilon = Float32(1e-6)
    var weight_offset = Float32(1.0).cast[T]()
    var full_n = num_rows * COLS

    var shard_dev = List[DeviceBuffer[T]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[T]](capacity=ngpus)
    var normed = List[DeviceBuffer[T]](capacity=ngpus)
    var sum_full = List[DeviceBuffer[T]](capacity=ngpus)
    var cons = List[DeviceBuffer[T]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)

    # Negative, so it differs from every value `_gamma_value` produces: the
    # producer rewrites gamma each round, the way a graph-computed cast would,
    # and a kernel that reads it before its wait sees the previous round's.
    var gamma_host = List(length=COLS, fill=Scalar[T](-3))
    for i in range(ngpus):
        # 0-row ranks still need a valid backing pointer.
        shard_dev.append(
            ctxs[i].enqueue_create_buffer[T](
                max(1, config.rank_units(i) * COLS)
            )
        )
        gamma_dev.append(ctxs[i].enqueue_create_buffer[T](COLS))
        ctxs[i].enqueue_copy(gamma_dev[i], gamma_host)
        normed.append(ctxs[i].enqueue_create_buffer[T](full_n))
        sum_full.append(ctxs[i].enqueue_create_buffer[T](full_n))
        cons.append(ctxs[i].enqueue_create_buffer[T](full_n))
        signal_buffers.append(
            ctxs[i].create_buffer_sync[.uint8](size_of[Signal]())
        )

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )
    # One init for the whole arm: the barrier counters advance across rounds,
    # as they do in a served process.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], ctxs[i])
    for i in range(ngpus):
        ctxs[i].synchronize()

    comptime ShardType = TileTensor[
        T, type_of(row_major(Coord(Index(0, COLS)))), ImmutAnyOrigin
    ]
    comptime FullType = TileTensor[
        mut=True,
        T,
        type_of(row_major(Coord(Index(0, COLS)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        T, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var in_shards = Array[_, ngpus](
        fill_with=lambda (i: Int) -> ShardType: ShardType(
            rebind[ImmPointer[Scalar[T], ImmutAnyOrigin]](
                shard_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(config.rank_units(i), COLS))),
        )
    )

    var bad_sum = 0
    var bad_normed = 0
    var bad_repl = 0
    var bad_cons = 0
    for rnd in range(rounds):
        for i in range(ngpus):
            ctxs[i].enqueue_memset[T](normed[i], Scalar[T](0))
            ctxs[i].enqueue_memset[T](sum_full[i], Scalar[T](0))
            ctxs[i].enqueue_memset[T](cons[i], Scalar[T](0))
            # Skew the ranks and the rounds; some producers do not sleep.
            var slot = (rnd + 3 * i) % (ngpus + 1)
            if sig_pred:
                ctxs[i].enqueue_function[_signal_producer[ngpus]](
                    shard_dev[i],
                    gamma_dev[i],
                    Int32(config.rank_units(i) * COLS),
                    Int32(i),
                    Int32(rnd),
                    Int32(slot * 150_000),
                    rank_sigs,
                    grid_dim=64,
                    block_dim=256,
                )
            else:
                _enqueue_producer(
                    ctxs[i],
                    shard_dev[i],
                    gamma_dev[i],
                    config.rank_units(i) * COLS,
                    i,
                    rnd,
                    slot * 150_000,
                    fat,
                )
            var normed_view = FullType(
                normed[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(num_rows, COLS))),
            )
            var sum_view = FullType(
                sum_full[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(num_rows, COLS))),
            )
            var gamma_view = GammaType(
                rebind[ImmPointer[Scalar[T], ImmutAnyOrigin]](
                    gamma_dev[i].unsafe_ptr()
                ),
                row_major(Coord(Index(COLS))),
            )
            # Twice by default, so the kernel is also its own PDL predecessor.
            # That hides a stale read in the first launch, because the second
            # rewrites the outputs after the producer is done; `launches=1`
            # exposes it.
            for _ in range(launches):
                allgather_rmsnorm[pdl_level=pdl_level](
                    in_shards,
                    normed_view,
                    sum_view,
                    gamma_view,
                    epsilon,
                    weight_offset,
                    rank_sigs,
                    ctxs[i],
                    my_rank=i,
                )
            _enqueue_consumer(ctxs[i], cons[i], normed[i], full_n, fat)
        for i in range(ngpus):
            ctxs[i].synchronize()

        var normed0 = List[Scalar[T]](length=full_n, fill=Scalar[T](0))
        for i in range(ngpus):
            var n_h = List[Scalar[T]](length=full_n, fill=Scalar[T](0))
            var s_h = List[Scalar[T]](length=full_n, fill=Scalar[T](0))
            var c_h = List[Scalar[T]](length=full_n, fill=Scalar[T](0))
            ctxs[i].enqueue_copy(n_h, normed[i])
            ctxs[i].enqueue_copy(s_h, sum_full[i])
            ctxs[i].enqueue_copy(c_h, cons[i])
            ctxs[i].synchronize()
            if i == 0:
                normed0 = n_h.copy()
                if record_ref:
                    if len(ref_normed[rnd % ROUND_PERIOD]) == 0:
                        ref_normed[rnd % ROUND_PERIOD] = n_h.copy()
            for g in range(ngpus):
                var start = config.rank_unit_start(g)
                for r in range(config.rank_units(g)):
                    for c in range(COLS):
                        if (
                            s_h[(start + r) * COLS + c].to_bits()
                            != _shard_value(rnd, g, r, c).to_bits()
                        ):
                            bad_sum += 1
            for e in range(full_n):
                if n_h[e].to_bits() != normed0[e].to_bits():
                    bad_repl += 1
                if c_h[e].to_bits() != n_h[e].to_bits():
                    bad_cons += 1
                if (
                    len(ref_normed[rnd % ROUND_PERIOD]) == full_n
                    and n_h[e].to_bits()
                    != ref_normed[rnd % ROUND_PERIOD][e].to_bits()
                ):
                    bad_normed += 1

    print(
        String(
            "  ngpus=",
            ngpus,
            " M=",
            num_rows,
            " fat=",
            fat,
            " signal-predecessor=",
            sig_pred,
            " launches=",
            launches,
            " rounds=",
            rounds,
            ": sum_out errors=",
            bad_sum,
            " normed-vs-OFF errors=",
            bad_normed,
            " replication errors=",
            bad_repl,
            " consumer errors=",
            bad_cons,
        )
    )
    assert_equal(bad_sum, 0)
    assert_equal(bad_normed, 0)
    assert_equal(bad_repl, 0)
    assert_equal(bad_cons, 0)

    _ = shard_dev^
    _ = gamma_dev^
    _ = gamma_host^
    _ = normed^
    _ = sum_full^
    _ = cons^
    _ = signal_buffers^


def _warm_up_all_gather[ngpus: Int](ctxs: List[DeviceContext]) raises:
    """Loads the all-gather kernel at every PDL level before any variant runs.

    The signal-predecessor variant spins in a barrier on a peer whose launches
    the single host thread has not issued yet, so a first-time kernel load on
    that device (which can wait for it to quiesce) would hang. Each level is a
    separate kernel instance; a one-round plain chain loads all three while the
    only thing in flight is a sleeping producer.
    """
    var ref_normed = List[List[Scalar[T]]](
        length=ROUND_PERIOD, fill=List[Scalar[T]]()
    )
    _run_chain[ngpus, PDLLevel.OFF](1, 1, False, ctxs, ref_normed, True)
    _run_chain[ngpus, PDLLevel.ON](1, 1, False, ctxs, ref_normed, False)
    _run_chain[ngpus, PDLLevel.OVERLAP_AT_BEGINNING](
        1, 1, False, ctxs, ref_normed, False
    )


def _run_ngpus[ngpus: Int](ctxs: List[DeviceContext]) raises:
    _warm_up_all_gather[ngpus](ctxs)
    # M=1 leaves all but one rank with an empty shard; 40 is ragged across
    # every TP size; 128 is the top of the fused-path band.
    var rounds = 20
    for m in [1, 40, 128]:
        # Variant 0: plain producer; 1: multi-wave shared-memory-heavy
        # producer/consumer; 2: a producer that is itself a collective on the
        # same `Signal` buffers (not multi-wave: its barrier needs the whole
        # grid resident); 3: a plain producer and a single all-gather launch,
        # so the first launch's read of gamma is what the output shows.
        for variant in range(4):
            var ref_normed = List[List[Scalar[T]]](
                length=ROUND_PERIOD, fill=List[Scalar[T]]()
            )
            var fat = variant == 1
            var sig_pred = variant == 2
            var launches = 1 if variant == 3 else 2
            _run_chain[ngpus, PDLLevel.OFF](
                m, rounds, fat, ctxs, ref_normed, True, sig_pred, launches
            )
            _run_chain[ngpus, PDLLevel.ON](
                m, rounds, fat, ctxs, ref_normed, False, sig_pred, launches
            )
            _run_chain[ngpus, PDLLevel.OVERLAP_AT_BEGINNING](
                m, rounds, fat, ctxs, ref_normed, False, sig_pred, launches
            )


def main() raises:
    var num_devices = DeviceContext.number_of_devices()
    if num_devices < 2:
        print("Need at least 2 GPUs but only found", num_devices, "- skipping.")
        return
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")
    if not is_p2p_enabled():
        print("P2P not enabled, skipping test.")
        return

    var ctxs = List[DeviceContext]()
    for i in range(num_devices):
        ctxs.append(DeviceContext(device_id=i))
    _warm_up(ctxs)

    print("devices", num_devices)
    _run_ngpus[2](ctxs)
    if num_devices >= 4:
        _run_ngpus[4](ctxs)
    if num_devices >= 8:
        _run_ngpus[8](ctxs)
    print("All fused all-gather + RMSNorm PDL tests passed!")
