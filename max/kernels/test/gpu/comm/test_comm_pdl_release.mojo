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

"""Peers' reads must retire before a collective releases its PDL dependent.

The collectives end with `_multi_gpu_barrier[is_start=False]`, which tells the
peers "I am done reading your buffer" and then spins, in the first warp only,
until every peer has said the same about this rank's buffer. Another warp that
reaches `launch_dependent_grids` (or the `PDL` exit) while the first is still
spinning would let the dependent start while a peer is still reading this
rank's input.

Each probe runs the collective with a dependent that overwrites this rank's
input with zeros before its own `wait_on_dependent_grids`, then requires every
output to match, bit for bit, the output of a run without that dependent. A
premature release shows up as a peer that summed or copied the zeros.

Each collective is probed launched both with and without the PDL attribute. The
attribute that lets a dependent start early sits on the dependent's launch, so a
collective launched without it still releases a dependent that carries it.

The allgather is probed twice: with a 16-byte-aligned length it takes the TMA
kernel on SM100, and with an odd element count the non-TMA P2P kernel. Its
wrapper launches both with the PDL attribute, so its reference differs from the
probe run only in the overwriting dependent. The multimem broadcast is not
probed: no peer reads the root's input and no rank's input is read after the
release, so there is no read-after-release hazard to observe.
"""

from std.atomic import Atomic, Ordering
from std.memory import bitcast
from std.sys import size_of

from comm import Signal, group_end, group_start
from comm.allgather import allgather
from comm.allreduce import allreduce
from comm.broadcast import broadcast
from comm.reducescatter import reducescatter
from comm.reducescatter_rmsnorm import reducescatter_rmsnorm
from comm.scatter import scatter
from comm.sync import enable_p2p, init_signal_buffer
from layout import Coord, TileTensor, row_major
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceStream, HostBuffer
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    _SUPPORT_PDL_LAUNCH,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.testing import assert_true
from std.utils.index import Index

comptime dtype = DType.bfloat16
comptime ROWS = 256
comptime COLS = 4096
comptime LENGTH = ROWS * COLS
# Probe iterations per collective. A premature release is a race against the
# peer's remaining reads, so a single iteration only sometimes catches it.
comptime ITERS = 300

# The slow rank's peer-read traffic: HOG_ROUNDS passes over HOG_ELEMS words.
comptime HOG_ELEMS = 16 * 1024 * 1024
comptime HOG_ROUNDS = 8
comptime HOG_GRID = 592

comptime KIND_ALLREDUCE = 0
comptime KIND_ALLGATHER = 1
comptime KIND_BROADCAST = 2
comptime KIND_SCATTER = 3
comptime KIND_REDUCESCATTER = 4
comptime KIND_REDUCESCATTER_RMSNORM = 5
comptime KIND_ALLGATHER_P2P = 6

# Output buffers a rank owns: the allgather writes one per peer and the fused
# reduce-scatter its normed and summed shards; the rest use one.
comptime NUM_OUTS = 2

comptime InFlat = TileTensor[dtype, type_of(row_major(LENGTH)), ImmutAnyOrigin]
comptime OutFlat = TileTensor[
    mut=True, dtype, type_of(row_major(LENGTH)), MutAnyOrigin
]
# An odd element count is not 16-byte aligned, which steers the allgather onto
# the non-TMA P2P kernel.
comptime ODD = LENGTH - 1
comptime InOdd = TileTensor[dtype, type_of(row_major(ODD)), ImmutAnyOrigin]
comptime OutOdd = TileTensor[
    mut=True, dtype, type_of(row_major(ODD)), MutAnyOrigin
]
comptime Row2D = type_of(row_major(Coord(Index(0, COLS))))
comptime In2D = TileTensor[dtype, Row2D, ImmutAnyOrigin]
comptime Shard2D = TileTensor[mut=True, dtype, Row2D, MutAnyOrigin]
comptime Gamma1D = TileTensor[
    dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
]


def _early_overwrite(dst: MutPointer[Scalar[dtype], MutAnyOrigin], n: Int32):
    """Zeroes `dst` before its own PDL wait, on purpose.

    Stands in for the next kernel reusing the buffer a collective just read.
    """
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(n), stride):
        dst[i] = Scalar[dtype](0)
    wait_on_dependent_grids()


def _hog(
    src: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    sink: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    n: Int32,
    rounds: Int32,
):
    """Saturates the link that reads `src` from a peer GPU.

    Slows the collective on this GPU relative to its peer, so the peer
    releases its dependent while this GPU is still reading the peer's input.
    """
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    var acc = UInt32(0)
    for _ in range(Int(rounds)):
        for i in range(tid, Int(n), stride):
            acc += src.unsafe_load[volatile=True](i)
    sink[tid] = acc


def _flag_diff(
    a: MutPointer[Scalar[dtype], MutAnyOrigin],
    b: MutPointer[Scalar[dtype], MutAnyOrigin],
    n: Int32,
    flag: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
):
    """Sets `flag[0]` when `a` and `b` differ in any bit of `[0, n)`."""
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(n), stride):
        if bitcast[DType.uint16](a[i]) != bitcast[DType.uint16](b[i]):
            Atomic[Scalar[DType.uint32]].store[ordering=Ordering.RELAXED](
                flag, UInt32(1)
            )


def _launch_collective[
    kind: Int, ngpus: Int, pdl_level: PDLLevel
](
    rank: Int,
    ctxs: List[DeviceContext],
    ins: List[DeviceBuffer[dtype]],
    mut outs: List[DeviceBuffer[dtype]],
    gammas: List[DeviceBuffer[dtype]],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
) raises:
    var in_flat = Array[InFlat, ngpus](
        fill_with=lambda (j: Int) -> InFlat: InFlat(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                ins[j].unsafe_ptr()
            ),
            row_major(LENGTH),
        )
    )
    var out_flat = Array[OutFlat, ngpus * NUM_OUTS](uninitialized=True)
    for j in range(ngpus * NUM_OUTS):
        out_flat[j] = OutFlat(
            outs[j].unsafe_ptr().as_unsafe_any_origin(), row_major(LENGTH)
        )

    comptime if kind == KIND_ALLREDUCE:
        allreduce[ngpus=ngpus, pdl_level=pdl_level](
            in_flat,
            out_flat[rank * NUM_OUTS],
            rank_sigs,
            ctxs[rank],
            my_rank=rank,
        )
    elif kind == KIND_ALLGATHER:
        comptime assert ngpus <= NUM_OUTS
        # Device d's output k holds peer k's input.
        var gather_out = Array[OutFlat, ngpus * ngpus](
            fill_with=lambda (j: Int) -> OutFlat: out_flat[
                (j // ngpus) * NUM_OUTS + (j % ngpus)
            ]
        )
        allgather(in_flat, gather_out, rank_sigs, ctxs[rank], rank)
    elif kind == KIND_ALLGATHER_P2P:
        var odd_in = Array[InOdd, ngpus](
            fill_with=lambda (j: Int) -> InOdd: InOdd(
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    ins[j].unsafe_ptr()
                ),
                row_major(ODD),
            )
        )
        var odd_out = Array[OutOdd, ngpus * ngpus](uninitialized=True)
        for j in range(ngpus * ngpus):
            odd_out[j] = OutOdd(
                outs[(j // ngpus) * NUM_OUTS + (j % ngpus)]
                .unsafe_ptr()
                .as_unsafe_any_origin(),
                row_major(ODD),
            )
        allgather(odd_in, odd_out, rank_sigs, ctxs[rank], rank)
    elif kind == KIND_BROADCAST:
        # Every rank pulls from the root's input.
        broadcast[ngpus, pdl_level=pdl_level](
            in_flat[0],
            out_flat[rank * NUM_OUTS],
            rank_sigs,
            ctxs[rank],
            0,
            rank=rank,
        )
    elif kind == KIND_SCATTER:
        var root_chunk = Array[InFlat, 1](
            fill_with=lambda (j: Int) -> InFlat: in_flat[0]
        )
        scatter[ngpus=ngpus, dp_size=1, pdl_level=pdl_level](
            root_chunk, out_flat[rank * NUM_OUTS], rank_sigs, ctxs[rank]
        )
    else:
        comptime shard_rows = ROWS // ngpus
        var in_2d = Array[In2D, ngpus](
            fill_with=lambda (j: Int) -> In2D: In2D(
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    ins[j].unsafe_ptr()
                ),
                row_major(Coord(Index(ROWS, COLS))),
            )
        )
        var shard_out = Array[Shard2D, ngpus](uninitialized=True)
        for j in range(ngpus):
            shard_out[j] = Shard2D(
                outs[j * NUM_OUTS].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(shard_rows, COLS))),
            )
        comptime if kind == KIND_REDUCESCATTER:
            reducescatter[
                dtype=dtype, ngpus=ngpus, axis=0, pdl_level=pdl_level
            ](in_2d, shard_out, rank_sigs, ctxs[rank], my_rank=rank)
        else:
            var sum_view = Shard2D(
                outs[rank * NUM_OUTS + 1].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(shard_rows, COLS))),
            )
            var gamma_view = Gamma1D(
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    gammas[rank].unsafe_ptr()
                ),
                row_major(Coord(Index(COLS))),
            )
            reducescatter_rmsnorm[pdl_level=pdl_level](
                in_2d,
                shard_out[rank],
                sum_view,
                gamma_view,
                Float32(1e-6),
                Scalar[dtype](0),
                rank_sigs,
                ctxs[rank],
                my_rank=rank,
            )


def _run_once[
    kind: Int, ngpus: Int, pdl_level: PDLLevel
](
    ctxs: List[DeviceContext],
    ins: List[DeviceBuffer[dtype]],
    mut outs: List[DeviceBuffer[dtype]],
    gammas: List[DeviceBuffer[dtype]],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    overwrite: Bool,
) raises:
    group_start()
    for rank in range(ngpus):
        _launch_collective[kind, ngpus, pdl_level](
            rank, ctxs, ins, outs, gammas, rank_sigs
        )
        if overwrite:
            ctxs[rank].enqueue_function[_early_overwrite](
                ins[rank],
                Int32(LENGTH),
                grid_dim=128,
                block_dim=256,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
    group_end()
    for rank in range(ngpus):
        ctxs[rank].synchronize()


def _refill[
    ngpus: Int
](
    ctxs: List[DeviceContext],
    ins: List[DeviceBuffer[dtype]],
    golden: List[DeviceBuffer[dtype]],
    outs: List[DeviceBuffer[dtype]],
) raises:
    """Restores every input and clears every output."""
    for i in range(ngpus):
        ctxs[i].enqueue_copy(ins[i], golden[i])
        for k in range(NUM_OUTS):
            ctxs[i].enqueue_memset[dtype](outs[i * NUM_OUTS + k], 0)
    for i in range(ngpus):
        ctxs[i].synchronize()


def _probe[
    kind: Int, ngpus: Int, pdl_level: PDLLevel
](ctxs: List[DeviceContext]) raises -> Int:
    """Returns how many of the `ITERS` overwrite runs changed an output.

    `pdl_level` is the collective's own launch level. The dependent always
    carries the PDL attribute, so the `OFF` probe covers a default-launched
    collective followed by a PDL kernel.
    """
    var ins = List[DeviceBuffer[dtype]](capacity=ngpus)
    var golden = List[DeviceBuffer[dtype]](capacity=ngpus)
    var outs = List[DeviceBuffer[dtype]](capacity=ngpus * NUM_OUTS)
    var gammas = List[DeviceBuffer[dtype]](capacity=ngpus)
    var signals = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var hosts = List[HostBuffer[dtype]](capacity=ngpus)
    var hog_srcs = List[DeviceBuffer[.uint32]](capacity=ngpus)
    var hog_sinks = List[DeviceBuffer[.uint32]](capacity=ngpus)
    var hog_streams = List[DeviceStream](capacity=ngpus)

    for i in range(ngpus):
        var c = ctxs[i]
        ins.append(c.enqueue_create_buffer[dtype](LENGTH))
        golden.append(c.enqueue_create_buffer[dtype](LENGTH))
        gammas.append(c.enqueue_create_buffer[dtype](COLS))
        for _ in range(NUM_OUTS):
            outs.append(c.enqueue_create_buffer[dtype](LENGTH))
        signals.append(
            c.create_buffer_sync[.uint8](
                size_of[Signal]() + ngpus * size_of[dtype]() * LENGTH
            )
        )
        init_signal_buffer(signals[i], c)
        hog_srcs.append(c.enqueue_create_buffer[.uint32](HOG_ELEMS))
        c.enqueue_memset[.uint32](hog_srcs[i], 1)
        hog_sinks.append(c.enqueue_create_buffer[.uint32](HOG_GRID * 1024))
        hog_streams.append(c.create_stream())

        var h = c.enqueue_create_host_buffer[dtype](LENGTH)
        for j in range(LENGTH):
            h[j] = Scalar[dtype](i + 1) + Scalar[dtype](j % 251)
        c.enqueue_copy(golden[i], h)
        var g = c.enqueue_create_host_buffer[dtype](COLS)
        for j in range(COLS):
            g[j] = (Float64(j + COLS) / Float64(COLS)).cast[dtype]()
        c.enqueue_copy(gammas[i], g)
        hosts.append(h^)
        hosts.append(g^)
    for i in range(ngpus):
        ctxs[i].synchronize()

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signals[i])
    )

    # Reference: the same kernel, no PDL, nothing overwritten.
    _refill[ngpus](ctxs, ins, golden, outs)
    _run_once[kind, ngpus, PDLLevel.OFF](
        ctxs, ins, outs, gammas, rank_sigs, False
    )
    var refs = List[DeviceBuffer[dtype]](capacity=ngpus * NUM_OUTS)
    var flags = List[DeviceBuffer[.uint32]](capacity=ngpus)
    var flag_hosts = List[HostBuffer[.uint32]](capacity=ngpus)
    for i in range(ngpus):
        for k in range(NUM_OUTS):
            refs.append(ctxs[i].enqueue_create_buffer[dtype](LENGTH))
            ctxs[i].enqueue_copy(refs[i * NUM_OUTS + k], outs[i * NUM_OUTS + k])
        flags.append(ctxs[i].enqueue_create_buffer[.uint32](1))
        flag_hosts.append(ctxs[i].enqueue_create_host_buffer[.uint32](1))
    for i in range(ngpus):
        ctxs[i].synchronize()

    # Load every module before the timed region: a first-use module load on
    # one rank stalls its host thread while the peer spins at the barrier.
    for i in range(ngpus):
        ctxs[i].enqueue_function[_early_overwrite](
            ins[i],
            Int32(0),
            grid_dim=1,
            block_dim=32,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        ctxs[i].enqueue_function[_flag_diff](
            outs[i * NUM_OUTS],
            refs[i * NUM_OUTS],
            Int32(0),
            flags[i],
            grid_dim=1,
            block_dim=32,
        )
        ctxs[i].synchronize()
        hog_streams[i].enqueue_function(
            ctxs[i].compile_function[_hog](),
            hog_srcs[(i + 1) % ngpus],
            hog_sinks[i],
            Int32(0),
            Int32(0),
            grid_dim=1,
            block_dim=32,
        )
        hog_streams[i].synchronize()
    _run_once[kind, ngpus, pdl_level](ctxs, ins, outs, gammas, rank_sigs, False)

    var bad_iters = 0
    for it in range(ITERS):
        _refill[ngpus](ctxs, ins, golden, outs)
        # Alternate which rank is slowed by reads of its peer's memory.
        var slow = it % ngpus
        hog_streams[slow].enqueue_function(
            ctxs[slow].compile_function[_hog](),
            hog_srcs[(slow + 1) % ngpus],
            hog_sinks[slow],
            Int32(HOG_ELEMS),
            Int32(HOG_ROUNDS),
            grid_dim=HOG_GRID,
            block_dim=1024,
        )
        _run_once[kind, ngpus, pdl_level](
            ctxs, ins, outs, gammas, rank_sigs, True
        )
        hog_streams[slow].synchronize()
        for i in range(ngpus):
            ctxs[i].enqueue_memset[.uint32](flags[i], 0)
            for k in range(NUM_OUTS):
                ctxs[i].enqueue_function[_flag_diff](
                    outs[i * NUM_OUTS + k],
                    refs[i * NUM_OUTS + k],
                    Int32(LENGTH),
                    flags[i],
                    grid_dim=128,
                    block_dim=256,
                )
            ctxs[i].enqueue_copy(flag_hosts[i], flags[i])
        var mismatched = False
        for i in range(ngpus):
            ctxs[i].synchronize()
            if flag_hosts[i][0] != 0:
                mismatched = True
        if mismatched:
            bad_iters += 1

    _ = refs^
    _ = flags^
    _ = flag_hosts^
    _ = ins^
    _ = golden^
    _ = outs^
    _ = gammas^
    _ = signals^
    _ = hosts^
    _ = hog_srcs^
    _ = hog_sinks^
    _ = hog_streams^
    _ = rank_sigs^
    return bad_iters


def main() raises:
    comptime if _SUPPORT_PDL_LAUNCH:
        assert_true(
            DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
        )
        assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

        comptime ngpus = 2
        var ctxs = List[DeviceContext]()
        for i in range(ngpus):
            ctxs.append(DeviceContext(device_id=i))

        var failed = 0
        # The allgather wrapper always launches with PDL, so one level covers
        # it.
        var bad = _probe[KIND_ALLGATHER, ngpus, PDLLevel.ON](ctxs)
        print("allgather (TMA): corrupted", bad, "of", ITERS)
        failed += bad
        bad = _probe[KIND_ALLGATHER_P2P, ngpus, PDLLevel.ON](ctxs)
        print("allgather (non-TMA P2P): corrupted", bad, "of", ITERS)
        failed += bad

        # Both launch levels: a collective launched without the PDL attribute
        # still releases a dependent that carries it.
        comptime for level in range(2):
            comptime pdl = PDLLevel.ON if level == 1 else PDLLevel.OFF
            comptime tag = "pdl ON" if level == 1 else "pdl OFF"
            bad = _probe[KIND_ALLREDUCE, ngpus, pdl](ctxs)
            print(t"allreduce (1-stage, {tag}): corrupted {bad} of {ITERS}")
            failed += bad
            bad = _probe[KIND_BROADCAST, ngpus, pdl](ctxs)
            print(t"broadcast (1-stage, {tag}): corrupted {bad} of {ITERS}")
            failed += bad
            bad = _probe[KIND_SCATTER, ngpus, pdl](ctxs)
            print(t"scatter ({tag}): corrupted {bad} of {ITERS}")
            failed += bad
            bad = _probe[KIND_REDUCESCATTER, ngpus, pdl](ctxs)
            print(t"reducescatter ({tag}): corrupted {bad} of {ITERS}")
            failed += bad
            bad = _probe[KIND_REDUCESCATTER_RMSNORM, ngpus, pdl](ctxs)
            print(t"reducescatter_rmsnorm ({tag}): corrupted {bad} of {ITERS}")
            failed += bad

        assert_true(
            failed == 0,
            "a dependent overwrote a buffer a peer was still reading",
        )
