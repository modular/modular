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
# REQUIRES: NVIDIA-GPU
# RUN: %mojo-build %s -o %t
# RUN: mpirun --allow-run-as-root --bind-to none -n 2 %t
"""Cost of the cross-rank arrival on two B200s.

A Moka megakernel's cross-rank edge hands a producer's stores on rank p to a
consumer's reads on rank q. On one node the ranks reach each other's memory
over P2P, so the arrival is a signal word in symmetric memory: shmem_signal_op
for a bare flag, or shmem_put_signal_nbi when the flag rides with the payload,
the shape a dispatch publishes in.

Both ranks exchange one token per round, so the round trip is bidirectional
like a production dispatch. One CTA of one thread per PE keeps the number an
arrival's cost rather than a queue's.
"""

from std.testing import assert_true
from std.time import global_perf_counter_ns

from max.gpu.host import DeviceBuffer, HostBuffer
from shmem import *


def pingpong(
    sig: Pointer[UInt64, MutAnyOrigin],
    payload: Pointer[UInt8, MutAnyOrigin],
    payload_bytes_dev: Int32,
    base_dev: UInt64,
    rounds_dev: Int32,
    out_ns: Pointer[UInt64, MutAnyOrigin],
):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes
    var nbytes = Int(payload_bytes_dev)
    var nr = Int(rounds_dev)
    var base = base_dev

    # No device-wide barrier and no reset: the word only moves forward, so a
    # cell starts from the count the previous one left in it.
    var t0 = UInt64(global_perf_counter_ns())

    for k in range(nr):
        var v = base + UInt64(k) + 1
        if nbytes == 0:
            shmem_signal_op(sig, v, SHMEM_SIGNAL_SET, peer)
        else:
            shmem_put_signal_nbi(
                payload, payload, nbytes, sig, v, SHMEM_SIGNAL_SET, peer
            )
        # The peer leaves round k as soon as our token lands and may already
        # have set round k + 1's value, so wait for at least v, not exactly v.
        shmem_signal_wait_until(sig, SHMEM_CMP_GE, v)

    out_ns[unsafe_offset=0] = UInt64(global_perf_counter_ns()) - t0


def _cell(
    ctx: SHMEMContext,
    sig: DeviceBuffer[DType.uint64],
    payload: SHMEMBuffer[DType.uint8],
    timing: SHMEMBuffer[DType.uint64],
    host: HostBuffer[DType.uint64],
    bytes: Int,
    base: UInt64,
    rounds: Int,
) raises:
    """Times one payload-size cell and prints its round trip."""
    ctx.enqueue_function_collective_checked[pingpong](
        sig,
        payload,
        Int32(bytes),
        base,
        Int32(rounds),
        timing,
        grid_dim=1,
        block_dim=1,
    )
    ctx.barrier_all()
    timing.enqueue_copy_to(host)
    ctx.synchronize()
    # The token made every round: the arrival chain is complete.
    assert_true(host[0] > 0, "the ping-pong token did not return")
    var rtt = Float64(host[0]) / Float64(rounds)
    print(
        "PINGPONG bytes=",
        bytes,
        " rounds=",
        rounds,
        " rtt_ns=",
        rtt,
        " one_way_ns=",
        rtt / 2.0,
    )


def test_pingpong(ctx: SHMEMContext) raises:
    var npes = shmem_n_pes()
    assert_true(npes >= 2, "needs at least two PEs")
    print("PINGPONG pEs=", npes)
    var sig = shmem_calloc[.uint64](1)
    var sig_buf = DeviceBuffer[DType.uint64](ctx._ctx, sig, 1, owning=False)
    var timing = ctx.enqueue_create_buffer[DType.uint64](1)
    var payload = ctx.enqueue_create_buffer[DType.uint8](65536)
    var host = ctx.get_device_context().enqueue_create_host_buffer[
        DType.uint64
    ](1)
    ctx.barrier_all()
    _cell(ctx, sig_buf, payload, timing, host, 0, 0, 2000)
    _cell(ctx, sig_buf, payload, timing, host, 512, 2000, 2000)
    _cell(ctx, sig_buf, payload, timing, host, 4096, 4000, 2000)
    _cell(ctx, sig_buf, payload, timing, host, 65536, 6000, 400)


def main() raises:
    with SHMEMContext() as ctx:
        test_pingpong(ctx)
