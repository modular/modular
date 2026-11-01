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

# Host-only: compiles `_load_peers_rotated` for gfx950 and checks the AMD peer
# loads issue in rotated order, back to back. That order rests on compiler
# behavior no GPU test can see: alias analysis treating the empty asm in
# `_pin_load_order` as a write to the peers' memory, AMDGPU inserting no wait
# around an asm with no operands, and the loads staying non-invariant. If one
# changes, the loads fall back into add order (or wait for each other) and the
# rank-agreement tests still pass, so only the codegen shows it.

from std.collections import Array
from std.testing import assert_equal

from comm.allreduce import _load_peers_rotated
from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host.compile import _compile_code, get_gpu_target

comptime NGPUS = 8
comptime WIDTH = 8
# Peer i is read 512 B past its base, so its 16-byte load carries i in its
# immediate offset.
comptime PEER_STRIDE = 256
comptime PEER_STRIDE_BYTES = PEER_STRIDE * 2

comptime PeerPtr = ImmPointer[BFloat16, ImmUntrackedOrigin]


def rotated_peer_sum[
    start: Int
](
    peer_bases: Array[PeerPtr, NGPUS],
    dst: MutPointer[Float32, MutUntrackedOrigin],
):
    var peers = Array[PeerPtr, NGPUS](
        fill_with_unrolled=lambda [i: Int]() -> PeerPtr: peer_bases[
            i
        ].unsafe_offset(i * PEER_STRIDE)
    )
    var offset = thread_idx.x * WIDTH
    # The allreduce kernels issue these loads after their start barrier. With
    # no memory operation before it, the first load would be tagged
    # `amdgpu.noclobber`, which lets the scheduler move it across the pins, so
    # a change that hoists the peer loads above `_multi_gpu_barrier` would lose
    # the rotation at rank 0 even though this test still passes.
    barrier()
    # Same call shape as the warp-tiling kernel, which passes its SIMD width as
    # the alignment.
    var vals = _load_peers_rotated[NGPUS, start, WIDTH, alignment=WIDTH](
        peers, offset
    )
    var acc = vals[0].cast[DType.float32]()
    comptime for g in range(1, NGPUS):
        acc += vals[g].cast[DType.float32]()
    dst.unsafe_store[width=WIDTH, alignment=32](offset, acc)


def _peer_load_order(asm: String) raises -> List[Int]:
    """Returns the peer each 16-byte global load reads, in issue order.

    Raises if a `vmcnt` wait sits between two of the loads.
    """
    var order = List[Int]()
    var waited = False
    for line in asm.split("\n"):
        if "global_load_dwordx4" in line:
            if waited:
                raise Error(
                    String("s_waitcnt vmcnt between peer loads:\n", asm)
                )
            var imm = 0
            var parts = line.split("offset:")
            if len(parts) > 1:
                imm = atol(parts[1].strip().split(" ")[0])
            order.append(imm // PEER_STRIDE_BYTES)
        elif len(order) > 0 and len(order) < NGPUS:
            if "s_waitcnt" in line and "vmcnt" in line:
                waited = True
    return order^


def test_rotated_issue_order[start: Int]() raises:
    var asm = _compile_code[
        rotated_peer_sum[start], target=get_gpu_target["mi355x"]()
    ]().asm
    var order = _peer_load_order(asm)
    assert_equal(len(order), NGPUS, String("peer loads:\n", asm))
    for i in range(NGPUS):
        assert_equal(
            order[i],
            (start + i) % NGPUS,
            String("start ", start, ": load ", i, " reads the wrong peer"),
        )


def main() raises:
    comptime for start in range(NGPUS):
        test_rotated_issue_order[start]()
