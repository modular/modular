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

# Host-only: compiles the multicast boundary's arrival tally for sm_100 and
# checks the emitted atomic is the acquire-release one. The ordering is what
# makes the last CTA's advance of the call number ordered after every CTA's
# probe read; a relaxed or release-only add is still correct on every
# correctness test the arm has (the windows are wide enough in practice), so
# only the codegen shows it.

from std.atomic import Atomic, Ordering
from std.testing import assert_true

from comm.allreduce_mcast_rmsnorm import _arrive_and_flip
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code

comptime SM100 = get_gpu_target["sm_100"]()


def arrive(
    state: MutPointer[UInt32, MutUntrackedOrigin],
) -> None:
    """Stands in for one CTA's arrival on the launch's tally."""
    _arrive_and_flip(state, 0, 64)


def arrive_relaxed(
    state: MutPointer[UInt32, MutUntrackedOrigin],
) -> None:
    """The same tally with the ordering removed: what the check must reject."""
    var arrived = Atomic[UInt32, scope="device"].fetch_add[
        ordering=Ordering.RELAXED
    ](state, UInt32(1))
    if Int(arrived) == 63:
        state.unsafe_store[width=1, volatile=True](1, UInt32(1))


def main() raises:
    var asm = _compile_code[arrive, target=SM100]().asm
    assert_true(
        "atom.acq_rel" in asm,
        String(
            (
                "the arrival tally is not an acquire-release atomic, so the"
                " last CTA's advance of the call number is not ordered after"
                " the others' probe reads:\n"
            ),
            asm,
        ),
    )

    # The contrast: the same shape with a relaxed add, which has to be absent
    # from what the check above accepts.
    var relaxed = _compile_code[arrive_relaxed, target=SM100]().asm
    assert_true(
        "atom.relaxed" in relaxed and "atom.acq_rel" not in relaxed,
        String("the relaxed contrast did not compile as expected:\n", relaxed),
    )
    print("mcar arrival tally: acquire-release atomic, PASS")
