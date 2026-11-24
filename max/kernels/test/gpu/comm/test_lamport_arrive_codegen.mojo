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
"""Which add `lamport_arrive` lowers to, for each world size and architecture.

The arrival add that ends a Lamport call is the one place the 4-GPU `sm_100a`
configuration differs from the rest. The 4-GPU kernel must lower to a relaxed
`gpu` scope atom with no `fence.sc.sys` around it; every other cell must keep
the sequentially consistent system scope add. A GPU test cannot see this: both
adds are correct, and only the time of a call differs. The compile is for the
named target, so the test runs on any host.
"""

from std.testing import assert_equal, assert_false, assert_true

from comm.lamport import lamport_arrive, lamport_arrive_is_relaxed
from max.gpu.host.compile import _compile_code, get_gpu_target

comptime Counter = MutPointer[Scalar[DType.uint32], MutUntrackedOrigin]


def arrive_kernel[
    ngpus: Int, sm_version: StaticString
](counter: Counter, result: Counter):
    result.unsafe_write(lamport_arrive[ngpus, sm_version](counter))


def _asm[ngpus: Int, sm_version: StaticString]() raises -> String:
    return _compile_code[
        arrive_kernel[ngpus, sm_version],
        target=get_gpu_target[sm_version](),
    ]().asm


def _check_relaxed[ngpus: Int, sm_version: StaticString]() raises:
    var asm = _asm[ngpus, sm_version]()
    var msg = String("ngpus=", ngpus, " ", sm_version, ":\n", asm)
    assert_true(lamport_arrive_is_relaxed[ngpus, sm_version](), msg)
    assert_true("atom.relaxed.gpu" in asm, msg)
    assert_false("fence.sc.sys" in asm, msg)
    assert_false("atom.acquire.sys" in asm, msg)


def _check_default[ngpus: Int, sm_version: StaticString]() raises:
    var asm = _asm[ngpus, sm_version]()
    var msg = String("ngpus=", ngpus, " ", sm_version, ":\n", asm)
    assert_false(lamport_arrive_is_relaxed[ngpus, sm_version](), msg)
    assert_false("atom.relaxed.gpu" in asm, msg)
    assert_true("fence.sc.sys" in asm, msg)
    assert_true("atom.acquire.sys" in asm, msg)


def main() raises:
    _check_relaxed[4, "sm_100a"]()
    _check_default[1, "sm_100a"]()
    _check_default[2, "sm_100a"]()
    _check_default[8, "sm_100a"]()
    _check_default[4, "sm_90a"]()
    _check_default[8, "sm_90a"]()
    assert_equal(lamport_arrive_is_relaxed[4, "CDNA4"](), False)
    assert_equal(lamport_arrive_is_relaxed[4, "sm_103a"](), False)
