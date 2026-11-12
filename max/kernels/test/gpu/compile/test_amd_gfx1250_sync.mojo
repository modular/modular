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
"""Checks the gfx1250 counter waits, split barrier and wave-priority wrappers.

Every wrapper is read back twice: the LLVM IR must call the wrapper's intrinsic
with the immediate the wrapper was given, and the AMDGPU assembly must contain
the matching instruction. One wrapper per kernel matters for the counters,
because the backend merges a load wait and an LDS wait that could be issued
together into a single `s_wait_loadcnt_dscnt`. The split barrier's signal and
wait share one kernel, since they are a pair and each instruction is checked on
its own; every other wrapper gets a kernel of its own.

Each kernel also has to meet the library's register budget: zero VGPR and SGPR
spills, an empty private segment, and no `scratch_*` load or store in the
kernel's own body. The private segment is checked separately because memory that
round-trips through it reports no spills at all; a build with assertions on
reserves a small frame for the failing path's message, which is the only
allowance made. Runs on a CPU-only machine; nothing is launched.
"""

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.sync import (
    AMDScheduleBarrierMask,
    s_barrier_signal,
    s_barrier_wait,
    s_setprio,
    s_setprio_inc_wg,
    s_wait_asynccnt,
    s_wait_dscnt,
    s_wait_kmcnt,
    s_wait_loadcnt,
    s_wait_storecnt,
    s_wait_tensorcnt,
)
from std.sys import get_defined_string
from std.sys._build import is_debug_build
from std.testing import assert_equal, assert_true

comptime MI455X_TARGET = get_gpu_target["mi455x"]()

comptime ASSERT_MODE = get_defined_string["ASSERT", "safe"]()
"""The `-D ASSERT=` level this test was compiled with; `safe` when unset."""

comptime FULL_ASSERTS = (
    ASSERT_MODE == "all" or ASSERT_MODE == "warn" or is_debug_build()
)
"""Whether every `debug_assert` in the compiled kernel is live.

Mirrors `std.builtin.debug_assert._assert_enabled`: the kernel's code changes
when `-D ASSERT=all` (bazel's setting) arms the assertions hidden inside the
abstractions it calls, and the budget checks below have to accept that build as
well as the default one.
"""

comptime ASSERT_FRAME_BYTES = 32
"""Private bytes an armed assertion reserves: the failing path's message frame.

Allowed only when `FULL_ASSERTS`. A kernel body that reserves more is keeping
state in the private segment rather than in registers."""


def _loadcnt_kernel():
    s_wait_loadcnt[1]()


def _storecnt_kernel():
    s_wait_storecnt[2]()


def _dscnt_kernel():
    s_wait_dscnt[3]()


def _kmcnt_kernel():
    s_wait_kmcnt[4]()


def _asynccnt_kernel():
    s_wait_asynccnt[5]()


def _tensorcnt_kernel():
    s_wait_tensorcnt[6]()


def _barrier_kernel():
    s_barrier_signal[-1]()
    s_barrier_wait[-1]()


def _setprio_kernel():
    s_setprio[2]()


def _setprio_inc_wg_kernel():
    s_setprio_inc_wg[3]()


def _assert_register_budget(asm: String, label: String, marker: String) raises:
    """Asserts zero spills and no private state in the kernel issuing `marker`.

    Memory that round-trips through the private segment reports no spills, so
    the segment size and the body's `scratch_*` traffic are the evidence.
    """
    var private_bytes = 0
    for directive in [
        ".vgpr_spill_count",
        ".sgpr_spill_count",
        ".private_segment_fixed_size",
    ]:
        var found = asm.find(directive + ":")
        assert_true(
            found is not None,
            String(t"assembly metadata is missing {directive}"),
        )
        var rest = asm[byte = found.value() + directive.byte_length() + 1 :]
        var line_end = rest.find("\n").or_else(rest.byte_length())
        var value = Int(String(rest[byte=0:line_end]).strip())
        if directive == ".private_segment_fixed_size":
            private_bytes = value
        else:
            assert_equal(value, 0, String(t"{label}: {directive}"))
    var limit = 0 if not FULL_ASSERTS else ASSERT_FRAME_BYTES
    assert_true(
        private_bytes <= limit,
        String(
            t"{label}: {private_bytes} private bytes, over the {limit} allowed"
        ),
    )
    # The kernel is the one non-local function whose body holds `marker`.
    var lines = asm.splitlines()
    var body = String("")
    var start = -1
    for index in range(len(lines)):
        var text = String(lines[index]).strip()
        if text.startswith(".type") and text.endswith(",@function"):
            # A `.L` symbol labels a block of the function before it, not a
            # function of its own.
            var name = text.removeprefix(".type").strip()
            if not name.startswith(".L"):
                start = index
        elif start >= 0 and text.startswith(".Lfunc_end"):
            var candidate = String("")
            for inner in range(start, index):
                candidate += String(lines[inner]) + "\n"
            if marker in candidate:
                body = candidate
            start = -1
    assert_true(
        body.byte_length() > 0 and "scratch_" not in body,
        String(t"{label}: no scratch-free kernel function in the dump"),
    )


def _assert_wrapper[
    kernel: def() thin -> None, call: StaticString, text: StaticString
]() raises:
    """Asserts `kernel` calls `call` in the IR and emits `text` in the ASM.

    The operand is part of `call` (`"s.wait.loadcnt(i16 1)"`), so the check pins
    the width and value of the immediate and not just the intrinsic's name.
    """
    var ir = _compile_code[
        kernel, target=MI455X_TARGET, emission_kind="llvm"
    ]().asm
    assert_true(
        String(t"call void @llvm.amdgcn.{call}") in ir,
        String(t"expected the LLVM IR to call llvm.amdgcn.{call}"),
    )

    var asm = _compile_code[
        kernel, target=MI455X_TARGET, emission_kind="asm"
    ]().asm
    assert_true(
        String(t"\t{text}") in asm,
        String(t"expected the assembly to contain `{text}`"),
    )
    _assert_register_budget(asm, String(text), String(text))


def test_s_wait_loadcnt() raises:
    _assert_wrapper[
        _loadcnt_kernel, "s.wait.loadcnt(i16 1)", "s_wait_loadcnt 0x1"
    ]()


def test_s_wait_storecnt() raises:
    _assert_wrapper[
        _storecnt_kernel, "s.wait.storecnt(i16 2)", "s_wait_storecnt 0x2"
    ]()


def test_s_wait_dscnt() raises:
    _assert_wrapper[_dscnt_kernel, "s.wait.dscnt(i16 3)", "s_wait_dscnt 0x3"]()


def test_s_wait_kmcnt() raises:
    _assert_wrapper[_kmcnt_kernel, "s.wait.kmcnt(i16 4)", "s_wait_kmcnt 0x4"]()


def test_s_wait_asynccnt() raises:
    _assert_wrapper[
        _asynccnt_kernel, "s.wait.asynccnt(i16 5)", "s_wait_asynccnt 0x5"
    ]()


def test_s_wait_tensorcnt() raises:
    _assert_wrapper[
        _tensorcnt_kernel, "s.wait.tensorcnt(i16 6)", "s_wait_tensorcnt 0x6"
    ]()


def test_s_barrier_signal_and_wait() raises:
    _assert_wrapper[
        _barrier_kernel, "s.barrier.signal(i32 -1)", "s_barrier_signal -1"
    ]()
    _assert_wrapper[
        _barrier_kernel, "s.barrier.wait(i16 -1)", "s_barrier_wait -1"
    ]()


def test_s_setprio() raises:
    _assert_wrapper[_setprio_kernel, "s.setprio(i16 2)", "s_setprio 2"]()


def test_s_setprio_inc_wg() raises:
    _assert_wrapper[
        _setprio_inc_wg_kernel, "s.setprio.inc.wg(i16 3)", "s_setprio_inc_wg 3"
    ]()


def test_schedule_barrier_mask_values() raises:
    """Pins the LLVM `sched_barrier` mask bits that `pin_interleave` relies on.
    """
    assert_equal(Int(AMDScheduleBarrierMask.MFMA), 0x008)
    assert_equal(Int(AMDScheduleBarrierMask.ALL_DS), 0x080)
    assert_equal(Int(AMDScheduleBarrierMask.DS_READ), 0x100)
    assert_equal(Int(AMDScheduleBarrierMask.DS_WRITE), 0x200)
    assert_equal(Int(AMDScheduleBarrierMask.TRANS), 0x400)
    assert_equal(Int(AMDScheduleBarrierMask.LDSDMA), 0x800)


def main() raises:
    test_s_wait_loadcnt()
    test_s_wait_storecnt()
    test_s_wait_dscnt()
    test_s_wait_kmcnt()
    test_s_wait_asynccnt()
    test_s_wait_tensorcnt()
    test_s_barrier_signal_and_wait()
    test_s_setprio()
    test_s_setprio_inc_wg()
    test_schedule_barrier_mask_values()
    print("gfx1250 sync wrappers: IR and assembly match the contract")
