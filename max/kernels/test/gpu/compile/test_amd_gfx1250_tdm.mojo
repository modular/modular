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
"""IR and ASM checks for the gfx1250 TDM instruction wrappers.

The descriptor here is built from kernel arguments, so it lives in uniform
registers: the interesting failure is a descriptor that reaches the instruction
through VGPRs (`v_readfirstlane`, an `s_and_saveexec`-style uniformization or
worse, a per-lane TDM issue), which this test rejects. The instruction wrappers
are intrinsic calls, never inline assembly, so the IR must name
`llvm.amdgcn.tensor.load.to.lds` / `llvm.amdgcn.tensor.store.from.lds`.

Each kernel also has to meet the library's register budget: zero VGPR and SGPR
spills, an empty private segment, and no `scratch_*` load or store in the
kernel's own body. The private segment is checked separately because memory that
round-trips through it reports no spills at all; a build with assertions on
reserves a small frame for the failing path's message, which is the only
allowance made.
"""

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.tdm import TDMDescriptor, tensor_load_to_lds, tensor_store_from_lds
from std.sys import get_defined_string
from std.sys._build import is_debug_build
from std.testing import assert_equal, assert_false, assert_true

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


def _tdm_kernel(g_origin: Int64, lds_base: Int32, extent: Int32):
    var load_desc = TDMDescriptor.make_2d[
        DType.bfloat16, 64, 64, pad_interval_bytes=256, pad_amount_bytes=16
    ](g_origin, lds_base, extent, extent, 2048)
    tensor_load_to_lds(load_desc)
    var store_desc = TDMDescriptor.make_2d[DType.float32, 32, 16](
        g_origin, lds_base, extent, extent, 640
    )
    tensor_store_from_lds(store_desc)


def _policied_kernel(g_origin: Int64, lds_base: Int32, extent: Int32):
    var desc = TDMDescriptor.make_2d[DType.bfloat16, 64, 64](
        g_origin, lds_base, extent, extent, 2048
    )
    tensor_load_to_lds[cache_policy=17](desc)


def _assert_no_line_containing(
    asm: String, needle: String, context: String
) raises:
    for line in asm.splitlines():
        assert_false(
            needle in line,
            String(t"{context}: unexpected '{needle}': {line}"),
        )


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
        var at: Int
        if case .Some(found) = asm.find(directive + ":"):
            at = found
        else:
            raise Error(t"assembly metadata is missing {directive}")
        var rest = asm[byte = at + directive.byte_length() + 1 :]
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


def test_tdm_intrinsics_in_ir() raises:
    var ir = _compile_code[
        _tdm_kernel, target=MI455X_TARGET, emission_kind="llvm"
    ]().asm
    assert_true(
        "llvm.amdgcn.tensor.load.to.lds" in ir,
        "expected the tensor.load.to.lds intrinsic in the IR",
    )
    assert_true(
        "llvm.amdgcn.tensor.store.from.lds" in ir,
        "expected the tensor.store.from.lds intrinsic in the IR",
    )
    # Groups 2, 3 and the trailing group are zero for a 2D transfer.
    assert_true(
        "<4 x i32> zeroinitializer, <4 x i32> zeroinitializer, "
        "<8 x i32> zeroinitializer"
        in ir,
        "expected zero descriptor groups 2, 3 and 4 in the IR",
    )


def test_tdm_instructions_in_asm() raises:
    var asm = _compile_code[
        _tdm_kernel, target=MI455X_TARGET, emission_kind="asm"
    ]().asm
    var row = ""
    for line in asm.splitlines():
        if "tensor_load_to_lds" in line:
            row = line
    assert_true(row != "", "expected tensor_load_to_lds in the ASM")
    # A 4-word group 0 and an 8-word group 1, both from the SGPR file.
    assert_true(
        "tensor_load_to_lds s[" in row,
        String(t"expected a scalar-register descriptor, got: {row}"),
    )
    assert_true(
        "v[" not in row,
        String(t"descriptor must not use VGPRs, got: {row}"),
    )
    assert_true(
        "tensor_store_from_lds s[" in asm,
        "expected a scalar-register descriptor for tensor_store_from_lds",
    )
    # A uniform descriptor must not be uniformized by hand.
    _assert_no_line_containing(asm, "v_readfirstlane", "asm")
    _assert_no_line_containing(asm, "s_and_saveexec", "asm")
    _assert_register_budget(asm, "asm", "tensor_load_to_lds")


def test_tdm_cache_policy() raises:
    var asm = _compile_code[
        _policied_kernel, target=MI455X_TARGET, emission_kind="asm"
    ]().asm
    # 17 is scope 2 (device) and temporal hint 1 (non-temporal load), which is
    # the 5-bit split: bits 0-2 the temporal hint, bits 3-4 the scope.
    assert_true(
        "tensor_load_to_lds" in asm
        and "th:TH_LOAD_NT" in asm
        and "scope:SCOPE_DEV" in asm,
        "expected cache_policy=17 to render as th:TH_LOAD_NT scope:SCOPE_DEV",
    )
    _assert_register_budget(asm, "cache_policy=17", "tensor_load_to_lds")


def main() raises:
    test_tdm_intrinsics_in_ir()
    test_tdm_instructions_in_asm()
    test_tdm_cache_policy()
    print("tdm instructions: ok")
