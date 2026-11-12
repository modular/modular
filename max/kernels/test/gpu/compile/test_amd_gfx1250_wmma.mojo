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

"""Checks every CDNA5 (gfx1250) WMMA combination the generic entry point takes.

Each combination `wmma_cdna` supports is compiled for the MI455X target and
must lower to exactly one `v_wmma` instruction of the expected shape, named by
both the LLVM intrinsic and the assembly mnemonic. The combinations chosen for
it also verify that the A/B reuse hints reach the instruction as the
`matrix_a_reuse` and `matrix_b_reuse` modifiers, and the generic `mma` must pick
the same instruction it did before the hints became a parameter. Each kernel is
also held to the library's register budget: zero VGPR and SGPR spills, an empty
private segment, and no `scratch_*` load or store in the kernel's own body.

The same is done for the block-scaled `wmma_scale_cdna`: one `v_wmma_scale`
(or `v_wmma_scale16`) of the expected form for every data format, scale format
and shape it takes, with the data-format, scale-format and scale-select
modifiers read back from the assembly verbatim. Its fp6 and bf6 formats get no
case: their 12-register tuple is not a SIMD length, so the entry point rejects
them at compile time. The containers a fragment can arrive in are checked too:
int32 bit lanes, the format's own element dtype, and 4-bit data packed into
`_uint4` lanes.
"""

from max.gpu.compute.arch.mma_amd import wmma_cdna, wmma_scale_cdna
from max.gpu.compute.mma import mma
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.sys import align_of, get_defined_string
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

comptime FP8 = DType.float8_e4m3fn
comptime BF8 = DType.float8_e5m2
comptime F16 = DType.float16
comptime BF16 = DType.bfloat16
comptime F32 = DType.float32


@fieldwise_init
struct _WMMA_Case(Copyable, Movable):
    """One dtype and lane-width combination of the CDNA5 WMMA dispatch.

    The expected intrinsic and mnemonic are spelled out rather than derived
    from the dtypes, so a mistake in the dispatch's name construction fails the
    test instead of being repeated by it.
    """

    var a_dtype: DType
    var b_dtype: DType
    var c_dtype: DType
    var d_dtype: DType
    var a_len: Int
    var b_len: Int
    var c_len: Int
    var d_len: Int
    var intrinsic: StaticString
    var opcode: StaticString
    var checks_reuse: Bool
    """Whether this case also compiles with every A/B reuse combination."""


comptime CASES: List[_WMMA_Case] = [
    # F16 = [F8 or BF8] * [F8 or BF8] + F16, 16x16x64, four A/B pairings.
    _WMMA_Case(
        FP8,
        FP8,
        F16,
        F16,
        32,
        32,
        8,
        8,
        "llvm.amdgcn.wmma.f16.16x16x64.fp8.fp8",
        "v_wmma_f16_16x16x64_fp8_fp8",
        True,
    ),
    _WMMA_Case(
        FP8,
        BF8,
        F16,
        F16,
        32,
        32,
        8,
        8,
        "llvm.amdgcn.wmma.f16.16x16x64.fp8.bf8",
        "v_wmma_f16_16x16x64_fp8_bf8",
        False,
    ),
    _WMMA_Case(
        BF8,
        FP8,
        F16,
        F16,
        32,
        32,
        8,
        8,
        "llvm.amdgcn.wmma.f16.16x16x64.bf8.fp8",
        "v_wmma_f16_16x16x64_bf8_fp8",
        False,
    ),
    _WMMA_Case(
        BF8,
        BF8,
        F16,
        F16,
        32,
        32,
        8,
        8,
        "llvm.amdgcn.wmma.f16.16x16x64.bf8.bf8",
        "v_wmma_f16_16x16x64_bf8_bf8",
        False,
    ),
    # F32 = [F8 or BF8] * [F8 or BF8] + F32, 16x16x64, four A/B pairings.
    _WMMA_Case(
        FP8,
        FP8,
        F32,
        F32,
        32,
        32,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8",
        "v_wmma_f32_16x16x64_fp8_fp8",
        False,
    ),
    _WMMA_Case(
        FP8,
        BF8,
        F32,
        F32,
        32,
        32,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8",
        "v_wmma_f32_16x16x64_fp8_bf8",
        False,
    ),
    _WMMA_Case(
        BF8,
        FP8,
        F32,
        F32,
        32,
        32,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8",
        "v_wmma_f32_16x16x64_bf8_fp8",
        False,
    ),
    _WMMA_Case(
        BF8,
        BF8,
        F32,
        F32,
        32,
        32,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8",
        "v_wmma_f32_16x16x64_bf8_bf8",
        False,
    ),
    # F16 = [F8 or BF8] * [F8 or BF8] + F16, 16x16x128, four A/B pairings.
    _WMMA_Case(
        FP8,
        FP8,
        F16,
        F16,
        64,
        64,
        8,
        8,
        "llvm.amdgcn.wmma.f16.16x16x128.fp8.fp8",
        "v_wmma_f16_16x16x128_fp8_fp8",
        True,
    ),
    _WMMA_Case(
        FP8,
        BF8,
        F16,
        F16,
        64,
        64,
        8,
        8,
        "llvm.amdgcn.wmma.f16.16x16x128.fp8.bf8",
        "v_wmma_f16_16x16x128_fp8_bf8",
        False,
    ),
    _WMMA_Case(
        BF8,
        FP8,
        F16,
        F16,
        64,
        64,
        8,
        8,
        "llvm.amdgcn.wmma.f16.16x16x128.bf8.fp8",
        "v_wmma_f16_16x16x128_bf8_fp8",
        False,
    ),
    _WMMA_Case(
        BF8,
        BF8,
        F16,
        F16,
        64,
        64,
        8,
        8,
        "llvm.amdgcn.wmma.f16.16x16x128.bf8.bf8",
        "v_wmma_f16_16x16x128_bf8_bf8",
        False,
    ),
    # F32 = [F8 or BF8] * [F8 or BF8] + F32, 16x16x128, four A/B pairings.
    _WMMA_Case(
        FP8,
        FP8,
        F32,
        F32,
        64,
        64,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x128.fp8.fp8",
        "v_wmma_f32_16x16x128_fp8_fp8",
        False,
    ),
    _WMMA_Case(
        FP8,
        BF8,
        F32,
        F32,
        64,
        64,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x128.fp8.bf8",
        "v_wmma_f32_16x16x128_fp8_bf8",
        False,
    ),
    _WMMA_Case(
        BF8,
        FP8,
        F32,
        F32,
        64,
        64,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x128.bf8.fp8",
        "v_wmma_f32_16x16x128_bf8_fp8",
        False,
    ),
    _WMMA_Case(
        BF8,
        BF8,
        F32,
        F32,
        64,
        64,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x128.bf8.bf8",
        "v_wmma_f32_16x16x128_bf8_bf8",
        False,
    ),
    # F16 = F16 * F16 + F16 and F32 = F16 * F16 + F32, 16x16x32.
    _WMMA_Case(
        F16,
        F16,
        F16,
        F16,
        16,
        16,
        8,
        8,
        "llvm.amdgcn.wmma.f16.16x16x32.f16",
        "v_wmma_f16_16x16x32_f16",
        True,
    ),
    _WMMA_Case(
        F16,
        F16,
        F32,
        F32,
        16,
        16,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x32.f16",
        "v_wmma_f32_16x16x32_f16",
        False,
    ),
    # BF16 = BF16 * BF16 + BF16 and F32 = BF16 * BF16 + F32, 16x16x32.
    _WMMA_Case(
        BF16,
        BF16,
        BF16,
        BF16,
        16,
        16,
        8,
        8,
        "llvm.amdgcn.wmma.bf16.16x16x32.bf16",
        "v_wmma_bf16_16x16x32_bf16",
        False,
    ),
    _WMMA_Case(
        BF16,
        BF16,
        F32,
        F32,
        16,
        16,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x32.bf16",
        "v_wmma_f32_16x16x32_bf16",
        False,
    ),
    # BF16 = BF16 * BF16 + F32, the bf16 result of an f32 accumulator.
    _WMMA_Case(
        BF16,
        BF16,
        F32,
        BF16,
        16,
        16,
        8,
        8,
        "llvm.amdgcn.wmma.bf16f32.16x16x32.bf16",
        "v_wmma_bf16f32_16x16x32_bf16",
        True,
    ),
    # F32 = F32 * F32 + F32, 16x16x4.
    _WMMA_Case(
        F32,
        F32,
        F32,
        F32,
        2,
        2,
        8,
        8,
        "llvm.amdgcn.wmma.f32.16x16x4.f32",
        "v_wmma_f32_16x16x4_f32",
        True,
    ),
]
"""Every dtype and lane-width combination `wmma_cdna` dispatches on gfx1250."""

comptime REUSE_COMBINATIONS: List[Tuple[Bool, Bool]] = [
    (True, False),
    (False, True),
    (True, True),
]
"""The reuse settings a case marked for it compiles on top of the default one.

The all-clear setting is the base compile of every case."""


def _wmma_kernel[
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    d_dtype: DType,
    a_len: Int,
    b_len: Int,
    c_len: Int,
    d_len: Int,
    a_reuse: Bool,
    b_reuse: Bool,
](
    a_src: ImmPointer[Scalar[a_dtype], ImmUntrackedOrigin],
    b_src: ImmPointer[Scalar[b_dtype], ImmUntrackedOrigin],
    c_src: ImmPointer[Scalar[c_dtype], ImmUntrackedOrigin],
    d_dst: MutPointer[Scalar[d_dtype], MutUntrackedOrigin],
):
    # The fragments come from memory so the WMMA cannot be constant-folded out
    # of the compiled kernel.
    var a = a_src.unsafe_load[
        width=a_len, alignment=align_of[SIMD[a_dtype, a_len]]()
    ]()
    var b = b_src.unsafe_load[
        width=b_len, alignment=align_of[SIMD[b_dtype, b_len]]()
    ]()
    var c = c_src.unsafe_load[
        width=c_len, alignment=align_of[SIMD[c_dtype, c_len]]()
    ]()
    var d = SIMD[d_dtype, d_len](0)
    wmma_cdna[a_reuse=a_reuse, b_reuse=b_reuse](d, a, b, c)
    d_dst.unsafe_store[width=d_len](d)


def _mma_kernel(
    src: ImmPointer[Scalar[DType.bfloat16], ImmUntrackedOrigin],
    dst: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
):
    var a = src.unsafe_load[
        width=16, alignment=align_of[SIMD[DType.bfloat16, 16]]()
    ]()
    var b = src.unsafe_load[
        width=16, alignment=align_of[SIMD[DType.bfloat16, 16]]()
    ](16)
    var c = dst.unsafe_load[
        width=8, alignment=align_of[SIMD[DType.float32, 8]]()
    ]()
    var d = SIMD[DType.float32, 8](0.0)
    mma(d, a, b, c)
    dst.unsafe_store(d[0])


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


def _check_combo[
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    d_dtype: DType,
    a_len: Int,
    b_len: Int,
    c_len: Int,
    d_len: Int,
    intrinsic_name: StaticString,
    opcode: StaticString,
    a_reuse: Bool,
    b_reuse: Bool,
    check_reuse: Bool,
]() raises:
    """Compiles one combination once per emission kind and checks both.

    The reuse modifiers show up in the assembly that already carries the
    opcode, so a third compile for them alone would assert nothing new.
    """
    comptime kernel = _wmma_kernel[
        a_dtype,
        b_dtype,
        c_dtype,
        d_dtype,
        a_len,
        b_len,
        c_len,
        d_len,
        a_reuse,
        b_reuse,
    ]
    var label = String(t"{opcode} with a_reuse={a_reuse}, b_reuse={b_reuse}")

    var ir = _compile_code[
        kernel, target=MI455X_TARGET, emission_kind="llvm"
    ]().asm
    assert_true(
        intrinsic_name in ir,
        String(t"expected {intrinsic_name} for {label}"),
    )

    var asm = _compile_code[
        kernel, target=MI455X_TARGET, emission_kind="asm"
    ]().asm
    var count = asm.count(opcode)
    assert_true(
        count == 1,
        String(t"expected exactly 1 {opcode} but found {count} for {label}"),
    )
    if check_reuse:
        assert_true(
            ("matrix_a_reuse" in asm) == a_reuse,
            String(t"expected matrix_a_reuse exactly when a_reuse for {label}"),
        )
        assert_true(
            ("matrix_b_reuse" in asm) == b_reuse,
            String(t"expected matrix_b_reuse exactly when b_reuse for {label}"),
        )
    _assert_register_budget(asm, label, String(opcode))


comptime FMT_FP8 = DType.float8_e4m3fn
comptime FMT_BF8 = DType.float8_e5m2
comptime FMT_FP4 = DType.float4_e2m1fn
comptime E8 = DType.float8_e8m0fnu
comptime E4M3 = DType.float8_e4m3fn


@fieldwise_init
struct _WMMA_Scale_Case(Copyable, Movable):
    """One data-format, scale-format and scale-select setting of a form.

    The modifiers are spelled out as `llvm-mc` prints them, and it does not
    print a modifier whose value is the encoding's zero, so the all-default
    MXFP8 case expects no modifiers at all.
    """

    var a_type: DType
    var b_type: DType
    var scale_type: DType
    var a_scale_sel: Int
    var b_scale_sel: Int
    var modifiers: StaticString
    """The modifiers after the operand list, verbatim."""
    var checks_reuse: Bool
    """Whether this case also compiles with every A/B reuse combination."""


comptime F8F6F4_CASES: List[_WMMA_Scale_Case] = [
    # MXFP8 and MXFP4 use e8 scales, and every pair of the formats a SIMD can
    # hold is valid with them. The modifier is absent where the format's code
    # is 0 (fp8), and the printer never omits a non-zero one. fp6 and bf6 are
    # missing because their 12-register tuple cannot be a SIMD length, so the
    # entry point rejects those two formats instead.
    _WMMA_Scale_Case(FMT_FP8, FMT_FP8, E8, 0, 0, "", True),
    _WMMA_Scale_Case(
        FMT_FP8, FMT_BF8, E8, 0, 0, "matrix_b_fmt:MATRIX_FMT_BF8", False
    ),
    _WMMA_Scale_Case(
        FMT_FP8, FMT_FP4, E8, 0, 0, "matrix_b_fmt:MATRIX_FMT_FP4", False
    ),
    _WMMA_Scale_Case(
        FMT_BF8, FMT_FP8, E8, 0, 0, "matrix_a_fmt:MATRIX_FMT_BF8", False
    ),
    _WMMA_Scale_Case(
        FMT_BF8,
        FMT_BF8,
        E8,
        0,
        0,
        "matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_BF8",
        False,
    ),
    _WMMA_Scale_Case(
        FMT_BF8,
        FMT_FP4,
        E8,
        0,
        0,
        "matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP4",
        False,
    ),
    _WMMA_Scale_Case(
        FMT_FP4, FMT_FP8, E8, 0, 0, "matrix_a_fmt:MATRIX_FMT_FP4", False
    ),
    _WMMA_Scale_Case(
        FMT_FP4,
        FMT_BF8,
        E8,
        0,
        0,
        "matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_BF8",
        False,
    ),
    _WMMA_Scale_Case(
        FMT_FP4,
        FMT_FP4,
        E8,
        0,
        0,
        "matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4",
        False,
    ),
    # An e4m3 scale is only valid when A and B are both fp4. The ISA's e5m3
    # scale format has no DType, so the entry point cannot name it.
    _WMMA_Scale_Case(
        FMT_FP4,
        FMT_FP4,
        E4M3,
        0,
        0,
        (
            "matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4"
            " matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3"
            " matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3"
        ),
        False,
    ),
    # Only a scale selector of 1 is printed; these check that A and B select
    # their rows independently of each other.
    _WMMA_Scale_Case(
        FMT_FP4,
        FMT_FP8,
        E8,
        1,
        1,
        (
            "matrix_a_fmt:MATRIX_FMT_FP4 matrix_a_scale:MATRIX_SCALE_ROW1"
            " matrix_b_scale:MATRIX_SCALE_ROW1"
        ),
        False,
    ),
    _WMMA_Scale_Case(
        FMT_FP4,
        FMT_FP8,
        E8,
        1,
        0,
        "matrix_a_fmt:MATRIX_FMT_FP4 matrix_a_scale:MATRIX_SCALE_ROW1",
        False,
    ),
]
"""The 16x16x128 f8f6f4 forms: fp8, bf8 and fp4 data formats."""


comptime F4_CASES: List[_WMMA_Scale_Case] = [
    # The 32x16x128 f4 form is fp4 on both sides and carries no data-format
    # operand, so only the scale format shows in its modifiers.
    _WMMA_Scale_Case(FMT_FP4, FMT_FP4, E8, 0, 0, "", True),
    _WMMA_Scale_Case(
        FMT_FP4,
        FMT_FP4,
        E4M3,
        0,
        0,
        (
            "matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3"
            " matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3"
        ),
        False,
    ),
    _WMMA_Scale_Case(
        FMT_FP4,
        FMT_FP4,
        E8,
        1,
        1,
        "matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1",
        False,
    ),
]
"""The 32x16x128 f4 form, which is fp4 on both sides."""


def _scale_operand_regs[d_len: Int, dtype: DType, is_a: Bool]() -> Int:
    """Returns the 32-bit registers one operand of the given shape needs.

    The 32x16x128 f4 form is fixed per operand, since both are fp4; the
    16x16x128 f8f6f4 form sizes each from its own type. The float6 types get
    no case, since the entry point rejects them.
    """
    comptime if d_len == 16:
        return 16 if is_a else 8
    elif dtype == FMT_FP4:
        return 8
    else:
        return 16


def _wmma_scale_kernel[
    a_type: DType,
    b_type: DType,
    scale_type: DType,
    scale16: Bool,
    a_len: Int,
    b_len: Int,
    d_len: Int,
    a_scale_sel: Int,
    b_scale_sel: Int,
    a_reuse: Bool,
    b_reuse: Bool,
    a_dtype: DType = DType.int32,
    b_dtype: DType = DType.int32,
](
    a_src: ImmPointer[Scalar[a_dtype], ImmUntrackedOrigin],
    b_src: ImmPointer[Scalar[b_dtype], ImmUntrackedOrigin],
    c_src: ImmPointer[Scalar[DType.float32], ImmUntrackedOrigin],
    a_scale: Scalar[DType.int64 if scale16 else DType.int32],
    b_scale: Scalar[DType.int64 if scale16 else DType.int32],
    d_dst: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
):
    # The fragments come from memory so the WMMA cannot be constant-folded out
    # of the compiled kernel. They arrive as int32 bits unless a case names a
    # typed or sub-byte container, which the entry point sizes by its width.
    var a = a_src.unsafe_load[
        width=a_len, alignment=align_of[SIMD[a_dtype, a_len]]()
    ]()
    var b = b_src.unsafe_load[
        width=b_len, alignment=align_of[SIMD[b_dtype, b_len]]()
    ]()
    var c = c_src.unsafe_load[
        width=d_len, alignment=align_of[SIMD[DType.float32, d_len]]()
    ]()
    var d = SIMD[DType.float32, d_len](0)
    wmma_scale_cdna[
        a_type=a_type,
        b_type=b_type,
        scale_type=scale_type,
        scale16=scale16,
        a_scale_sel=a_scale_sel,
        b_scale_sel=b_scale_sel,
        a_reuse=a_reuse,
        b_reuse=b_reuse,
    ](d, a, b, c, a_scale, b_scale)
    d_dst.unsafe_store[width=d_len](d)


def _modifier_tail(asm: String, opcode: StaticString) -> Optional[String]:
    """Returns the modifiers the printer wrote after `opcode`'s operands.

    Every modifier this test looks at starts with `matrix_`, and the operand
    list never does, so the first one marks where the tail starts.
    """
    for line in asm.splitlines():
        if opcode in line:
            var text = String(line).strip()
            if case .Some(at) = text.find(" matrix_"):
                return String(text[byte = at + 1 :])
            return String("")
    return None


def _check_scale_combo[
    a_type: DType,
    b_type: DType,
    scale_type: DType,
    scale16: Bool,
    d_len: Int,
    a_scale_sel: Int,
    b_scale_sel: Int,
    intrinsic_name: StaticString,
    opcode: StaticString,
    modifiers: StaticString,
    a_reuse: Bool,
    b_reuse: Bool,
    a_dtype: DType = DType.int32,
    b_dtype: DType = DType.int32,
    a_elements: Int = 0,
    b_elements: Int = 0,
]() raises:
    """Compiles one block-scaled combination and checks both emissions.

    `a_dtype` and `b_dtype` are the dtypes the fragments are held in, and
    `a_elements` and `b_elements` their lane counts. They default to int32
    bits, where one lane is one register, which is what most of the format
    grid uses.
    """
    comptime int32_lanes_a = _scale_operand_regs[d_len, a_type, True]()
    comptime int32_lanes_b = _scale_operand_regs[d_len, b_type, False]()
    comptime a_len = a_elements if a_elements > 0 else int32_lanes_a
    comptime b_len = b_elements if b_elements > 0 else int32_lanes_b
    comptime kernel = _wmma_scale_kernel[
        a_type,
        b_type,
        scale_type,
        scale16,
        a_len,
        b_len,
        d_len,
        a_scale_sel,
        b_scale_sel,
        a_reuse,
        b_reuse,
        a_dtype,
        b_dtype,
    ]
    var label = String(
        t"{opcode} with a_type {a_type}, b_type {b_type},"
        t" scale_type {scale_type}, a_scale_sel {a_scale_sel},"
        t" b_scale_sel {b_scale_sel}, a_reuse={a_reuse}, b_reuse={b_reuse},"
        t" a_dtype {a_dtype}, b_dtype {b_dtype}"
    )

    var ir = _compile_code[
        kernel, target=MI455X_TARGET, emission_kind="llvm"
    ]().asm
    assert_true(
        intrinsic_name in ir,
        String(t"expected {intrinsic_name} for {label}"),
    )

    var asm = _compile_code[
        kernel, target=MI455X_TARGET, emission_kind="asm"
    ]().asm
    var count = asm.count(opcode)
    assert_true(
        count == 1,
        String(t"expected exactly 1 {opcode} but found {count} for {label}"),
    )

    # The reuse hints come last, after the format, scale and scale-format
    # modifiers, and only where they are set. A hint appended to an empty
    # modifier list leaves the separator space behind, so strip the tail.
    var hinted = String(modifiers)
    if a_reuse:
        hinted += " matrix_a_reuse"
    if b_reuse:
        hinted += " matrix_b_reuse"
    var expected = hinted.strip()
    var found = _modifier_tail(asm, opcode).or_else("")
    assert_true(
        found == expected,
        String(
            t"expected the modifiers '{expected}' on {label}, found '{found}'"
        ),
    )
    _assert_register_budget(asm, label, String(opcode))


def _check_f8f6f4_cases[
    scale16: Bool, intrinsic_name: StaticString, opcode: StaticString
]() raises:
    """Checks the 16x16x128 f8f6f4 data formats in one scale form."""
    comptime for entry in F8F6F4_CASES:
        _check_scale_combo[
            entry.a_type,
            entry.b_type,
            entry.scale_type,
            scale16,
            8,
            entry.a_scale_sel,
            entry.b_scale_sel,
            intrinsic_name,
            opcode,
            entry.modifiers,
            False,
            False,
        ]()
        comptime if entry.checks_reuse:
            comptime for combo in REUSE_COMBINATIONS:
                _check_scale_combo[
                    entry.a_type,
                    entry.b_type,
                    entry.scale_type,
                    scale16,
                    8,
                    entry.a_scale_sel,
                    entry.b_scale_sel,
                    intrinsic_name,
                    opcode,
                    entry.modifiers,
                    combo[0],
                    combo[1],
                ]()


def _check_f4_cases[
    scale16: Bool, intrinsic_name: StaticString, opcode: StaticString
]() raises:
    """Checks the 32x16x128 f4 scale formats in one scale form."""
    comptime for entry in F4_CASES:
        _check_scale_combo[
            entry.a_type,
            entry.b_type,
            entry.scale_type,
            scale16,
            16,
            entry.a_scale_sel,
            entry.b_scale_sel,
            intrinsic_name,
            opcode,
            entry.modifiers,
            False,
            False,
        ]()
        comptime if entry.checks_reuse:
            comptime for combo in REUSE_COMBINATIONS:
                _check_scale_combo[
                    entry.a_type,
                    entry.b_type,
                    entry.scale_type,
                    scale16,
                    16,
                    entry.a_scale_sel,
                    entry.b_scale_sel,
                    intrinsic_name,
                    opcode,
                    entry.modifiers,
                    combo[0],
                    combo[1],
                ]()


def test_all_combinations() raises:
    """Verifies every supported combination is one WMMA of the expected shape
    within the register budget, and that the cases marked for it reach the
    instruction under every reuse setting."""
    comptime for entry in CASES:
        _check_combo[
            entry.a_dtype,
            entry.b_dtype,
            entry.c_dtype,
            entry.d_dtype,
            entry.a_len,
            entry.b_len,
            entry.c_len,
            entry.d_len,
            entry.intrinsic,
            entry.opcode,
            False,
            False,
            entry.checks_reuse,
        ]()
        comptime if entry.checks_reuse:
            comptime for combo in REUSE_COMBINATIONS:
                _check_combo[
                    entry.a_dtype,
                    entry.b_dtype,
                    entry.c_dtype,
                    entry.d_dtype,
                    entry.a_len,
                    entry.b_len,
                    entry.c_len,
                    entry.d_len,
                    entry.intrinsic,
                    entry.opcode,
                    combo[0],
                    combo[1],
                    True,
                ]()


def test_scale_combinations() raises:
    """Verifies every valid block-scaled format pair is one `v_wmma_scale` of
    the expected form, with the data-format, scale-format and scale-select
    modifiers reaching the instruction as `llvm-mc` prints them.

    Each shape is checked in both the one-register `scale` and two-register
    `scale16` forms, and one case per form also compiles under every reuse
    setting.
    """
    _check_f8f6f4_cases[
        False,
        "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4",
        "v_wmma_scale_f32_16x16x128_f8f6f4",
    ]()
    _check_f8f6f4_cases[
        True,
        "llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4",
        "v_wmma_scale16_f32_16x16x128_f8f6f4",
    ]()
    _check_f4_cases[
        False,
        "llvm.amdgcn.wmma.scale.f32.32x16x128.f4",
        "v_wmma_scale_f32_32x16x128_f4",
    ]()
    _check_f4_cases[
        True,
        "llvm.amdgcn.wmma.scale16.f32.32x16x128.f4",
        "v_wmma_scale16_f32_32x16x128_f4",
    ]()


def test_scale_operand_dtypes() raises:
    """Verifies the format check takes the two ways a caller can hold a
    fragment besides int32 bits: the format's own element dtype, and a
    sub-byte dtype whose lanes pack into the same registers.

    A typed fragment only decodes as the format its bits are encoded in, and
    a packed one is sized by its whole vector's bit width, which is what makes
    64 `_uint4` lanes the 8 registers fp4 asks for.
    """
    comptime INTRINSIC_NAME = "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"
    comptime ASM_OPCODE = "v_wmma_scale_f32_16x16x128_f8f6f4"

    # 64 float8_e5m2 elements are the 16 registers bf8 decodes, so a fragment
    # kept in its element dtype passes the check.
    _check_scale_combo[
        FMT_BF8,
        FMT_BF8,
        E8,
        False,
        8,
        0,
        0,
        INTRINSIC_NAME,
        ASM_OPCODE,
        "matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_BF8",
        False,
        False,
        DType.float8_e5m2,
        DType.float8_e5m2,
        64,
        64,
    ]()

    # 64 `_uint4` lanes are also 256 bits, the 8 registers fp4 decodes, so
    # fp4 data packed in 4-bit lanes is accepted rather than counted as a byte
    # per element.
    _check_scale_combo[
        FMT_FP4,
        FMT_FP4,
        E8,
        False,
        8,
        0,
        0,
        INTRINSIC_NAME,
        ASM_OPCODE,
        "matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4",
        False,
        False,
        DType._uint4,
        DType._uint4,
        64,
        64,
    ]()


def test_generic_mma_lowers_to_wmma() raises:
    """Verifies the generic `mma` picks the same instruction for bf16
    16x16x32 inputs and an f32 8-wide accumulator, and meets the register
    budget."""
    comptime INTRINSIC_NAME = "llvm.amdgcn.wmma.f32.16x16x32.bf16"
    comptime ASM_OPCODE = "v_wmma_f32_16x16x32_bf16"

    var ir = _compile_code[
        _mma_kernel, target=MI455X_TARGET, emission_kind="llvm"
    ]().asm
    assert_true(
        INTRINSIC_NAME in ir,
        String(t"expected the generic mma to emit {INTRINSIC_NAME}"),
    )

    var asm = _compile_code[
        _mma_kernel, target=MI455X_TARGET, emission_kind="asm"
    ]().asm
    assert_true(
        asm.count(ASM_OPCODE) == 1,
        String(t"expected exactly 1 {ASM_OPCODE} from the generic mma"),
    )
    assert_true(
        "matrix_a_reuse" not in asm and "matrix_b_reuse" not in asm,
        String(t"expected the generic mma to leave the reuse hints unset"),
    )
    _assert_register_budget(asm, "generic mma", String(ASM_OPCODE))


def main() raises:
    test_all_combinations()
    test_scale_combinations()
    test_scale_operand_dtypes()
    test_generic_mma_lowers_to_wmma()
