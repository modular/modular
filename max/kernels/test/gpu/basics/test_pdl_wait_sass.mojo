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
"""Checks in SASS that read-only loads stay below the PDL wait (NVIDIA SM90+).

`wait_on_dependent_grids` issues `griddepcontrol.wait` from a function that is
not inlined, because `ptxas` moves `ld.global.nc` loads (`LDG.E.CONSTANT`)
above an inlined wait (`ACQBULK`) but not above a call. That is observed
`ptxas` behaviour, not a documented guarantee, so this test dumps the SASS of
PDL consumers and requires, in each kernel with a wait:

1. the wait is a `CALL` whose callee reaches `ACQBULK` before it returns, so
   calls `ptxas` inserts itself (the `tcgen05` guardrail traps) do not count;
2. no `LDG*.CONSTANT` of any width comes before that call.

Which loads `ptxas` hoists depends on its version, so the SASS comes from the
runtime's own PTX compiler (the bundled `libNVPTX.so`, at the options the
runtime passes it) and the `nvdisasm` release that matches it, both from the
test's runfiles, rather than from `/usr/local/cuda` as `_to_sass` and
`DUMP_GPU_SASS` do. A missing tool or an empty dump fails the test. Nothing is
launched.
"""

from std.ffi import OwnedDLHandle
from std.os import getenv
from std.pathlib import Path
from std.subprocess import run
from std.tempfile import TemporaryDirectory
from std.testing import TestSuite, assert_equal, assert_true

from layout import Idx, coord, row_major
from layout.tma_async import _default_desc_shape
from linalg.grouped_matmul_sm100 import (
    blackwell_tma_umma_warp_specialized_kernel,
)
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import PDLLevel, wait_on_dependent_grids

comptime ROW = 256


def consumer(
    table: ImmPointer[Int32, ImmutAnyOrigin],
    payload: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutUntrackedOrigin],
):
    """Reads a table and the row it names through immutable pointers."""
    wait_on_dependent_grids()
    var slot = Int(table.unsafe_offset(Int(block_idx.x)).unsafe_load())
    var x = payload.unsafe_offset(slot * ROW + Int(thread_idx.x)).unsafe_load()
    var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
    output.unsafe_offset(gid).unsafe_store(x)


def canary(
    table: ImmPointer[Int32, ImmutAnyOrigin],
    output: MutPointer[Int32, MutUntrackedOrigin],
):
    """Reads the table through an immutable pointer before the wait."""
    # The store keeps the load above the call: LLVM may sink an unused
    # immutable load below it.
    var slot = table.unsafe_offset(Int(block_idx.x)).unsafe_load()
    output.unsafe_offset(Int(block_idx.x)).unsafe_store(slot)
    wait_on_dependent_grids()
    output.unsafe_offset(Int(block_idx.x) + 1).unsafe_store(0)


# The instantiation `grouped_matmul` launches for four BF16 experts of N=2880
# and K=512, as in `test_grouped_matmul`, but with PDL on: the parameters
# `_grouped_matmul_sm100_persistent` derives for that shape (one CTA per MMA,
# operands swapped, stage counts sized to the B200 shared memory). With PDL off
# it has the same kernel name and hash as that test's kernel.
comptime GG_TYPE = DType.bfloat16
comptime GG_EXPERTS = 4
comptime GG_N = 2880
comptime GG_K = 512
comptime GG_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime GG_C_SWIZZLE = TensorMapSwizzle.SWIZZLE_32B
comptime GG_A_TILE = coord[128, 64]
comptime GG_B_TILE = coord[32, 64]
comptime GG_C_TILE = coord[32, 16]

comptime grouped_matmul_kernel = blackwell_tma_umma_warp_specialized_kernel[
    GG_TYPE,
    GG_TYPE,
    GG_TYPE,
    GG_N,
    GG_A_TILE,
    _default_desc_shape[GG_TYPE, GG_A_TILE, GG_SWIZZLE](),
    GG_B_TILE,
    _default_desc_shape[GG_TYPE, GG_B_TILE, GG_SWIZZLE](),
    GG_C_TILE,
    _default_desc_shape[GG_TYPE, GG_C_TILE, GG_C_SWIZZLE](),
    (128, 32, 64),
    (128, 32, 16),
    cluster_shape=(1, 1, 1),
    num_pipeline_stages=10,
    num_accum_pipeline_stages=16,
    output_tile_shape=coord[32, 128],
    a_swizzle=GG_SWIZZLE,
    b_swizzle=GG_SWIZZLE,
    c_swizzle=GG_C_SWIZZLE,
    cta_group=1,
    transpose_c=True,
    K_actual=GG_K,
    a_gmem_layout=type_of(row_major[GG_EXPERTS * GG_N, GG_K]()),
    b_gmem_layout=type_of(row_major(0, Idx[GG_K])),
    pdl_level=PDLLevel.ON,
]


def ptx[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    arch: StaticString,
]() raises -> String:
    var asm = String(_compile_code[func, target=get_gpu_target[arch]()]().asm)
    assert_true("griddepcontrol.wait" in asm, "the PTX has no PDL wait")
    return asm^


def _runfile(env_var: StaticString, basename: StaticString) raises -> Path:
    """Returns the file named `basename` under the paths in `env_var`."""
    var paths = getenv(env_var)
    assert_true(paths.byte_length() > 0, String("$", env_var, " is not set"))
    var found = run(
        String("find -L ", paths, " -name ", basename, " -type f 2>/dev/null")
    )
    for line in found.splitlines():
        var path = Path(String(line))
        if path.is_file():
            return path^
    raise Error("no ", basename, " under $", env_var, ": ", paths)


def _sass(ptx: String) raises -> String:
    """Compiles PTX as the runtime does and disassembles it."""
    var arch = String(ptx.split(".target")[1].split()[0])
    var lib = OwnedDLHandle(_runfile("NVPTX_LIBRARY", "libNVPTX.so"))
    var handle = 0
    assert_equal(
        lib.call["nvPTXCompilerCreate", Int32](
            Pointer(to=handle), ptx.byte_length(), ptx.unsafe_ptr()
        ),
        0,
    )
    var gpu_name = String("--gpu-name=", arch)
    var opt_level = String("--opt-level=4")
    var opts: List[Int] = [
        Int(gpu_name.as_c_string_span().ptr()),
        Int(opt_level.as_c_string_span().ptr()),
    ]
    var status = lib.call["nvPTXCompilerCompile", Int32](
        handle, Int32(2), opts.unsafe_ptr()
    )
    _ = gpu_name^
    _ = opt_level^
    var size = 0
    if status != 0:
        _ = lib.call["nvPTXCompilerGetErrorLogSize", Int32](
            handle, Pointer(to=size)
        )
        var log = List[Byte](length=size + 1, fill=0)
        _ = lib.call["nvPTXCompilerGetErrorLog", Int32](
            handle, log.unsafe_ptr()
        )
        raise Error(
            "nvPTXCompiler failed (",
            status,
            "): ",
            StringSlice(unsafe_from_utf8=Span(log)[:size]),
        )
    _ = lib.call["nvPTXCompilerGetCompiledProgramSize", Int32](
        handle, Pointer(to=size)
    )
    var cubin = List[Byte](length=size, fill=0)
    _ = lib.call["nvPTXCompilerGetCompiledProgram", Int32](
        handle, cubin.unsafe_ptr()
    )
    _ = lib.call["nvPTXCompilerDestroy", Int32](Pointer(to=handle))
    _ = ptx

    var nvdisasm = _runfile("NVDISASM", "nvdisasm")
    var sass = String()
    with TemporaryDirectory() as tmp:
        var elf = Path(tmp) / "kernel.cubin"
        elf.write_bytes(cubin)
        sass = run(String(nvdisasm, " -ndf -c ", elf))
    assert_true(".text." in sass, String("empty SASS dump: ", sass))
    return sass^


@fieldwise_init
struct Kernel(Copyable, Movable):
    """One entry point's SASS, with the local labels of its callees."""

    var name: String
    var ops: List[String]
    var offsets: List[String]
    var call_targets: List[String]
    var labels: Dict[String, Int]


def _kernels(sass: String) -> List[Kernel]:
    var kernels = List[Kernel]()
    for raw in sass.splitlines():
        var line = String(raw)
        if ".section" in line and ".text." in line:
            var name = String(line.split(".text.")[1].split(",")[0])
            kernels.append(Kernel(name, [], [], [], {}))
            continue
        if len(kernels) == 0:
            continue
        ref kernel = kernels[len(kernels) - 1]
        if not line.startswith(" ") and line.endswith(":"):
            kernel.labels[String(line.removesuffix(":"))] = len(kernel.ops)
        elif "/*" in line and "*/" in line:
            var halves = line.split("*/", 1)
            var tokens = halves[1].split()
            if len(tokens) == 0:
                continue
            var op = String(
                tokens[1] if tokens[0].startswith("@") else tokens[0]
            )
            var target = String()
            if "`(" in halves[1]:
                target = String(halves[1].split("`(")[1].split(")")[0])
            kernel.ops.append(String(op.removesuffix(";").strip()))
            kernel.offsets.append(String(halves[0].split("/*")[1]))
            kernel.call_targets.append(target)
    return kernels^


def _reaches_acqbulk(kernel: Kernel, label: String) -> Bool:
    var start = kernel.labels.get(label)
    if not start:
        return False
    for i in range(start.value(), len(kernel.ops)):
        if kernel.ops[i] == "ACQBULK":
            return True
        if kernel.ops[i].startswith("RET") or kernel.ops[i].startswith("EXIT"):
            return False
    return False


def check(kernel: Kernel) -> Optional[String]:
    """Returns None without a wait, "" if the wait holds, else the failure."""
    if "ACQBULK" not in kernel.ops:
        return None
    var loads = String()
    for i in range(len(kernel.ops)):
        ref op = kernel.ops[i]
        if op == "ACQBULK":
            return String(
                "the wait is inline (ACQBULK at 0x",
                kernel.offsets[i],
                "), not a call; read-only loads above it:",
                loads if loads else " none",
            )
        if op.startswith("CALL") and _reaches_acqbulk(
            kernel, kernel.call_targets[i]
        ):
            if loads:
                return String(
                    "read-only load before the wait call at 0x",
                    kernel.offsets[i],
                    ":",
                    loads,
                )
            return String()
        if op.startswith("LDG") and ".CONSTANT" in op:
            loads += String(" 0x", kernel.offsets[i], " ", op)
    return String("ACQBULK is reachable from no call")


def gate(ptx: String) raises -> String:
    """Returns the one kernel with a wait in `ptx`'s `check` result."""
    var results = List[String]()
    for kernel in _kernels(_sass(ptx)):
        var result = check(kernel)
        if result:
            print(kernel.name, ":", result.value() or "ok")
            results.append(result.value())
    assert_equal(len(results), 1, "expected one kernel with a wait")
    return results[0]


def inline_wait(ptx: String) raises -> String:
    """Replaces every call to the wait function with the wait itself."""
    for piece in ptx.split(".func ")[1:]:
        if "griddepcontrol.wait;" in piece.split("}")[0]:
            var call = String("call.uni ", piece.split("(")[0], ", ();")
            assert_true(call in ptx, "the wait function is never called")
            return ptx.replace(call, "griddepcontrol.wait;")
    raise Error("no function in the PTX issues griddepcontrol.wait")


def test_consumer_sm90() raises:
    assert_equal(gate(ptx[consumer, "sm_90a"]()), "")


def test_consumer_sm100() raises:
    assert_equal(gate(ptx[consumer, "sm_100a"]()), "")


def test_grouped_matmul_sm100() raises:
    assert_equal(gate(ptx[grouped_matmul_kernel, "sm_100a"]()), "")


def test_gate_rejects_an_inlined_wait() raises:
    var c = gate(inline_wait(ptx[consumer, "sm_100a"]()))
    assert_true(c.startswith("the wait is inline"), c)
    var g = gate(inline_wait(ptx[grouped_matmul_kernel, "sm_100a"]()))
    assert_true(g.startswith("the wait is inline"), g)


def test_gate_reports_a_readonly_load_above_the_call() raises:
    comptime for arch in ["sm_90a", "sm_100a"]:
        var r = gate(ptx[canary, arch]())
        assert_true(r.startswith("read-only load before the wait call"), r)


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle the kernels
    # defined in this module.
    var suite = TestSuite()
    suite.test[test_consumer_sm90]()
    suite.test[test_consumer_sm100]()
    suite.test[test_grouped_matmul_sm100]()
    suite.test[test_gate_rejects_an_inlined_wait]()
    suite.test[test_gate_reports_a_readonly_load_above_the_call]()
    suite^.run()
