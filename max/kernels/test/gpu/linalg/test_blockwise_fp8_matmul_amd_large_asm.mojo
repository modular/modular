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
"""Compile-only register report for the large-M blockwise FP8 matmul.

Compiles `blockwise_scaled_fp8_matmul_amd_large_kernel` for gfx950 at the
GLM-5.3 shapes and asserts no register spills and no scratch traffic: on
gfx9 a spill reload's `vmcnt(0)` drains every load in flight, which would
silently wreck the kernel's one-slot-ahead streaming. Prints the
instruction mix; set `BLOCKWISE_LARGE_ASM_DIR` to also write the assembly.

Also compiles, for the host, a default-configuration
`blockwise_scaled_fp8_matmul_amd` call at M = 2048 and asserts it launches
the large-M kernel: the tiled kernel computes the same numbers, so no
correctness test notices the dispatch losing it.
"""

from std.compile import compile_info
from std.os import getenv
from std.pathlib import Path

from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.compile import _compile_code
from layout import TileTensor, row_major
from std.testing import assert_true

from linalg.matmul.gpu.amd.blockwise_scaled_fp8_matmul_amd import (
    blockwise_scaled_fp8_matmul_amd,
)
from linalg.matmul.gpu.amd.blockwise_scaled_fp8_matmul_amd_large import (
    blockwise_scaled_fp8_matmul_amd_large_kernel,
)


def directive(asm: String, name: String) -> Int:
    var i = asm.find(name)
    if i < 0:
        return 0
    var end = asm.find("\n", i)
    var line = asm[byte=i:end]
    var colon = line.find(":")
    try:
        return Int(line[byte = colon + 1 :].strip())
    except:
        return -1


def check[N: Int, K: Int]() raises -> Bool:
    var asm = _compile_code[
        blockwise_scaled_fp8_matmul_amd_large_kernel[N, K],
        target=get_gpu_target["gfx950"](),
    ]().asm
    var mfma = 0
    var pk_fma = 0
    var vmem = 0
    var lds_dma = 0
    var ds_read = 0
    var vmcnt0 = 0
    var scratch = 0
    for line in asm.split("\n"):
        var t = line.strip()
        if t.startswith("v_mfma"):
            mfma += 1
        elif t.startswith("v_pk_fma_f32"):
            pk_fma += 1
        elif t.startswith("buffer_load") and "lds" in t:
            lds_dma += 1
        elif t.startswith("buffer_load"):
            vmem += 1
        elif t.startswith("ds_read"):
            ds_read += 1
        elif t.startswith("s_waitcnt") and "vmcnt(0)" in t:
            vmcnt0 += 1
        if t.startswith("scratch_"):
            scratch += 1
    var vspill = directive(asm, ".vgpr_spill_count")
    var sspill = directive(asm, ".sgpr_spill_count")
    print(
        t"large N={N} K={K}: vgpr={directive(asm, '.vgpr_count')}"
        t" agpr={directive(asm, '.agpr_count')}"
        t" sgpr={directive(asm, '.sgpr_count')} vspill={vspill}"
        t" sspill={sspill} scratch_ops={scratch} mfma={mfma}"
        t" v_pk_fma={pk_fma} buffer_load={vmem} lds_dma={lds_dma}"
        t" ds_read={ds_read} vmcnt0={vmcnt0}"
    )
    var out = getenv("BLOCKWISE_LARGE_ASM_DIR")
    if out:
        Path(out + "/large_N" + String(N) + "_K" + String(K) + ".s").write_text(
            asm
        )
    return vspill == 0 and sspill == 0 and scratch == 0


def dispatch_m2048(
    ctx: DeviceContext, c: Int, a: Int, a_scales: Int, b: Int, b_scales: Int
) raises:
    """A default-configuration call at M = N = K = 2048; compiled only."""
    comptime F8 = Scalar[DType.float8_e4m3fn]
    blockwise_scaled_fp8_matmul_amd[transpose_b=True](
        TileTensor(
            Pointer[BFloat16, MutAnyOrigin](unsafe_from_address=c),
            row_major[2048, 2048](),
        ),
        TileTensor(
            Pointer[F8, ImmutAnyOrigin](unsafe_from_address=a),
            row_major[2048, 2048](),
        ),
        TileTensor(
            Pointer[F8, ImmutAnyOrigin](unsafe_from_address=b),
            row_major[2048, 2048](),
        ),
        TileTensor(
            Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=a_scales),
            row_major[16, 2048](),
        ),
        TileTensor(
            Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=b_scales),
            row_major[16, 16](),
        ),
        ctx,
    )


def main() raises:
    var host_ir = compile_info[dispatch_m2048, emission_kind="llvm"]().asm
    assert_true(
        "blockwise_scaled_fp8_matmul_amd_large_N2048_K2048" in host_ir,
        "the default dispatch at M = 2048 does not launch the large-M kernel",
    )
    var ok = True
    ok = check[2624, 6144]() and ok
    ok = check[2048, 2048]() and ok
    ok = check[6144, 2048]() and ok
    ok = check[4096, 2048]() and ok
    ok = check[128, 6144]() and ok
    ok = check[24576, 6144]() and ok
    ok = check[6144, 12288]() and ok
    assert_true(ok, "register spills or scratch traffic")
    print("PASSED")
