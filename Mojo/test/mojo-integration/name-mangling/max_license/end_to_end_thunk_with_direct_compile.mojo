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
#
# Regression test for MOCO-3842: when the same user kernel is offloaded both
# directly (via `enqueue_function[func]`'s non-parametric
# offload path, which stamps `kgen.offload.kernelid` onto the user kernel
# itself) AND through an `enqueue_function` conversion thunk
# (the parametric path that wraps the user kernel and inherits its LLVM
# metadata onto a wrapper), two invariants must hold:
#
#   1. Compilation must succeed. The original bug stamped two
#      `kgen.offload.kernelid` entries onto the wrapper FuncOp and tripped
#      `findFuncByKernelId` with "internal error: cannot find kernel by its
#      ID". `processCompileOffload` now routes thunked offloads to the
#      wrapped function, deduplicating the entry.
#
#   2. The user kernel surfaces in PTX with its linkage-name prefix and
#      auto-mangled hash. Both host paths look up that same symbol via
#      `evaluateMangledName`'s transparent-thunk redirect.
#
# Cross-compiled to NVPTX without requiring GPU hardware.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/out.s %s 2>&1 | FileCheck %s --allow-empty
# RUN: cat %t/*.ptx > %t/all_kernels.ptx
# RUN: FileCheck %s --check-prefix=CHECK-PTX --input-file=%t/all_kernels.ptx

# COM: Compilation must succeed without the regression's diagnostics.
# CHECK-NOT: duplicate LLVM metadata attribute for "kgen.offload.kernelid"
# CHECK-NOT: cannot find kernel by its ID

from max.gpu import global_idx
from max.gpu.host import DeviceContext


@__name("shared_kernel")
def shared_kernel(data: Pointer[Float32, MutAnyOrigin], n: Int32):
    var idx = global_idx.x
    if idx < Int(n):
        data[unsafe_offset=idx] = data[unsafe_offset=idx] + 1.0


def main() raises:
    with DeviceContext() as ctx:
        comptime n = 64
        var buf = ctx.enqueue_create_buffer[.float32](n)
        buf.enqueue_fill(0.0)

        # Direct offload path. `enqueue_function[func]` calls
        # `compile_offload` with `func = shared_kernel`, going through
        # `compileOffloads`'s non-parametric branch.
        ctx.enqueue_function[shared_kernel](
            buf, Int32(n), grid_dim=1, block_dim=32
        )
        ctx.synchronize()

        # Thunk-wrapped offload path. `enqueue_function` builds a
        # conversion thunk and binds `shared_kernel` as its `callee` input
        # parameter. `processCompileOffload` looks through the thunk and
        # reroutes the offload back to `shared_kernel`, so this collapses
        # onto the same GPU entry as the direct path.
        ctx.enqueue_function[shared_kernel](
            buf, Int32(n), grid_dim=1, block_dim=32
        )
        ctx.synchronize()


# ---------------------------------------------------------------------------
# FileCheck patterns
# ---------------------------------------------------------------------------

# A single GPU entry surfaces with the linkage-name prefix + hash; both host
# paths dispatch to it.
# CHECK-PTX: .visible .entry shared_kernel_{{[0-9a-f]+}}(
