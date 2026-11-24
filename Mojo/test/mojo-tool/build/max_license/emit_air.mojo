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
# `--emit` accepts the emission kinds declared by the registered TargetTraits:
# the help text lists them per kind, unknown kinds are rejected, and `air`
# (Apple AIR bitcode) is only accepted for Metal targets.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build --help | grep -v ' (tpu) ' | FileCheck %s --check-prefix=CHECK_HELP

# The emission kinds print inside the `--emit` help block, deduplicated per
# (kind, description) with the sharing targets listed, kind-sorted. The grep
# drops the `--config=tpu` row.
# CHECK_HELP: --emit <FILE_TYPE>
# CHECK_HELP: air (metal)
# CHECK_HELP-NEXT: asm (amdgpu, host, metal, nvptx)
# CHECK_HELP-NEXT: exe (amdgpu, host, metal, nvptx)
# CHECK_HELP-NEXT: llvm (amdgpu, host, metal, nvptx)
# CHECK_HELP-NEXT: llvm-bitcode (amdgpu, host, metal, nvptx)
# CHECK_HELP-NEXT: object (amdgpu, host, metal, nvptx)
# CHECK_HELP-NEXT: shared-lib (amdgpu, host, metal, nvptx)

# RUN: not %mojo-build --emit bogus %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_UNKNOWN
# CHECK_UNKNOWN: unknown emission kind 'bogus'

# RUN: not %mojo-build --emit air %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_NO_METAL_HOST
# CHECK_NO_METAL_HOST: target 'host' does not support emission kind 'air'
# RUN: not %mojo-build --emit air --target-accelerator sm_90 %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_NO_METAL_GPU
# CHECK_NO_METAL_GPU: target 'nvptx' does not support emission kind 'air'
# RUN: not %mojo-build --emit air --target-accelerator not-an-arch %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_BAD_ARCH
# CHECK_BAD_ARCH: unknown accelerator architecture 'not-an-arch'

# `--emit air` with a Metal accelerator produces host bitcode plus an AIR
# (LLVM bitcode wrapper, magic 0xDEC04342) sidecar for the kernel.
# RUN: %mojo-build --emit air --target-accelerator apple-m4 %s -o %t.bc
# RUN: test -e %t.bc
# RUN: od -A n -t x1 -N 4 %t_*.air | FileCheck %s --check-prefix=CHECK_AIR_MAGIC
# CHECK_AIR_MAGIC: {{de c0 17 0b|42 43 c0 de}}

# RUN: not kgen -emit=air %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK_KGEN_TRIPLE
# CHECK_KGEN_TRIPLE: target 'host' does not support emission kind 'air'

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime layout = row_major[16]()


def fill_kernel(x: TileTensor[DType.float32, type_of(layout), MutAnyOrigin]):
    comptime assert x.flat_rank == 1
    var i = global_idx.x
    if i < 16:
        x[i] = 1.0


def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](16)
    var x = TileTensor(buf, layout)
    ctx.enqueue_function[fill_kernel](x, grid_dim=1, block_dim=16)
    ctx.synchronize()
