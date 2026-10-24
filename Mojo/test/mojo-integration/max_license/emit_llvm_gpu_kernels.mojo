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
# Test that `mojo build --emit llvm` saves GPU kernel LLVM IR files alongside
# the host LLVM IR output.
#
# Both targets are cross-compiled and run unconditionally on any host platform.
#
# The same kernel is instantiated for two element types (Float32 and Float16),
# producing two offload files per target. File names match the symbol inside
# each file (MOCO-3785).
#
# The kernel uses enqueue_function, which invokes kernels via
# compiler-generated conversion thunks. @__name on add_one_kernel propagates
# through the thunk to the emitted offload entry (MOCO-3842), so file names
# match `out_*_add_one_kernel_<hash>.<ext>` where the hash is the mangled
# uniqueness suffix derived per parametric instantiation.
#
# NVIDIA / PTX: expects three output files:
#   - <out>.ll                  : host LLVM IR (x86)
#   - <out>_<sym1>.nvptx.ll     : one element-type instantiation
#   - <out>_<sym2>.nvptx.ll     : the other element-type instantiation
#
# Apple / Metal: expects three output files:
#   - <out>.ll                  : host LLVM IR (ARM64)
#   - <out>_<sym1>.metal.ll     : one element-type instantiation
#   - <out>_<sym2>.metal.ll     : the other element-type instantiation
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t

# --- NVIDIA / PTX ---
# RUN: %mojo-build --emit llvm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/out_ptx.ll %s
# RUN: test -e %t/out_ptx.ll
# RUN: test $(ls %t/out_ptx_add_one_kernel_*.nvptx.ll | wc -l) -eq 2
# RUN: for f in %t/out_ptx_add_one_kernel_*.nvptx.ll; do \
# RUN:   FileCheck %s --check-prefix=CHECK-PTX --input-file="$f"; \
# RUN: done

# --- Apple / Metal (LLVM IR) ---
# RUN: %mojo-build --emit llvm \
# RUN:   --target-triple arm64-apple-darwin25.3.0 \
# RUN:   --target-accelerator metal:4 \
# RUN:   -o %t/out_metal.ll %s
# RUN: test -e %t/out_metal.ll
# RUN: test $(ls %t/out_metal_add_one_kernel_*.metal.ll | wc -l) -eq 2
# RUN: for f in %t/out_metal_add_one_kernel_*.metal.ll; do \
# RUN:   FileCheck %s --check-prefix=CHECK-METAL --input-file="$f"; \
# RUN: done

from max.gpu import global_idx
from max.gpu.host import DeviceContext


@__name("add_one_kernel")
def add_one_kernel[
    dtype: DType
](data: Pointer[Scalar[dtype], MutAnyOrigin], n: Int32):
    var idx = global_idx.x
    if idx < Int(n):
        data[unsafe_offset=idx] = data[unsafe_offset=idx] + 1


def main() raises:
    with DeviceContext() as ctx:
        comptime n = 64
        var buf_f32 = ctx.enqueue_create_buffer[.float32](n)
        var buf_f16 = ctx.enqueue_create_buffer[.float16](n)
        buf_f32.enqueue_fill(0.0)
        buf_f16.enqueue_fill(0.0)
        ctx.enqueue_function[add_one_kernel[.float32]](
            buf_f32, Int32(n), grid_dim=1, block_dim=n
        )
        ctx.enqueue_function[add_one_kernel[.float16]](
            buf_f16, Int32(n), grid_dim=1, block_dim=n
        )
        ctx.synchronize()


# Each file is checked individually by the loop above. Checks verify that
# each file is a valid GPU kernel LLVM IR module.

# CHECK-PTX: target triple = "nvptx64-
# CHECK-PTX: fadd
# CHECK-PTX: llvm.nvvm

# CHECK-METAL: target triple = "air64-
# CHECK-METAL: fadd contract
# CHECK-METAL: "metal.kernel"="true"
