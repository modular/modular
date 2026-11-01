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
# Regression test for MOCO-3842: a GPU kernel invoked via
# `enqueue_function` (single-arg form) goes through a compiler-
# generated conversion thunk before offload bundling. The thunk has no
# linkage name or LLVM metadata of its own; the user kernel's @__name and
# @__llvm_metadata must propagate transparently through the thunk to the
# emitted offload entry.
#
# Without the fix, `enqueue_function` produced auto-mangled PTX
# entry symbols and dropped @__llvm_metadata, regardless of what the user
# authored. With the fix:
#
#   * Kernels surface @__name plus a uniqueness hash, allowing multiple
#     parametric instantiations to coexist.
#   * Parametric @__name expressions (tstrings interpolating user-kernel
#     params, e.g. `t"foo_{c_type}"`) are evaluated in the user kernel's
#     own parameter scope at stub-gen time, so each concrete instantiation
#     surfaces a distinct human-readable PTX entry symbol.
#   * @__llvm_metadata such as MAX_THREADS_PER_BLOCK_METADATA flows through
#     the thunk as a `.maxntid` directive on the emitted PTX entry, even
#     when its value depends on a user-kernel parameter.
#
# Cross-compiled to NVPTX without requiring GPU hardware.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/out.s %s
# RUN: cat %t/*.ptx > %t/all_kernels.ptx
# RUN: FileCheck %s --check-prefix=CHECK-PTX --input-file=%t/all_kernels.ptx

from max.gpu import global_idx, MAX_THREADS_PER_BLOCK_METADATA
from max.gpu.host import DeviceContext
from std.utils.static_tuple import StaticTuple


# Parametric kernel: each instantiation gets a unique suffix, and
# @__llvm_metadata propagates per-instantiation.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(64))
)
@__name("experimental_mangled_kernel")
def mangled_name_kernel[x: Int](data: Pointer[Float32, MutAnyOrigin], n: Int32):
    var idx = global_idx.x
    if idx < Int(n):
        data[unsafe_offset=idx] = data[unsafe_offset=idx] + Float32(x)


# Parametric @__name and parametric @__llvm_metadata: the
# .maxntid directive depends on a user-kernel parameter, exercising metadata
# parameter substitution through the thunk.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"exp_param_mangled_{threads}")
def parametric_mangled_kernel[
    threads: Int
](data: Pointer[Float32, MutAnyOrigin], n: Int32):
    var idx = global_idx.x
    if idx < Int(n):
        data[unsafe_offset=idx] = data[unsafe_offset=idx] + 1.0


def main() raises:
    with DeviceContext() as ctx:
        comptime n = 64
        var buf_f32 = ctx.enqueue_create_buffer[.float32](n)
        buf_f32.enqueue_fill(0.0)

        # Two parametric instantiations of the same @__name kernel: each must
        # surface in PTX with a distinct hash suffix.
        ctx.enqueue_function[mangled_name_kernel[5]](
            buf_f32, Int32(n), grid_dim=1, block_dim=64
        )
        ctx.enqueue_function[mangled_name_kernel[6]](
            buf_f32, Int32(n), grid_dim=1, block_dim=64
        )

        # Parametric tstring @__name plus parametric @__llvm_metadata: name and
        # .maxntid both depend on the same user kernel param, exercising
        # metadata parameter substitution.
        ctx.enqueue_function[parametric_mangled_kernel[16]](
            buf_f32, Int32(n), grid_dim=1, block_dim=16
        )
        ctx.enqueue_function[parametric_mangled_kernel[48]](
            buf_f32, Int32(n), grid_dim=1, block_dim=48
        )
        ctx.synchronize()


# ---------------------------------------------------------------------------
# FileCheck patterns
# ---------------------------------------------------------------------------

# Non-parametric @__name: two instantiations, each with a unique hash suffix
# and the authored .maxntid 64 directive. The DAG matches ensure both entries
# exist and that FileCheck binds them to different lines.
# CHECK-PTX-DAG: .visible .entry experimental_mangled_kernel_{{[0-9a-f]+}}(
# CHECK-PTX-DAG: .visible .entry experimental_mangled_kernel_{{[0-9a-f]+}}(
# CHECK-PTX-DAG: .maxntid 64
# CHECK-PTX-DAG: .maxntid 64

# Parametric @__name and parametric @__llvm_metadata: both the literal portion
# of the name and the .maxntid value depend on the user-kernel param. Each
# instantiation produces a unique entry with the bound int interpolated and its
# own hash suffix, plus the matching .maxntid directive.
# CHECK-PTX-DAG: .visible .entry exp_param_mangled_16_{{[0-9a-f]+}}(
# CHECK-PTX-DAG: .visible .entry exp_param_mangled_48_{{[0-9a-f]+}}(
# CHECK-PTX-DAG: .maxntid 16
# CHECK-PTX-DAG: .maxntid 48
