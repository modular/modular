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
# Test that --emit=asm output file names match the PTX symbol names inside.
# Cases covered:
#   1. @__name     -> file name equals @__name prefix + hash suffix
#   2. No @__name  -> file exists, name matches the PTX symbol
#   3. Host target -> symbol name is not sanitized, but file name still is
#
# The per-case checks verify name and symbol independently. A universal
# invariant loop at the end confirms every produced .ptx file's basename
# equals the .visible .entry symbol inside.
#
# Cross-compiled to NVPTX without requiring GPU hardware.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/out.s %s

# Case 1: @__name - file name is @__name prefix + hash suffix, and the same
# name appears as the PTX entry symbol.
# RUN: ls %t/ | grep '^out_mangled_kernel_[0-9a-f]*\.ptx$' | \
# RUN:   FileCheck %s --check-prefix=CHECK-MANGLE-FILE
# RUN: cat %t/out_mangled_kernel_*.ptx | \
# RUN:   FileCheck %s --check-prefix=CHECK-MANGLE-PTX

# Case 2: No @__name — exactly one PTX file is produced for this kernel;
# its name is the sanitized PTX symbol and is covered by the invariant below.
# RUN: test $(ls %t/out_*.ptx | wc -l) -eq 2

# Case 3: an offload to a target whose symbol names are not sanitized. The
# file name must be sanitized regardless.
# RUN: ls %t/ | grep '^out_.*\.s$' | FileCheck %s --check-prefix=CHECK-HOST-FILE

# Universal invariant: for every produced .ptx file, stripping the "out_"
# prefix and ".ptx" extension from the filename yields the exact PTX entry
# symbol inside the file.
# RUN: for f in %t/out_*.ptx; do \
# RUN:   sym="$(basename "$f" .ptx)"; sym="${sym#out_}"; \
# RUN:   grep -q ".visible .entry $sym(" "$f" || \
# RUN:     { echo "FAIL: $(basename $f) contains no entry $sym" >&2; exit 1; }; \
# RUN: done

from std.compile import compile_info
from max.gpu import global_idx
from max.gpu.host import DeviceContext


# Case 1: @__name — PTX file must be out_mangled_kernel_<hash>.ptx.
@__name("mangled_kernel")
def mangled_kernel_fn(data: Pointer[Float32, MutAnyOrigin], n: Int32):
    var idx = global_idx.x
    if idx < Int(n):
        data[unsafe_offset=idx] = data[unsafe_offset=idx] * 2.0


# Case 2: No @__name — file name is determined by the sanitized PTX symbol.
def no_name_kernel_fn(data: Pointer[Float32, MutAnyOrigin], n: Int32):
    var idx = global_idx.x
    if idx < Int(n):
        data[unsafe_offset=idx] = data[unsafe_offset=idx] - 1.0


# Case 3: offloaded to the host target, so its symbol keeps the mangled form.
def host_offload_fn(data: Pointer[Float32, MutAnyOrigin], n: Int32):
    for i in range(Int(n)):
        data[unsafe_offset=i] = data[unsafe_offset=i] + 1.0


def main() raises:
    _ = compile_info[host_offload_fn, emission_kind="llvm"]()

    with DeviceContext() as ctx:
        comptime n = 64
        var buf = ctx.enqueue_create_buffer[.float32](n)
        buf.enqueue_fill(0.0)

        comptime k1 = mangled_kernel_fn
        ctx.enqueue_function[k1](buf, Int32(n), grid_dim=1, block_dim=n)

        comptime k2 = no_name_kernel_fn
        ctx.enqueue_function[k2](buf, Int32(n), grid_dim=1, block_dim=n)

        ctx.synchronize()


# ---------------------------------------------------------------------------
# FileCheck patterns
# ---------------------------------------------------------------------------

# Case 1: a file with the expected prefix + hex-hash pattern exists.
# CHECK-MANGLE-FILE: out_mangled_kernel_{{[0-9a-f]+}}.ptx
# Case 1: the symbol inside carries the same prefix + hash.
# CHECK-MANGLE-PTX: .visible .entry mangled_kernel_{{[0-9a-f]+}}(

# Case 3: the mangled symbol is `<module>::host_offload_fn(::Pointer[...],...)`.
# Only its sanitized form is a file name.
# CHECK-HOST-FILE: out_{{[A-Za-z0-9_]+}}_host_offload_fn_{{[A-Za-z0-9_]+}}.s
