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
# Regression test for MOCO-3842: a parametric conversion thunk wrapping a
# user kernel where the callee's `SymbolConstantAttr` does NOT carry the
# matching compile-time params directly. The missing params are sourced
# from the thunk's own params at the inner `kgen.call_param[callee](…)`
# site inside the thunk body.
#
# The synthesised thunk is marked as a transparent thunk so
# `processCompileOffload` and `evaluateMangledName` both redirect through to
# the wrapped function — preserving the user's linkage name on the emitted
# GPU entry symbol. See `IREvaluatorContext.h`.
#
# Trigger: pass a parametric kernel into a generic helper whose `op_fn`
# formal differs in calling-convention effects (`thin` vs `thin raises`).
# The conv-thunk synthesised at the helper's invocation point is itself
# parametric in `width`, with the thunk's body applying that `width` to
# the wrapped callee at the inner call site.
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

from max.gpu import global_idx
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code


@__name("named_partial_kernel")
def named_partial_kernel[width: Int](data: Pointer[Float32, MutAnyOrigin]):
    var idx = global_idx.x
    data[unsafe_offset=idx] = data[unsafe_offset=idx] + Float32(width)


# Generic helper. The `op_fn` formal is `thin raises` while the kernel
# `named_partial_kernel` is `thin`, so a calling-convention conversion
# thunk is synthesised at the helper's instantiation point. The thunk's
# `expected` signature inherits the parametric `<width: Int>`, so the
# thunk itself ends up parametric — and its `callee` paramValue carries
# `named_partial_kernel` *without* `width` applied directly to the
# SymbolConstantAttr. The redirect machinery still recovers the
# fully-bound callee via `kgen.transparent_thunk_callee_expr`.
def helper[
    op_fn: def[width: Int](
        data: Pointer[Float32, MutAnyOrigin]
    ) thin raises -> None,
]() raises:
    comptime target = get_gpu_target["sm_90a"]()
    var info = _compile_code[op_fn[width=2], target=target]()
    # Touch the result so the call isn't optimised away.
    print(info.function_name)


def main() raises:
    helper[named_partial_kernel]()


# ---------------------------------------------------------------------------
# FileCheck patterns
# ---------------------------------------------------------------------------

# The user's `@__name` reaches the PTX entry symbol even though the conv
# thunk is parametric and the wrapped callee's SymbolConstantAttr is
# partially bound at the SymbolConstantAttr level.
# CHECK-PTX: .visible .entry named_partial_kernel_{{[0-9a-f]+}}(
