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
# The cubin arch of an NVIDIA `emission_kind="object"` offload comes from the
# offload target itself, never from `--target-accelerator`. Both accelerators
# below disagree with the sm_90a offload target: `nvidia:sm_75` names a lower
# arch, and the bare `sm_90a` spelling has no `nvidia:` prefix to strip.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: system-linux
# RUN: %mojo --target-accelerator=nvidia:sm_75 %s | FileCheck %s
# RUN: %mojo --target-accelerator=sm_90a %s | FileCheck %s

from std.compile import compile_info
from max.gpu.host import get_gpu_target


def add_one(dst: Pointer[Float32, MutAnyOrigin], x: Float32):
    dst[] = x + 1


def main():
    var cubin = compile_info[
        add_one, emission_kind="object", target=get_gpu_target["sm_90a"]()
    ]().asm.as_bytes()
    # A cubin is an ELF64, whose `e_flags` word starts at byte 48. A cubin keeps
    # its SM version in the second of those bytes (llvm `ELF::EF_CUDA_SM_MASK`
    # is 0xff00 with `EF_CUDA_SM_OFFSET` 8, and `EF_CUDA_SM90` is 0x5a == 90).
    comptime e_flags_sm_byte = 49
    # CHECK: cubin sm: 90
    print("cubin sm:", Int(cubin[e_flags_sm_byte]))
