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
# Test that `mojo build --emit asm` writes the offload kernel file for a kernel
# that an earlier build already compiled without asking for it.
#
# The offload file is recorded in the compile-cache entry, so the `--emit object`
# build below fills the cache with a recording that holds no file at all. The
# `--emit asm` build must not reuse that recording.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t && mkdir -p %t

# RUN: %mojo-build --emit object \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/out.o %s

# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/out.s %s
# RUN: test $(ls %t/out_offload_kind_probe_kernel_*.ptx | wc -l) -eq 1

from max.gpu.host import DeviceContext


@__name("offload_kind_probe_kernel")
def offload_kind_probe_kernel():
    pass


def main() raises:
    with DeviceContext() as ctx:
        ctx.enqueue_function[offload_kind_probe_kernel](grid_dim=1, block_dim=1)
