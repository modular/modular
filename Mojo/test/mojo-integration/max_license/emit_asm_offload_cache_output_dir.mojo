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
# Test that `mojo build --emit asm` writes the offload kernel file next to the
# host output of the build that asked for it, however many times the kernel was
# built before.
#
# The offload file is recorded in the compile-cache entry, so every build after
# the first one replays a recording that an earlier build made.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t && mkdir -p %t/d1 %t/d2

# Fill the cache.
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/d1/out.s %s
# RUN: test $(ls %t/d1/out_offload_dir_probe_kernel_*.ptx | wc -l) -eq 1
# RUN: touch %t/before_second_build

# Replay it into another directory: d2 gets the file, d1 keeps the one it has.
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/d2/out.s %s
# RUN: test $(ls %t/d2/out_offload_dir_probe_kernel_*.ptx | wc -l) -eq 1
# RUN: test -z "$(find %t/d1 -name '*.ptx' -newer %t/before_second_build)"

# Delete the file and build into the same directory again: it comes back.
# RUN: rm %t/d2/out_offload_dir_probe_kernel_*.ptx
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/d2/out.s %s
# RUN: test $(ls %t/d2/out_offload_dir_probe_kernel_*.ptx | wc -l) -eq 1

from max.gpu.host import DeviceContext


@__name("offload_dir_probe_kernel")
def offload_dir_probe_kernel():
    pass


def main() raises:
    with DeviceContext() as ctx:
        ctx.enqueue_function[offload_dir_probe_kernel](grid_dim=1, block_dim=1)
