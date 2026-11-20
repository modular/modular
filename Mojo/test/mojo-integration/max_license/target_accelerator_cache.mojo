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

# RUN: %mojo --target-accelerator=nvidia:sm_90a %s | FileCheck %s --check-prefix=CHECK-SM90
# RUN: %mojo --target-accelerator=nvidia:sm_100a %s | FileCheck %s --check-prefix=CHECK-SM100

# COM: testing with different target-accelerator values here with the same
#      mojo input module to make sure that target-accelerator is also included
#      in the mojo cache key generation so that the two runs produces different
#      binaries.

from std.compile import compile_info
from max.gpu.host import get_gpu_target
from std.sys import argv
from std.sys.info import _is_sm_100x


def kernel_alias() -> Int:
    comptime if _is_sm_100x():
        print("Compiling for sm100x")
        return 42
    else:
        print("Compiling for not sm100x")
        return 27


def kernel_entry() -> Int:
    comptime K = kernel_alias()
    return K


def main():
    var r = compile_info[
        kernel_entry,
        target=get_gpu_target(),
        emission_kind="asm",
    ]()
    print(r.asm)


# CHECK-SM90: Compiling for not sm100x
# CHECK-SM90: 27

# CHECK-SM100: Compiling for sm100x
# CHECK-SM100: 42
