# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# RUN: %mojo-no-debug --target-accelerator=nvidia:80 %s | FileCheck --check-prefix=CHECK-NV80 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:90 %s | FileCheck --check-prefix=CHECK-NV90 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:120 %s | FileCheck --check-prefix=CHECK-NV120 %s

from sys.info import _accelerator_arch, _is_sm_9x, _is_sm_9x_or_newer

from gpu.host.compile import _compile_code_asm
from gpu.host import get_gpu_target
from testing import *


fn check_sm9x() -> Bool:
    alias v = _is_sm_9x()
    return v


fn check_sm9x_or_newer() -> Bool:
    alias v = _is_sm_9x_or_newer()
    return v


def main():
    alias accelerator_arch = _accelerator_arch()

    # CHECK-NV80: ret i1 false
    # CHECK-NV90: ret i1 true
    # CHECK-NV120: ret i1 false
    print(
        _compile_code_asm[
            check_sm9x,
            emission_kind="llvm",
            target = get_gpu_target[_accelerator_arch()](),
        ]()
    )

    # CHECK-NV80: ret i1 false
    # CHECK-NV90: ret i1 true
    # CHECK-NV120: ret i1 true
    print(
        _compile_code_asm[
            check_sm9x_or_newer,
            emission_kind="llvm",
            target = get_gpu_target[_accelerator_arch()](),
        ]()
    )
