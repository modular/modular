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
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm80 %s | FileCheck --check-prefix=CHECK-NV80 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm90 %s | FileCheck --check-prefix=CHECK-NV90 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm100 %s | FileCheck --check-prefix=CHECK-NV100 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm120 %s | FileCheck --check-prefix=CHECK-NV120 %s
# RUN: %mojo-no-debug --target-accelerator=some_amd:300 %s | FileCheck --check-prefix=CHECK-A300 %s

from std.sys.info import (
    _has_sm_8x_or_newer,
    _has_sm_9x_or_newer,
    _has_sm_100x_or_newer,
    _has_sm_120x_or_newer,
)


def main() raises:
    # CHECK-NV80: True
    # CHECK-NV90: True
    # CHECK-NV100: True
    # CHECK-NV120: True
    # CHECK-A300: False
    print(_has_sm_8x_or_newer())

    # CHECK-NV80: False
    # CHECK-NV90: True
    # CHECK-NV100: True
    # CHECK-NV120: True
    # CHECK-A300: False
    print(_has_sm_9x_or_newer())

    # CHECK-NV80: False
    # CHECK-NV90: False
    # CHECK-NV100: True
    # CHECK-NV120: True
    # CHECK-A300: False
    print(_has_sm_100x_or_newer())

    # CHECK-NV80: False
    # CHECK-NV90: False
    # CHECK-NV100: False
    # CHECK-NV120: True
    # CHECK-A300: False
    print(_has_sm_120x_or_newer())
