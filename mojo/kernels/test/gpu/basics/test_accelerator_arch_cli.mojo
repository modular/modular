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
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm80 %s | FileCheck --check-prefix=CHECK-NV80 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm90 %s | FileCheck --check-prefix=CHECK-NV90 %s
# RUN: %mojo-no-debug --target-accelerator=some_amd:300 %s | FileCheck --check-prefix=CHECK-A300 %s
# RUN: %mojo-no-debug --target-accelerator=quantum:3000 %s | FileCheck --check-prefix=CHECK-Q3000 %s

from sys.info import _accelerator_arch


def main():
    # CHECK-NV80: nvidia:sm80
    # CHECK-NV90: nvidia:sm90
    # CHECK-A300: some_amd:300
    # CHECK-Q3000: quantum:3000
    print(_accelerator_arch())
