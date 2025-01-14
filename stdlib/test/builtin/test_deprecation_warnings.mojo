# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s 2>&1 1>/dev/null | FileCheck %s --check-prefix=CHECK-STDERR


fn main():
    # FIXME(25.1): Move `int` deprecation warnings to a compiler error
    # CHECK-STDERR: warning: the `int` function is deprecated, use the `Int` constructor instead
    _ = int(42)
