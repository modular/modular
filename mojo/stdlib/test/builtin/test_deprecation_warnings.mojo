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
# RUN: %mojo %s 2>&1 1>/dev/null | FileCheck %s --check-prefix=CHECK-STDERR


fn main():
    # FIXME(25.2): Move deprecation warnings to compiler errors
    # CHECK-STDERR: warning: the `str` function is deprecated, use the `String` constructor instead
    _ = str(0)
    # CHECK-STDERR: warning: the `bool` function is deprecated, use the `Bool` constructor instead
    _ = bool(False)
    # CHECK-STDERR: warning: the `float` function is deprecated, use the `Float64` constructor instead
    _ = float(42.4)
