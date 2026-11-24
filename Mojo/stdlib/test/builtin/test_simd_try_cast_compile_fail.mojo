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
# RUN: not %mojo %s 2>&1 | FileCheck %s

# Test that `try_cast` rejects a floating-point source or target.


# CHECK: error: invalid call to 'try_cast': violated constraint
# CHECK-NEXT: Float32(1.5).try_cast[.int32]()
# CHECK: expected 'dtype.is_integral() and target.is_integral()'
# CHECK: error: invalid call to 'try_cast': violated constraint
# CHECK-NEXT: Int32(1).try_cast[.float32]()
def main() raises:
    _ = Float32(1.5).try_cast[.int32]()
    _ = Int32(1).try_cast[.float32]()
