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

# Test that `try_cast` rejects a sub-byte target, which passes the `where`
# clause but whose `cast` doesn't truncate.


# CHECK: `try_cast` doesn't support sub-byte dtypes
def main() raises:
    _ = UInt8(1).try_cast[DType._uint4]()
