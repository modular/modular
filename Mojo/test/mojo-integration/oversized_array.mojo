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

# Oversized Array types used to wrap in layout math (size 0 / INT64_MIN) and
# alias neighbouring locals. Reject them at compile time.


def main():
    # CHECK: error: type is too large to fit in addressable memory
    var a = Array[Int, 1 << 62](fill=7)
    print(a[0])
