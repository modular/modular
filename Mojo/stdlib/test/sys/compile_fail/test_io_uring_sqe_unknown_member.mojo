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

# A name that is not a C member of `struct io_uring_sqe`, here a typo of
# `fsync_flags`, fails to compile.

from std.sys._io_uring import IORING_OP_NOP, io_uring_sqe


def main():
    var sqe = io_uring_sqe(opcode=IORING_OP_NOP)
    # CHECK: constraint failed: not a member of `io_uring_sqe`
    sqe.fsync_flag = 1
