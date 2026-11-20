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
# RUN: %mojo %s | FileCheck %s

# A closure returning `None` passed where the result type is a generic
# `R: AnyType` binds `R` to `__MLIRType[none]`. Its byref result slot must
# still match the closure's conversion to `def() -> R`.


def run[R: AnyType, //, F: def() -> R](f: F) -> R:
    return f()


def main():
    var n = 3

    def attempt() {imm} -> None:
        print(n)

    # CHECK: 3
    run(attempt)
