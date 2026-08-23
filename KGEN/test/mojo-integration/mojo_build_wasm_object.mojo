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

# RUN: %mojo-build %s -o %t.o --emit object --target-triple=wasm32-unknown-unknown
# RUN: file %t.o | FileCheck %s
# CHECK: WebAssembly (wasm) binary module


@export("add_one")
def add_one(value: Int32) abi("C") -> Int32:
    return value + 1
