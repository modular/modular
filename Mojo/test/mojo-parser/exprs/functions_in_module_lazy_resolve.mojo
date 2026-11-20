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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Verify the intrinsic works when resolving a function's signature reallocates
# the module scope it iterates. `f` must stay after `main` and name prelude
# types not yet looked up, so its signature resolves during the intrinsic.


# CHECK-LABEL: lit.fn @"main
# CHECK: store_to_mem(@functions_in_module_lazy_resolve::@"f(
# CHECK-SAME: store_to_mem(@functions_in_module_lazy_resolve::@"take[
def main():
    take[__functions_in_module()]()


def f(
    a: Int,
    b: Bool,
    c: Float32,
    d: Float64,
    e: Int32,
    g: UInt32,
    h: UInt8,
    i: Byte,
    j: String,
    k: StaticString,
    l: NoneType,
    m: Error,
):
    pass


def take[func_type: Movable, //, func: func_type]():
    pass
