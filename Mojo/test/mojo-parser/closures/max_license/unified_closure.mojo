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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s

# COM: Verify that @__llvm_metadata on a closure is preserved on the op

# CHECK: fnAttrs = ["nvvm.maxntid", #pop.array<256> : !pop.array<1, i32>]


def metadata_closure(x: Int):
    @__llvm_metadata(
        `nvvm.maxntid`=__mlir_attr.`#pop.array<256> : !pop.array<1, i32>`
    )
    def _kernel() {var x} -> Int:
        return x

    _ = _kernel()

# // -----

# COM: Verify that @__llvm_arg_metadata on a closure is preserved

# CHECK: fnArgAttrs
# CHECK-SAME: "nvvm.grid_constant", unit


def arg_metadata_closure(x: Int):
    @__llvm_arg_metadata(x, `nvvm.grid_constant`)
    def _kernel(x: Int) {var} -> Int:
        return x

    _ = _kernel(x)
