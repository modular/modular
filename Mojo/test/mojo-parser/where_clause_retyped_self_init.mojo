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

# Regression test: calling a constructor whose `out self` retypes `Self` (here
# pinning `n` to 1) on a struct with a `where` constraint, with the remaining
# parameters inferred from the call's arguments inside a caller whose own
# `where` clause supplies the evidence. The discharged-constraint mask computed
# during inference was sized to one parameter list's body constraints but
# applied to a generator with a different count, tripping
# "discharged body constraint mask must match the metadata" in
# PogListAttr::getSpecializedMetadata.


trait HasSize:
    comptime size: Int


struct Operand[T: HasSize, n: Int] where T.size == 1:
    def __init__(out self: Operand[Self.T, 1], value: Self.T):
        pass


# CHECK-LABEL: lit.fn @"infer_with_retyped_self
def infer_with_retyped_self[T: HasSize](value: T) where T.size == 1:
    # CHECK: lit.var.decl "op"
    # CHECK-SAME: #Operand <:{{.*}} T, :!Int {:scalar<index> 1}>
    var op = Operand(value)
    _ = op
