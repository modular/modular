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

# Regression test: a parameter default that references an earlier parameter
# left as `_` at the use site must be evaluated against the inferred binding.
# The quoted default's reference to `T` was resolved against only the
# explicitly bound parameters, so it indexed out of range (or bound the wrong
# parameter), instead of using the `T` inferred from the constructor argument.


trait HasSize:
    comptime size: Int


@fieldwise_init
struct Impl(HasSize):
    comptime size = 42


struct Holder[n: Int, T: HasSize, idx: Int = T.size]:
    def __init__(out self, value: Self.T):
        pass


# CHECK-LABEL: lit.fn @"infer_param_with_dependent_default()
def infer_param_with_dependent_default():
    # CHECK: lit.var.decl "h"
    # CHECK-SAME: #Holder <:!Int {:scalar<index> 0}, :{{.*}} !Impl, :!Int {:scalar<index> 42}>
    var h = Holder[0, _](Impl())
    _ = h
