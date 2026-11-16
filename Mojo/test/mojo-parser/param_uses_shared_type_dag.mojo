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

# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

# Related to MOCO-5240.

# `Twice[T]` names `T` twice. The type in `nested` nests `Twice` 16 levels deep
# and depends on the parameter `n`. The type is therefore a DAG with 2^16 paths.
# Collecting the parameter uses of the type must visit each shared sub-type
# once.

# CHECK-LABEL: nested


struct Leaf[n: Int]:
    def __init__(out self):
        pass


struct Pair[A: AnyType, B: AnyType]:
    def __init__(out self):
        pass


comptime Twice[T: AnyType] = Pair[T, T]

comptime Nest4[T: AnyType] = Twice[Twice[Twice[Twice[T]]]]


def sink[T: AnyType](x: T):
    pass


def nested[n: Int]():
    var x = Nest4[Nest4[Nest4[Nest4[Leaf[n]]]]]()
    sink(x)
