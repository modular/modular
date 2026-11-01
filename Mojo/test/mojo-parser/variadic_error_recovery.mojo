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


# RUN: %parse-mojo-isolated --disable-builtins -verify-diagnostics %s

trait A:
    pass

trait B:
    pass

# expected-note @+1 {{'TypeList' declared here}}
struct TypeList[Trait: type_of(A), //, values: __mlir_type.index]:
    pass

# expected-error @+1 {{'TypeList' parameter 'Trait' has 'AnyTrait[A]' type, but value has type 'AnyTrait[B]'}}
struct S[*Ts: B]:
    pass

def foo[T: B]():
    _ = S[T]()

def bar():
    comptime T = S[]
