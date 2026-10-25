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


# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s

# CHECK: error: 'TypeList' parameter 'Trait' has
# CHECK: note: 'TypeList' declared here
trait AnyType:
    pass

struct S[*Ts: AnyType]:
    pass

def foo[T: AnyType]():
    _ = S[T]()

def bar():
    comptime T = S[]
