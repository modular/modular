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

# A failed struct binding reports only the error that failed it, not the
# constraints it deferred before failing.


struct CondCopyable[T: Movable & Deinitable](
    Copyable where conforms_to(T, Copyable), Movable
):
    var value: Self.T


struct NeedsCopyablePos[T: Copyable, n: Int](Movable where False) where n > 0:
    pass


# CHECK: [[@LINE+3]]:{{[0-9]+}}: error: violated constraint
# CHECK-NOT: error:
def failed_binding[U: Movable & Deinitable](
    x: NeedsCopyablePos[CondCopyable[U], 0]
):
    pass


# CHECK: [[@LINE+3]]:{{[0-9]+}}: error: invalid bindings in signature: lacking evidence
# CHECK-NOT: error:
def successful_binding[U: Movable & Deinitable](
    x: NeedsCopyablePos[CondCopyable[U], 1]
):
    pass
