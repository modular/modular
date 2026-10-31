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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# `List` and `Array` only convert to a `PythonObject` when their element type
# conforms to `ConvertibleToPython`.

from std.python import PythonObject


@fieldwise_init
struct NotConvertible(Copyable):
    var value: Int


def main() raises:
    # CHECK: argument type 'List[NotConvertible]' does not conform to trait 'ConvertibleToPython'
    var list: List[NotConvertible] = [NotConvertible(1)]
    _ = PythonObject(list^)

    # CHECK: argument type 'Array[NotConvertible, Int(1)]' does not conform to trait 'ConvertibleToPython'
    var array: Array[NotConvertible, 1] = [NotConvertible(1)]
    _ = PythonObject(array^)
