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

# RUN: %mojo %s

from std.traits import Deinitable as BuiltinDeinitable


trait Deinitable:
    pass


__enum Value(not Deinitable):
    case first


def main():
    comptime assert conforms_to(Value, BuiltinDeinitable)
    comptime assert not conforms_to(Value, Deinitable)
    comptime assert conforms_to(Value, TrivialRegisterPassable)
