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

# Dispatching a parametric trait method through `Ts[i]` in a `comptime for`
# resolves the witness at elaboration time. The witness entry's unbound slots
# carry types that depend on each other (`instance`'s type depends on the
# inferred `instance.Ts` pack); elaboration must not try to evaluate those
# types while the pack is still unbound.

# RUN: %mojo %s | FileCheck %s


def count_types[*Ts: AnyType]() -> Int:
    var count = 0
    comptime for i in range(len(Ts)):
        count += 1
    return count


struct Model[*Ts: AnyType](Copyable, Movable):
    var data: Array[Int, count_types[*Self.Ts]()]

    def __init__(out self):
        self.data = {fill = 0}


trait Plugin:
    @staticmethod
    def function[instance: Model]() -> Int:
        return len(materialize[instance]().data)


struct PluginA(Plugin):
    pass


struct PluginB(Plugin):
    pass


struct Plugins[*Ts: Plugin](TrivialRegisterPassable):
    pass


def main():
    comptime plugins = Plugins[PluginA, PluginB]
    comptime instance = Model[Int, Float32]()
    # CHECK: 2
    # CHECK-NEXT: 2
    comptime for i in range(len(plugins.Ts)):
        print(plugins.Ts[i].function[instance]())
