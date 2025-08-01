# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


# Note: this code doesn't appear in the doc; it tests the assertions
# in the doc.
@register_passable
trait RegPassableTrait:
    fn __init__(out self):
        ...

    fn say_hello(self):
        ...


@register_passable
struct RegPassableType(RegPassableTrait):
    fn __init__(out self):
        pass

    fn say_hello(self):
        print("Hello from a register passable type!")


fn test_register_passable_type():
    var a: RegPassableType = RegPassableType()
    # Ensure that the value is movable
    var b = a^
    b.say_hello()


# Ensure that we can define a register-passable trivial type that conforms to a
# register-passable trait
@register_passable("trivial")
struct RegPassableType2(RegPassableTrait):
    fn __init__(out self):
        pass

    fn say_hello(self):
        print("Hello from a register passable type!")


@register_passable("trivial")
trait RegPassableTrivialTrait:
    fn __init__(out self, value: Int = 0):
        ...

    fn say_hello(self):
        ...


@register_passable("trivial")
struct RegPassableTrivialType(RegPassableTrivialTrait):
    var value: Int

    fn __init__(out self, value: Int = 0):
        self.value = value

    fn say_hello(self):
        print("Hello from a register passable trivial type!")


fn test_register_passable_trivial_type():
    # Type is copyable and movable
    var a_list = List[RegPassableTrivialType]()
    a_list.append(RegPassableTrivialType())
    a_list.append(RegPassableTrivialType())
    for item in a_list:
        item.say_hello()


def main():
    test_register_passable_type()
    test_register_passable_trivial_type()
