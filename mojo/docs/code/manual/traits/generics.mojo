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


trait Stacklike:
    alias EltType: Copyable & Movable

    fn push(mut self, var item: Self.EltType):
        ...

    fn pop(mut self) -> Self.EltType:
        ...


struct MyStack[type: Copyable & Movable](Stacklike):
    """A simple Stack built using a List."""

    alias EltType = type
    alias list_type = List[Self.EltType]

    var list: Self.list_type

    fn __init__(out self):
        self.list = Self.list_type()

    fn push(mut self, var item: Self.EltType):
        self.list.append(item)

    fn pop(mut self) -> Self.EltType:
        return self.list.pop()

    fn dump[
        WritableEltType: Writable & Copyable & Movable
    ](self: MyStack[WritableEltType]):
        print("[", end="")
        for item in self.list:
            print(item, end=", ")
        print("]")


def add_to_stack[S: Stacklike](mut stack: S, item: S.EltType):
    stack.push(item)


def main():
    # list example
    var list: List[Int]
    list = [1, 2, 3, 4]
    for i in range(len(list)):
        print(list[i], end=" ")
    print()

    # Stacklike example
    s = MyStack[Int]()
    add_to_stack(s, 12)
    add_to_stack(s, 33)
    s.dump()  # [12, 33]
    print(s.pop())  # 33
