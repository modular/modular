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


trait Quackable:
    fn quack(self):
        ...


trait Flyable:
    fn fly(self):
        ...


fn quack_and_go[type: Quackable & Flyable](quacker: type):
    quacker.quack()
    quacker.fly()


@fieldwise_init
struct FlyingDuck(Copyable, Flyable, Movable, Quackable):
    fn quack(self):
        print("Quack")

    fn fly(self):
        print("Whoosh!")


alias DuckLike = Quackable & Flyable


@fieldwise_init
struct ToyDuck(Copyable, DuckLike, Movable):
    fn quack(self):
        print("Squeak!")

    fn fly(self):
        print("Boing!")


def main():
    quack_and_go(FlyingDuck())
    quack_and_go(ToyDuck())
