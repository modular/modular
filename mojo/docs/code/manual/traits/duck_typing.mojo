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


trait Quackable:
    def quack(self):
        ...


@fieldwise_init
struct Duck(Copyable, Quackable):
    def quack(self):
        print("Quack")


@fieldwise_init
struct StealthCow(Copyable, Quackable):
    def quack(self):
        print("Moo!")


def make_it_quack[DuckType: Quackable](maybe_a_duck: DuckType):
    maybe_a_duck.quack()


def make_it_quack2(maybe_a_duck: Some[Quackable]):
    maybe_a_duck.quack()


def take_two_quackers[
    DuckType: Quackable
](quacker1: DuckType, quacker2: DuckType):
    pass


def main():
    make_it_quack(Duck())
    make_it_quack(StealthCow())
    make_it_quack2(Duck())
    make_it_quack2(StealthCow())
