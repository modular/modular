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


@fieldwise_init
struct Duck(Copyable, Movable, Quackable):
    fn quack(self):
        print("Quack")


@fieldwise_init
struct StealthCow(Copyable, Movable, Quackable):
    fn quack(self):
        print("Moo!")


fn make_it_quack[type: Quackable](maybe_a_duck: type):
    maybe_a_duck.quack()


def main():
    make_it_quack(Duck())
    make_it_quack(StealthCow())
