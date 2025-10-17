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


trait DefaultQuackable:
    fn quack(self):
        print("Quack")


@fieldwise_init
struct DefaultDuck(Copyable, DefaultQuackable, Movable):
    pass


fn make_it_quack(duck: DefaultDuck):
    duck.quack()


def main():
    make_it_quack(DefaultDuck())
