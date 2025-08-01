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


trait Animal:
    fn make_sound(self):
        ...


# Bird inherits from Animal
trait Bird(Animal):
    fn fly(self):
        ...


trait Named:
    fn get_name(self) -> String:
        ...


trait NamedAnimal(Animal, Named):
    fn emit_name_and_sound(self):
        ...


@fieldwise_init
struct Parrot(Bird, Copyable, Movable, NamedAnimal):
    fn make_sound(self):
        print("Squawk!")

    fn fly(self):
        print("Flap flap!")

    fn get_name(self) -> String:
        return "Parrot"

    fn emit_name_and_sound(self):
        print("The", self.get_name(), "says ", end="")
        self.make_sound()


def main():
    parrot = Parrot()
    parrot.make_sound()
    parrot.fly()
    parrot.emit_name_and_sound()
