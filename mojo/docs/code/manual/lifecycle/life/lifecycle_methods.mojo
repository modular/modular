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

# ===----------------------------------------------------------------------=== #
# This example shows the hand-written equivalents of the lifecycle methods
# that Mojo synthesizes for the struct in lifecycle_methods_sythensized
# ===----------------------------------------------------------------------=== #


struct MyPet:
    var name: String
    var age: Int

    fn __init__(out self, var name: String, age: Int):
        self.name = name^
        self.age = age

    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
        self.age = existing.age

    fn __moveinit__(out self, deinit existing: Self):
        self.name = existing.name^
        self.age = existing.age

    fn copy(self) -> Self:
        return Self(self.name, self.age)


def main():
    pet = MyPet("Fido", 3)
    pet2 = pet.copy()
    print(pet2.name)
    print(pet2.age)
    pet3 = pet
    print(pet3.name)
    pet4 = pet^
    print(pet4.name)
