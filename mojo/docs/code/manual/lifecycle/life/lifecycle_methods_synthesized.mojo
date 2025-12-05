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


@fieldwise_init
struct MyPet(ImplicitlyCopyable):
    var name: String
    var age: Int


@fieldwise_init
struct Pair(ImplicitlyCopyable):
    var first: Int
    var second: Int


def main():
    pet = MyPet("Fido", 3)
    pet2 = pet
    print(pet2.name)
    print(pet2.age)
    pet3 = pet.copy()
    print(pet3.name)
    pet4 = pet^
    print(pet4.name)

    pair = Pair(3, 4)
    pair_copy = pair
    print(pair.first, pair_copy.second)
