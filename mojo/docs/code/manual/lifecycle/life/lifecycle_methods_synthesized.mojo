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


@fieldwise_init
struct MyPet(ImplicitlyCopyable):
    var name: String
    var age: Int


@fieldwise_init
struct Pair(ImplicitlyCopyable):
    var first: Int
    var second: Int


def main():
    var pet = MyPet("Fido", 3)
    var pet2 = pet
    print(pet2.name)
    print(pet2.age)
    var pet3 = pet.copy()
    print(pet3.name)
    var pet4 = pet^
    print(pet4.name)

    var pair = Pair(3, 4)
    var pair_copy = pair
    print(pair.first, pair_copy.second)
