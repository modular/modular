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


def main():
    # start-returned-references-no-copy
    animals: List[String] = ["Cats", "Dogs", "Zebras"]
    print(animals[2])  # Prints "Zebras", does not copy the value.
    # end-returned-references-no-copy

    # start-returned-references-implicit-copy
    items = [99, 77, 33, 12]
    item = items[1]  # item is a copy of items[1]
    item += 1  # increments item
    print(items[1])  # prints 77
    # end-returned-references-implicit-copy

    # start-reference-binding
    ref item_ref = items[1]  # item_ref is a reference to item[1]
    item_ref += 1  # increments items[1]
    print(items[1])  # prints 78
    # end-reference-binding

    # start-reference-binding-error
    # ref item_ref = items[2]
    # end-reference-binding-error
