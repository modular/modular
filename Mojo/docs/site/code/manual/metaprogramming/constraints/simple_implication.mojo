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


# start-simple-implication
def create_list[T: Copyable, size: Int]() -> List[T] where size >= 0:
    return List[T](capacity=size)


def create_even_list[
    T: Copyable, size: Int
]() -> List[T] where size >= 0 and (size / 2) * 2 == size:
    # No need to individually prove `size >= 0` — it's part of the
    # `and` above, so the call to `create_list` type-checks.
    return create_list[T, size=size]()


def main():
    var l1 = create_even_list[Int, 4]()
    print(len(l1) % 2)
    # Prints 0, since the list has an even number of elements.
    # end-simple-implication
