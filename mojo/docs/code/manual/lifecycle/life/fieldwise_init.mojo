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


# NOTE: This exact example is not currently included in the docs.
# start-fieldwise-init-example
@fieldwise_init
struct MyPet(Copyable, Movable):
    var name: String
    var age: Int

    fn __init__(out self, var name: String):
        self.name = name^
        self.age = 0


def main():
    spot = MyPet("Spot")  # Use new constructor
    willow = MyPet("Willow", 4)  # Use the field-wise constructor
    # end-fieldwise-init-example
    _ = spot^
    _ = willow^
