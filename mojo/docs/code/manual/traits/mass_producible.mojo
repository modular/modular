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

# start-trait-lifecycle-methods
# trait Defaultable
#     fn __init__(out self): ...

# trait Movable
#     fn __moveinit__(out self, deinit existing: Self):

alias MassProducible = Defaultable & Movable


fn factory[type: MassProducible]() -> type:
    return type()


struct Thing(MassProducible):
    var id: Int

    fn __init__(out self):
        self.id = 0

    fn __moveinit__(out self, deinit existing: Self):
        self.id = existing.id


def main():
    var thing = factory[Thing]()
    # end-trait-lifecycle-methods
    _ = thing^
