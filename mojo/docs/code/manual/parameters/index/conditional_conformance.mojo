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
struct Container[ElementType: Movable]:
    var element: ElementType

    def __str__[
        StrElementType: Writable & Copyable & Movable, //
    ](self: Container[StrElementType]) -> String:
        return String(self.element)


def main():
    float_container = Container(5.0)
    string_container = Container("Hello")
    print(float_container.__str__())
    print(string_container.__str__())
