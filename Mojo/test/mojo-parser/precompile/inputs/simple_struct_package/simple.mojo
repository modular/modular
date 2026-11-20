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
struct PlainStruct:
    pass


struct GenericBox[T: ImplicitlyCopyable & Deinitable]:
    var value: Self.T

    def __init__(out self, value: Self.T):
        self.value = value

    def get(self) -> Self.T:
        return self.value


struct MyStruct(Copyable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    def accelerate(mut self):
        self.value = 10
