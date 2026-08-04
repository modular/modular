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


struct ParameterizedArray[ElementType: Copyable & Deinitable]:
    var data: Pointer[Self.ElementType, MutUntrackedOrigin]
    var size: Int

    def __init__(out self, var *elements: Self.ElementType):
        self.size = len(elements)
        self.data = alloc[Self.ElementType](self.size)
        for i in range(self.size):
            self.data.unsafe_offset(i).unsafe_write(elements[i].copy())

    def __deinit__(deinit self):
        for i in range(self.size):
            self.data.unsafe_offset(i).unsafe_deinit_pointee()
        self.data.unsafe_free()

    def __getitem__(self, i: Int) raises -> ref[self] Self.ElementType:
        if i < self.size:
            return self.data[unsafe_offset=i]
        else:
            raise Error("Out of bounds")


def main() raises:
    # start-generic-array-usage
    var array = ParameterizedArray(1, 2, 3)
    for i in range(array.size):
        var end = ", " if i < array.size - 1 else "\n"
        print(array[i], end=end)
    # end-generic-array-usage
