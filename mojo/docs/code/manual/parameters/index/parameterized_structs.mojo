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


struct GenericArray[ElementType: Copyable & Movable]:
    var data: UnsafePointer[ElementType]
    var size: Int

    fn __init__(out self, var *elements: ElementType):
        self.size = len(elements)
        self.data = UnsafePointer[ElementType].alloc(self.size)
        for i in range(self.size):
            (self.data + i).init_pointee_move(elements[i].copy())

    fn __del__(deinit self):
        for i in range(self.size):
            (self.data + i).destroy_pointee()
        self.data.free()

    fn __getitem__(self, i: Int) raises -> ref [self] ElementType:
        if i < self.size:
            return self.data[i]
        else:
            raise Error("Out of bounds")


def main():
    var array = GenericArray(1, 2, 3)
    for i in range(array.size):
        end = ", " if i < array.size - 1 else "\n"
        print(array[i], end=end)
