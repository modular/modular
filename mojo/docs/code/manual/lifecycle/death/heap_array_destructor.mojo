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


struct HeapArray(Writable):
    var data: UnsafePointer[Int]
    var size: Int

    fn __init__(out self, *values: Int):
        self.size = len(values)
        self.data = UnsafePointer[Int].alloc(self.size)
        for i in range(self.size):
            (self.data + i).init_pointee_copy(values[i])

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("[")
        for i in range(self.size):
            writer.write(self.data[i])
            if i < self.size - 1:
                writer.write(", ")
        writer.write("]")

    fn __del__(deinit self):
        print("Destroying", self.size, "elements")
        for i in range(self.size):
            (self.data + i).destroy_pointee()
        self.data.free()


def main():
    var a = HeapArray(10, 1, 3, 9)
    print(a)
