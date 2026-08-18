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
from std.memory import alloc, dealloc, ThinAllocation


struct HeapArray(Writable):
    var data: ThinAllocation[Int]
    var size: Int

    def __init__(out self, *values: Int):
        self.size = len(values)
        self.data = alloc[Int]({count = self.size}).into_thin()
        var ptr = self.data.unsafe_ptr()
        for i in range(self.size):
            ptr.unsafe_offset(i).unsafe_write(values[i])

    def write_to(self, mut writer: Some[Writer]):
        writer.write("[")
        var ptr = self.data.unsafe_ptr()
        for i in range(self.size):
            writer.write(ptr[unsafe_offset=i])
            if i < self.size - 1:
                writer.write(", ")
        writer.write("]")

    def __deinit__(deinit self):
        print("Destroying", self.size, "elements")
        var ptr = self.data.unsafe_ptr()
        for i in range(self.size):
            ptr.unsafe_offset(i).unsafe_deinit_pointee()
        dealloc(self.data^.unsafe_with_layout({count = self.size}))


def main():
    var a = HeapArray(10, 1, 3, 9)
    print(a)
