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
from std.testing import assert_equal


struct HeapArray(Copyable, Writable):
    var data: ThinAllocation[Int]
    var size: Int
    var cap: Int

    def __init__(out self, size: Int, val: Int):
        self.size = size
        self.cap = size * 2
        self.data = alloc[Int]({count = self.cap}).into_thin()
        var ptr = self.data.unsafe_ptr()
        for i in range(self.size):
            ptr.unsafe_offset(i).unsafe_write(val)

    def __init__(out self, *, copy: Self):
        # Deep-copy the existing value
        self.size = copy.size
        self.cap = copy.cap
        self.data = alloc[Int]({count = self.cap}).into_thin()
        var ptr = self.data.unsafe_ptr()
        for i in range(self.size):
            var copy_ptr = copy.data.unsafe_ptr()
            ptr.unsafe_offset(i).unsafe_write(copy_ptr[unsafe_offset=i])
        # The lifetime of `copy` continues unchanged

    def __deinit__(deinit self):
        # We must free the heap-allocated data, but
        # Mojo knows how to destroy the other fields
        var ptr = self.data.unsafe_ptr()
        for i in range(self.size):
            ptr.unsafe_offset(i).unsafe_deinit_pointee()
        dealloc(self.data^.unsafe_with_layout({count = self.cap}))

    def append(mut self, val: Int):
        # Update the array for demo purposes
        if self.size < self.cap:
            var ptr = self.data.unsafe_ptr()
            ptr.unsafe_offset(self.size).unsafe_write(val)
            self.size += 1
        else:
            print("Out of bounds")

    def write_to(self, mut writer: Some[Writer]):
        writer.write("[")
        var ptr = self.data.unsafe_ptr()
        for i in range(self.size):
            writer.write(ptr[unsafe_offset=i])
            if i < self.size - 1:
                writer.write(", ")
        writer.write("]")


def copies() raises:
    var a = HeapArray(2, 1)
    var b = a.copy()  # Calls the copy method
    # print(a)  # Prints [1, 1]
    # print(b)  # Prints [1, 1]
    assert_equal("[1, 1]", String(a))
    assert_equal("[1, 1]", String(b))

    b.append(2)  # Changes the copied data
    # print(b)   # Prints [1, 1, 2]
    # print(a)   # Prints [1, 1] (the original did not change)
    assert_equal("[1, 1, 2]", String(b))
    assert_equal("[1, 1]", String(a))


def main() raises:
    copies()
