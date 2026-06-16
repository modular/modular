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

from std.bit import byte_swap


# start-read-chunks
def read_chunks(
    var ptr: UnsafePointer[mut=False, UInt8, _],
) -> List[List[UInt32]]:
    var chunks = List[List[UInt32]]()
    # A chunk size of 0 indicates the end of the data
    var chunk_size = Int(ptr[])
    while chunk_size > 0:
        # Skip the 1 byte chunk_size and get a pointer to the first
        # UInt32 in the chunk
        var ui32_ptr = (ptr + 1).bitcast[UInt32]()
        var chunk = List[UInt32](capacity=chunk_size)
        for i in range(chunk_size):
            chunk.append(ui32_ptr[i])
        # list is not implicitly copyable, so it needs the transfer sigil (^)
        chunks.append(chunk^)
        # Move our pointer to the next byte after the current chunk
        ptr += 1 + 4 * chunk_size
        # Read the size of the next chunk
        chunk_size = Int(ptr[])
    return chunks^


# end-read-chunks


def main():
    # Build a test buffer: 2 chunks.
    # Chunk 1: size=2, values=[10, 20] (as UInt32 little-endian)
    # Chunk 2: size=1, values=[30]
    # Terminator: size=0
    var buf = alloc[UInt8](1 + 8 + 1 + 4 + 1)
    var offset = 0

    # Chunk 1 header
    (buf + offset).init_pointee_copy(UInt8(2))
    offset += 1
    # Chunk 1 values (little-endian UInt32)
    (buf + offset).bitcast[UInt32]().init_pointee_copy(UInt32(10))
    offset += 4
    (buf + offset).bitcast[UInt32]().init_pointee_copy(UInt32(20))
    offset += 4

    # Chunk 2 header
    (buf + offset).init_pointee_copy(UInt8(1))
    offset += 1
    # Chunk 2 value
    (buf + offset).bitcast[UInt32]().init_pointee_copy(UInt32(30))
    offset += 4

    # Terminator
    (buf + offset).init_pointee_copy(UInt8(0))

    var result = read_chunks(buf)
    print(len(result))  # 2
    print(result[0][0])  # 10
    print(result[0][1])  # 20
    print(result[1][0])  # 30

    buf.free()
