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

from linalg.utils_gpu import block_swizzle


# CHECK-LABEL: test_block_swizzle
def test_block_swizzle():
    print("=== test_block_swizzle")

    var grid_dim0 = (3, 4)

    # 3 blocks in N, no swizzle
    # CHECK: (0, 0) (1, 0) (2, 0)
    # CHECK: (0, 1) (1, 1) (2, 1)
    # CHECK: (0, 2) (1, 2) (2, 2)
    # CHECK: (0, 3) (1, 3) (2, 3)
    for j in range(grid_dim0[1]):
        for i in range(grid_dim0[0]):
            var block_idx = block_swizzle((i, j), grid_dim0)
            print(block_idx, end=" ")
        print()

    var grid_dim1 = (8, 2)

    # 8 blocks in N, 8 partitions
    # CHECK: (0, 0) (0, 1) (1, 0) (1, 1) (2, 0) (2, 1) (3, 0) (3, 1)
    # CHECK: (4, 0) (4, 1) (5, 0) (5, 1) (6, 0) (6, 1) (7, 0) (7, 1)
    for j in range(grid_dim1[1]):
        for i in range(grid_dim1[0]):
            var block_idx = block_swizzle((i, j), grid_dim1)
            print(block_idx, end=" ")
        print()

    var grid_dim2 = (4, 2)

    # 4 blocks in N, 4 partitions
    # CHECK: (0, 0) (0, 1) (1, 0) (1, 1)
    # CHECK: (2, 0) (2, 1) (3, 0) (3, 1)
    for j in range(grid_dim2[1]):
        for i in range(grid_dim2[0]):
            var block_idx = block_swizzle((i, j), grid_dim2)
            print(block_idx, end=" ")
        print()


# The SM90 and multistage matmul kernels swizzle 32-bit block coordinates.
# CHECK-LABEL: test_block_swizzle_uint32
def test_block_swizzle_uint32():
    print("=== test_block_swizzle_uint32")

    var grid_dim = (UInt32(8), UInt32(2))

    # CHECK: (0, 0) (0, 1) (1, 0) (1, 1) (2, 0) (2, 1) (3, 0) (3, 1)
    # CHECK: (4, 0) (4, 1) (5, 0) (5, 1) (6, 0) (6, 1) (7, 0) (7, 1)
    for j in range(Int(grid_dim[1])):
        for i in range(Int(grid_dim[0])):
            var block_idx = block_swizzle((UInt32(i), UInt32(j)), grid_dim)
            print(block_idx, end=" ")
        print()


def main():
    test_block_swizzle()
    test_block_swizzle_uint32()
