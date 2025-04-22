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
# RUN: %mojo-no-debug %s | FileCheck %s

from linalg.utils import SubMatmulConfig, get_partitioned_matmul_mojo

alias kernel_rows = 6
alias kernel_cols = 64
alias b_type = DType.float32


# CHECK-LABEL: test_partition
fn test_partition():
    print("== test_partition")
    # Matmul dimensions
    var M = 4
    var N = 768
    var K = 3072

    var num_tasks = 8

    var config: SubMatmulConfig

    config = get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
        M, N, K, 0, num_tasks
    )
    # CHECK: (0, 0, 0)
    print(config.offset)
    # CHECK: (4, 128, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
        M, N, K, 1, num_tasks
    )
    # CHECK: (0, 128, 0)
    print(config.offset)
    # CHECK: (4, 128, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
        M, N, K, 2, num_tasks
    )
    # CHECK: (0, 256, 0)
    print(config.offset)
    # CHECK: (4, 128, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
        M, N, K, 3, num_tasks
    )
    # CHECK: (0, 384, 0)
    print(config.offset)
    # CHECK: (4, 128, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
        M, N, K, 4, num_tasks
    )
    # CHECK: (0, 512, 0)
    print(config.offset)
    # CHECK: (4, 64, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
        M, N, K, 5, num_tasks
    )
    # CHECK: (0, 576, 0)
    print(config.offset)
    # CHECK: (4, 64, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
        M, N, K, 6, num_tasks
    )
    # CHECK: (0, 640, 0)
    print(config.offset)
    # CHECK: (4, 64, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
        M, N, K, 7, num_tasks
    )
    # CHECK: (0, 704, 0)
    print(config.offset)
    # CHECK: (4, 64, 3072)
    print(config.shape)


fn main():
    test_partition()
