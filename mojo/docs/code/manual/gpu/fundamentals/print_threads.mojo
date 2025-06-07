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

from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, global_idx, grid_dim, thread_idx
from sys import exit, has_accelerator


fn print_threads():
    """Print thread block and thread indices."""

    print(
        "block_idx: [",
        block_idx.x,
        block_idx.y,
        block_idx.z,
        "]\tthread_idx: [",
        thread_idx.x,
        thread_idx.y,
        thread_idx.z,
        "]\tglobal_idx: [",
        global_idx.x,
        global_idx.y,
        global_idx.z,
        "]\tcalculated global_idx: [",
        block_dim.x * block_idx.x + thread_idx.x,
        block_dim.y * block_idx.y + thread_idx.y,
        block_dim.z * block_idx.z + thread_idx.z,
        "]",
    )


def main():
    @parameter
    if not has_accelerator():
        print("No GPU detected")
        exit(0)
    else:
        # Initialize GPU context for device 0 (default GPU device).
        ctx = DeviceContext()

        ctx.enqueue_function[print_threads](
            grid_dim=(2, 2, 1),  # 2x2x1 blocks per grid
            block_dim=(4, 4, 2),  # 4x4x2 threads per block
        )

        ctx.synchronize()
        print("Done")
