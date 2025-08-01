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
from gpu.id import block_idx, thread_idx
from sys import has_accelerator


fn print_threads():
    """Print thread IDs."""

    print(
        "Block index: [",
        block_idx.x,
        block_idx.y,
        block_idx.z,
        "]\tThread index: [",
        thread_idx.x,
        thread_idx.y,
        thread_idx.z,
        "]",
    )


def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        ctx.enqueue_function[print_threads](
            grid_dim=(2, 2, 1), block_dim=(16, 4, 2)
        )
        ctx.synchronize()
        print("Program finished")
