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

# Tested on T4 GPU 2 Dec 2025
# DOC: mojo/tools/notebooks.mdx

from gpu.host import DeviceContext


fn kernel():
    print("Hello from the GPU")


def main():
    # Launch GPU kernel
    with DeviceContext() as ctx:
        ctx.enqueue_function_checked[kernel, kernel](grid_dim=1, block_dim=1)
        ctx.synchronize()
