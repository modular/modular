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
from gpu import global_idx


# CHECK-LABEL: test_fail
def main():
    print("== test_fail")
    # CHECK: block: [0,0,0] thread: [0,0,0] Assert Error: forcing failure on thread: 0 with multiple args: 2.0 True
    with DeviceContext() as ctx:

        fn fail_assert():
            debug_assert(
                False,
                "forcing failure on thread: ",
                global_idx.x,
                " with multiple args: ",
                Float64(2.0),
                " ",
                True,
            )
            # CHECK-NOT: won't print this due to assert failure
            print("won't print this due to assert failure")

        ctx.enqueue_function[fail_assert](
            grid_dim=2,
            block_dim=(2, 2),
        )

        ctx.synchronize()
