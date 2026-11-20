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

from max.gpu.host import DeviceContext
from std.testing import assert_equal


def main() raises:
    with DeviceContext() as ctx:
        if not ctx.supports_flag_wait():
            print("Skipping: no flag waits on", ctx.api())
            return
        var flag = ctx.allocate_completion_flag()
        var stream = ctx.stream()

        # Already past the target.
        flag.signal(5)
        stream.wait_for_flag(flag, 3)
        stream.synchronize()

        # Gated until the host signals.
        stream.wait_for_flag(flag, 9)
        flag.signal(9)
        stream.synchronize()
        assert_equal(flag.load(), 9)

        flag.reset()
        assert_equal(flag.load(), 0)
