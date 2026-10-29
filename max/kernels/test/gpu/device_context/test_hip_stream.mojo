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
"""`HIP(stream)` returns the live `hipStream_t` behind a `DeviceStream`."""

from std.ffi import OwnedDLHandle, c_int

from max.gpu.host import HIP, DeviceContext
from std.testing import assert_equal, assert_true


def test_hip_stream_is_a_live_stream(ctx: DeviceContext) raises:
    var stream = HIP(ctx.stream())
    # A null handle names HIP's default stream, not the context's stream.
    assert_true(Bool(stream))
    assert_equal(HIP(ctx.stream()), stream)
    # The runtime rejects anything that is not a live stream.
    var libhip = OwnedDLHandle("libamdhip64.so")
    assert_equal(libhip.get_function[c_int]("hipStreamSynchronize")(stream), 0)


def main() raises:
    with DeviceContext() as ctx:
        test_hip_stream_is_a_live_stream(ctx)
