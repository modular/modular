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
"""The EP combine receive buffer's linear offset past 2^31 and past 2^32.

`recv_buf_layout` is a comptime layout over `(max_tokens_per_rank, top_k,
msg_bytes)`, so a buffer past 2^31 bytes forms offsets a 32-bit index cannot
hold; each is compared here with the same product in `Int`.
"""

from std.math import divmod
from std.testing import assert_equal

from shmem.ep_comm import EPCombineKernel

comptime TOP_K = 4
comptime MSG = 1024
# With top_k and msg_bytes this makes the receive buffer 2^33 bytes, so both
# boundaries below are inside it.
comptime MAX_T = 1 << 21
comptime N_RANKS = 2

comptime Combine = EPCombineKernel[
    num_threads=128,
    n_sms=148,
    top_k=TOP_K,
    n_experts=256,
    n_ranks=N_RANKS,
    msg_bytes=MSG,
    max_tokens_per_rank=MAX_T,
    p2p_world_size=2,
]


def _check(offset: Int, label: String) raises:
    """The layout at the coordinate holding `offset` equals it in `Int`."""
    comptime per_token = TOP_K * MSG
    var t, rem = divmod(offset, per_token)
    var k, b = divmod(rem, MSG)
    var got = Int(Combine.recv_buf_layout((t, k, b)))
    var want = (t * TOP_K + k) * MSG + b
    assert_equal(want, offset)
    assert_equal(got, want)
    print(label, " offset=", offset, " layout=", got, " int=", want)


def main() raises:
    _check((1 << 31) + 4096, "past 2^31")
    _check((1 << 32) + 4096, "past 2^32")
    print("EP recv offset: PASS")
