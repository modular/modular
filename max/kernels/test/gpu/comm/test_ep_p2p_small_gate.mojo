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

"""Pins `ep_p2p_small_on` at the boundaries of each bound.

`EPConfig.uses_p2p_small_path` mirrors this gate, so the graph sizes the
small path's buffers only when the kernels take it. The cases here are the
same ones `test_the_p2p_small_gate_matches_the_kernels_at_the_boundaries` in
`max/tests/tests/nn/test_ep_config.py` runs against the Python property; keep
the two tables in step.
"""

from shmem.ep import ep_p2p_small_on
from shmem.ep_comm import EP_P2P_SMALL_WORDS, EPLocalSyncCounters
from std.testing import assert_equal, TestSuite

comptime N_EXPERTS = 256
comptime COUNTER_WORDS = (
    EPLocalSyncCounters[N_EXPERTS].total_size() + EP_P2P_SMALL_WORDS
)


def _gate[
    n_experts: Int = N_EXPERTS,
    top_k: Int = 8,
    max_tokens: Int = 128,
    n_gpus: Int = 4,
    hidden_size: Int = 7168,
]() -> Bool:
    comptime words = EPLocalSyncCounters[n_experts].total_size() + (
        EP_P2P_SMALL_WORDS
    )
    return ep_p2p_small_on[
        n_experts,
        top_k,
        max_tokens,
        n_gpus,
        1,
        True,
        False,
        True,
        32,
        hidden_size,
        words,
    ]()


def test_base_config_is_on() raises:
    assert_equal(_gate(), True)


def test_token_bound() raises:
    assert_equal(_gate[max_tokens=32](), True)
    assert_equal(_gate[max_tokens=31](), False)


def test_local_expert_threads_bound() raises:
    # n_experts // n_gpus + 1 < 256
    assert_equal(_gate[n_experts=1016, n_gpus=4](), True)
    assert_equal(_gate[n_experts=1020, n_gpus=4](), False)


def test_combine_flag_bytes_bound() raises:
    # top_k * 32 * 4 <= 4096
    assert_equal(_gate[top_k=32](), True)
    assert_equal(_gate[top_k=33](), False)


def test_hidden_multiple_bound() raises:
    assert_equal(_gate[hidden_size=1024](), True)
    assert_equal(_gate[hidden_size=512](), False)
    assert_equal(_gate[hidden_size=7168 + 512](), False)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
