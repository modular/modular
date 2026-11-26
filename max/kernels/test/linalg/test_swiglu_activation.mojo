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
"""Host-side tests for the clamp order of the gated SiLU activations."""

from std.math import exp
from std.testing import assert_almost_equal, TestSuite

from linalg.swiglu_activation import get_swiglu_activation

comptime LIMIT = Float32(4.0)
comptime pre_clamp = get_swiglu_activation["silu", limit=LIMIT]
comptime post_clamp = get_swiglu_activation["silu_post_clamp", limit=LIMIT]

# The activations use an approximate reciprocal on some targets.
comptime RTOL = 1e-3


def _silu(x: Float32) -> Float32:
    return x / (1.0 + exp(-x))


def test_post_clamp_saturates_the_gate_at_the_limit() raises:
    # silu(4) is about 2% under 4, so the two clamp orders differ past the
    # limit by more than the tolerance.
    assert_almost_equal(post_clamp(8.0, 1.0), LIMIT, rtol=RTOL)
    assert_almost_equal(pre_clamp(8.0, 1.0), _silu(LIMIT), rtol=RTOL)


def test_post_clamp_keeps_silu_between_the_two_bounds() raises:
    # silu(4.05) is under the limit, though the gate itself is over it.
    assert_almost_equal(post_clamp(4.05, 1.0), _silu(4.05), rtol=RTOL)


def test_post_clamp_matches_silu_under_the_limit() raises:
    for gate in [Float32(-6.0), Float32(-1.0), Float32(0.5), Float32(3.0)]:
        assert_almost_equal(post_clamp(gate, 2.0), _silu(gate) * 2.0, rtol=RTOL)


def test_post_clamp_clamps_up_on_both_sides() raises:
    assert_almost_equal(post_clamp(1.0, 9.0), _silu(1.0) * LIMIT, rtol=RTOL)
    assert_almost_equal(post_clamp(1.0, -9.0), -_silu(1.0) * LIMIT, rtol=RTOL)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
