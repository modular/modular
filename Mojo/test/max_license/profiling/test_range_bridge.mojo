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

from std.ffi import external_call
from std.testing import assert_equal


def test_range_bridge_symbols_exist_and_disabled_by_default() raises:
    # Profiler starts disabled; KGEN_CompilerRT_RangeIsEnabled must return 0.
    var enabled = external_call["KGEN_CompilerRT_RangeIsEnabled", Int]()
    assert_equal(0, enabled)
    # No trace is live either; KGEN_CompilerRT_RangeIsRecording must return 0.
    var recording = external_call["KGEN_CompilerRT_RangeIsRecording", Int]()
    assert_equal(0, recording)


def test_range_bridge_recording_distinct_from_enabled() raises:
    # IsEnabled reflects enable *intent*; IsRecording reflects a live trace.
    # With no profiler backend available in this test environment, enable()
    # must flip IsEnabled but can never raise IsRecording — the two
    # predicates must not be conflated. Cross-copy gate visibility (the
    # process-global gate in libMSupportGlobals.so) is exercised for real by
    # the GPU trace-routing test in max/tests/internal/profiler/.
    external_call["KGEN_CompilerRT_RangeEnable", NoneType]()
    try:
        assert_equal(1, external_call["KGEN_CompilerRT_RangeIsEnabled", Int]())
        assert_equal(
            0, external_call["KGEN_CompilerRT_RangeIsRecording", Int]()
        )
    finally:
        external_call["KGEN_CompilerRT_RangeDisable", NoneType]()
    assert_equal(0, external_call["KGEN_CompilerRT_RangeIsRecording", Int]())


def test_range_bridge_enable_disable_round_trip() raises:
    # Round-trip enable() → IsEnabled == 1 → disable() → IsEnabled == 0. The
    # try/finally guarantees disable() runs even if the enabled-state
    # assertion fails, so a regression here cannot leak enabled state into
    # other tests sharing this process.
    external_call["KGEN_CompilerRT_RangeEnable", NoneType]()
    try:
        var enabled = external_call["KGEN_CompilerRT_RangeIsEnabled", Int]()
        assert_equal(1, enabled)
    finally:
        external_call["KGEN_CompilerRT_RangeDisable", NoneType]()
    var enabled = external_call["KGEN_CompilerRT_RangeIsEnabled", Int]()
    assert_equal(0, enabled)


def test_range_bridge_begin_end_disabled_is_safe() raises:
    # Calling begin/end while disabled must not crash. This exercises the
    # hot path that production code will hit when profiling is off.
    var name = "disabled-range"
    external_call["KGEN_CompilerRT_RangeBegin", NoneType](
        name.unsafe_ptr(), name.byte_length(), UInt32(0)
    )
    external_call["KGEN_CompilerRT_RangeEnd", NoneType]()
    var enabled = external_call["KGEN_CompilerRT_RangeIsEnabled", Int]()
    assert_equal(0, enabled)


def test_range_bridge_step_disabled_is_safe() raises:
    # step() is called every iteration of Model::execute(); must be safe
    # when disabled.
    for _ in range(1000):
        external_call["KGEN_CompilerRT_RangeStep", NoneType]()
    var enabled = external_call["KGEN_CompilerRT_RangeIsEnabled", Int]()
    assert_equal(0, enabled)


def main() raises:
    test_range_bridge_symbols_exist_and_disabled_by_default()
    test_range_bridge_begin_end_disabled_is_safe()
    test_range_bridge_step_disabled_is_safe()
    test_range_bridge_enable_disable_round_trip()
    test_range_bridge_recording_distinct_from_enabled()
