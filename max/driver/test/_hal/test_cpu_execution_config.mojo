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

"""Tests for `CPUExecutionConfiguration`.

The type is tracked as a per-thread bitmask with cores modeled as contiguous
groups of `MAX_THREADS_PER_CORE` threads. The interesting (and off-by-one-prone)
logic is the thread<->core derivation, so these tests focus there rather than
on the trivial passthrough getters/setters.
"""

from std.collections import BitSet
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from _hal.execution_config.cpu import (
    CPUExecutionConfiguration,
)

comptime TPC = CPUExecutionConfiguration.MAX_THREADS_PER_CORE
comptime MAX_CORES = CPUExecutionConfiguration.MAX_CORE_COUNT
comptime MAX_THREADS = CPUExecutionConfiguration.MAX_THREAD_COUNT


def _num_threads(config: CPUExecutionConfiguration) -> Int:
    return config.get_num_threads()


def _num_cores(config: CPUExecutionConfiguration) -> Int:
    return config.get_num_cores()


def test_num_cores_ctor_enables_whole_cores() raises:
    """`num_cores=N` enables every thread of the first N cores."""
    var config = CPUExecutionConfiguration(num_cores=3)
    assert_equal(_num_cores(config), 3)
    assert_equal(_num_threads(config), 3 * TPC)

    # The first three cores' threads are set; the fourth core's are not.
    for thread in range(3 * TPC):
        assert_true(config.thread_mask.test(thread))
    assert_false(config.thread_mask.test(3 * TPC))

    var core_mask = config.get_core_mask()
    assert_true(core_mask.test(0))
    assert_true(core_mask.test(2))
    assert_false(core_mask.test(3))


def test_threads_per_core_ctor_is_sparse_within_cores() raises:
    """`num_cores`/`num_threads_per_core` leaves gaps inside each core."""
    var config = CPUExecutionConfiguration(num_cores=2, num_threads_per_core=2)
    assert_equal(_num_threads(config), 2 * 2)
    assert_equal(_num_cores(config), 2)

    # Core 0: threads 0..1 set, 2..(TPC-1) clear.
    for thread in range(2):
        assert_true(config.thread_mask.test(thread))
    for thread in range(2, TPC):
        assert_false(config.thread_mask.test(thread))

    # Core 1: threads TPC..TPC+1 set, the rest of the core clear.
    for thread in range(TPC, TPC + 2):
        assert_true(config.thread_mask.test(thread))
    assert_false(config.thread_mask.test(TPC + 2))


def test_threads_per_core_is_clamped() raises:
    """`num_threads_per_core` above `MAX_THREADS_PER_CORE` never bleeds into the
    next core."""
    var config = CPUExecutionConfiguration(
        num_cores=1, num_threads_per_core=TPC + 100
    )
    assert_equal(_num_threads(config), TPC)
    assert_equal(_num_cores(config), 1)
    # The next core stays empty despite the oversized request.
    assert_false(config.thread_mask.test(TPC))


def test_core_mask_ctor_maps_bit_to_thread_window() raises:
    """A single set core bit maps to that core's contiguous thread window."""
    var core_mask = BitSet[MAX_CORES]()
    core_mask.set(5)
    var config = CPUExecutionConfiguration(core_mask=core_mask)

    assert_equal(_num_cores(config), 1)
    assert_equal(_num_threads(config), TPC)
    for thread in range(5 * TPC, 6 * TPC):
        assert_true(config.thread_mask.test(thread))
    assert_false(config.thread_mask.test(5 * TPC - 1))
    assert_false(config.thread_mask.test(6 * TPC))

    var derived = config.get_core_mask()
    assert_true(derived.test(5))
    assert_equal(len(derived), 1)


def test_partial_core_still_counts_as_enabled() raises:
    """A core with even one active thread is reported as an enabled core."""
    # num_threads spanning 1.x cores: core 0 full, core 1 partially filled.
    var config = CPUExecutionConfiguration(num_threads=TPC + 1)
    assert_equal(_num_threads(config), TPC + 1)
    assert_equal(_num_cores(config), 2)

    var core_mask = config.get_core_mask()
    assert_true(core_mask.test(0))
    assert_true(core_mask.test(1))
    assert_false(core_mask.test(2))


def test_set_num_threads_per_core_rederives_active_cores() raises:
    """`set_num_threads_per_core` reshapes each currently-active core."""
    var config = CPUExecutionConfiguration(num_cores=3)
    config.set_num_threads_per_core(2)

    # Still three active cores, now two threads each.
    assert_equal(_num_cores(config), 3)
    assert_equal(_num_threads(config), 3 * 2)
    for core in range(3):
        var base = core * TPC
        assert_true(config.thread_mask.test(base))
        assert_true(config.thread_mask.test(base + 1))


def test_set_num_cores_and_threads() raises:
    """`set_num_cores` / `set_num_threads` overwrite the mask by prefix."""
    var config = CPUExecutionConfiguration(num_cores=1)

    config.set_num_cores(5)
    assert_equal(_num_cores(config), 5)
    assert_equal(_num_threads(config), 5 * TPC)

    config.set_num_threads(7)
    assert_equal(_num_threads(config), 7)
    for thread in range(7):
        assert_true(config.thread_mask.test(thread))
    assert_false(config.thread_mask.test(7))


def test_request_all() raises:
    """`request_all_threads` / `request_all_cores` fill the whole mask."""
    var config = CPUExecutionConfiguration(num_cores=1)

    config.request_all_threads()
    assert_equal(_num_threads(config), MAX_THREADS)
    assert_equal(_num_cores(config), MAX_CORES)

    config.set_num_threads(0)
    assert_equal(_num_threads(config), 0)
    assert_equal(_num_cores(config), 0)

    config.request_all_cores()
    assert_equal(_num_threads(config), MAX_THREADS)
    assert_equal(_num_cores(config), MAX_CORES)


def test_thread_mask_roundtrip_and_options() raises:
    """`get`/`set_thread_mask` round-trip and the unit/scratchpad options
    stick."""
    var mask = BitSet[MAX_THREADS]()
    mask.set(0)
    mask.set(100)
    mask.set(383)
    var config = CPUExecutionConfiguration(thread_mask=mask)

    var got = config.get_thread_mask()
    assert_equal(len(got), 3)
    assert_true(got.test(0))
    assert_true(got.test(100))
    assert_true(got.test(383))

    # Defaults, then flips.
    assert_true(config.get_matrix_unit_usage())

    config.set_matrix_unit_usage(False)
    assert_false(config.get_matrix_unit_usage())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
