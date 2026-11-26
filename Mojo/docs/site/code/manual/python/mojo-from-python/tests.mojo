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
# tests.mojo
# Tests for the mojo-from-python.mdx claims that need Mojo-side machinery.
#
# Compiler verified: 2026-11-24, Mojo 1.3.0.dev2026112105
# Last updated: 2026-11-24
#
# `test.py` drives the page's modules from Python, the way a reader's program
# does. This file imports the same `fast_primes.mojo` and `fast_stats.mojo` as
# ordinary Mojo modules, so the code under test is the page's code, not a copy.
#
# Page claims, and where each is verified
#
#   1. `count_primes()` is ordinary Mojo: other Mojo code can call it, and you
#      can test it without Python. -> test_count_primes_is_ordinary_mojo
#   2. Two partial `RunningStats` results merge into one.
#      -> test_merge_equals_one_pass, test_merge_with_empty
#   3. `stddev()` is the sample standard deviation.
#      -> test_stddev_is_sample_stddev
#   4. Python prints a Mojo object through the struct's `Writable`
#      conformance. -> test_python_print_uses_writable
#   5. Registering the same struct a second time raises an error.
#      -> test_second_registration_raises
#
# Claims about calling these modules from Python are mapped in `test.py`.
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.testing import assert_almost_equal, assert_equal, assert_raises

from fast_primes import count_primes
from fast_stats import RunningStats

# --- Counting primes ---


def test_count_primes_is_ordinary_mojo() raises:
    assert_equal(count_primes(0), 0)
    assert_equal(count_primes(2), 0)
    assert_equal(count_primes(3), 1)
    assert_equal(count_primes(10), 4)
    assert_equal(count_primes(100), 25)
    assert_equal(count_primes(2_000_000), 148_933)


# --- Capturing running stats ---


def stats_of(values: List[Float64]) -> RunningStats:
    var stats = RunningStats()
    for value in values:
        stats.add(value)
    return stats^


def test_merge_equals_one_pass() raises:
    var whole = stats_of([12.5, 9.8, 14.1, 11.0, 30.2, 10.7])
    var merged = stats_of([12.5, 9.8, 14.1, 11.0])
    merged.merge(stats_of([30.2, 10.7]))

    assert_equal(merged.count, whole.count)
    assert_almost_equal(merged.mean, whole.mean)
    assert_almost_equal(merged.stddev(), whole.stddev())


def test_merge_with_empty() raises:
    var stats = stats_of([1.0, 2.0, 4.0])
    stats.merge(RunningStats())
    assert_equal(stats.count, 3)

    var empty = RunningStats()
    empty.merge(stats_of([1.0, 2.0, 4.0]))
    assert_equal(empty.count, 3)
    assert_almost_equal(empty.mean, 7.0 / 3.0)
    assert_almost_equal(empty.stddev(), stats.stddev())


def test_stddev_is_sample_stddev() raises:
    # Sample (n - 1) standard deviation of 2, 4, 4, 4, 5, 5, 7, 9 is
    # sqrt(32 / 7).
    var stats = stats_of([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert_almost_equal(stats.stddev(), (32.0 / 7.0) ** 0.5)
    # Fewer than two samples have no spread.
    assert_equal(stats_of([5.0]).stddev(), 0.0)


# --- Bridge Python to your Mojo types ---


def test_python_print_uses_writable() raises:
    var stats = stats_of([1.0, 3.0])
    var expected = repr(stats)
    var py_stats = PythonObject(alloc=stats^)
    assert_equal(String(py=Python.str(py_stats)), expected)


def test_second_registration_raises() raises:
    # `main()` already registered `RunningStats`. The duplicate shows up when
    # the second module finalizes.
    var again = PythonModuleBuilder("fast_stats_again")
    _ = again.add_type[RunningStats]("RunningStats")
    with assert_raises(contains="multiple Python type objects"):
        _ = again.finalize()


def main() raises:
    # Registers `RunningStats` once, as `import fast_stats` would.
    var builder = PythonModuleBuilder("fast_stats")
    _ = builder.add_type[RunningStats]("RunningStats")
    _ = builder.finalize()

    test_count_primes_is_ordinary_mojo()
    test_merge_equals_one_pass()
    test_merge_with_empty()
    test_stddev_is_sample_stddev()
    test_python_print_uses_writable()
    test_second_registration_raises()
    print("all checks passed")
