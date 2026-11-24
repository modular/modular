# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for the `MODULAR_DRIVER_PLUGINS` grammar."""

from max._driver.max_license.hal.plugin_spec import (
    DriverPluginEntry,
    parse_driver_plugins,
)
from max._driver.max_license.hal.status import STATUS_INVALID_ARG
from std.testing import assert_equal, TestSuite


def _assert_entry(
    entry: DriverPluginEntry, name: StringSpan, path: StringSpan
) raises:
    assert_equal(entry.name, name)
    assert_equal(entry.path, path)


def test_multiple_labelled_entries() raises:
    var entries = parse_driver_plugins("CUDA@/a/cuda.so;CPU@/b/cpu.so")
    assert_equal(len(entries), 2)
    _assert_entry(entries[0], "CUDA", "/a/cuda.so")
    _assert_entry(entries[1], "CPU", "/b/cpu.so")


def test_bare_path_has_no_name() raises:
    var entries = parse_driver_plugins("/a/cpu.so;Metal@/b/metal.so")
    assert_equal(len(entries), 2)
    _assert_entry(entries[0], "", "/a/cpu.so")
    _assert_entry(entries[1], "Metal", "/b/metal.so")


def test_name_ends_at_first_at_sign() raises:
    var entries = parse_driver_plugins("Mock@/a/b@c.so")
    assert_equal(len(entries), 1)
    _assert_entry(entries[0], "Mock", "/a/b@c.so")


def test_whitespace_is_dropped() raises:
    var entries = parse_driver_plugins(" CUDA @ /a/cuda.so ;\t/b/cpu.so\n")
    assert_equal(len(entries), 2)
    _assert_entry(entries[0], "CUDA", "/a/cuda.so")
    _assert_entry(entries[1], "", "/b/cpu.so")


def test_empty_entries_are_dropped() raises:
    assert_equal(len(parse_driver_plugins("")), 0)
    assert_equal(len(parse_driver_plugins(" ; ;")), 0)
    var entries = parse_driver_plugins(";CUDA@/a/cuda.so;;Empty@;Blank@ ;")
    assert_equal(len(entries), 1)
    _assert_entry(entries[0], "CUDA", "/a/cuda.so")


def test_colon_does_not_separate_entries() raises:
    var entries = parse_driver_plugins("/a/cuda.so:/b/cpu.so")
    assert_equal(len(entries), 1)
    _assert_entry(entries[0], "", "/a/cuda.so:/b/cpu.so")


def test_entry_writes_its_spec() raises:
    var entries = parse_driver_plugins(" CUDA @ /a/cuda.so ; /b/cpu.so ")
    assert_equal(String(entries[0]), "CUDA@/a/cuda.so")
    assert_equal(String(entries[1]), "/b/cpu.so")


def test_parse_one_entry() raises:
    _assert_entry(
        DriverPluginEntry.parse(" Mock@/a/mock.so ;"), "Mock", "/a/mock.so"
    )


def test_parse_one_rejects_none_or_several() raises:
    for spec in ["", ";", "A@/a.so;B@/b.so"]:
        try:
            _ = DriverPluginEntry.parse(spec)
        except e:
            assert_equal(e.status, STATUS_INVALID_ARG)
            continue
        raise Error(String(t"'{spec}' parsed as one entry"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
