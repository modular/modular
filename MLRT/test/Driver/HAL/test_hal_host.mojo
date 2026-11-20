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
"""Tests which plugin `Host` loads from a multi-entry `MODULAR_DRIVER_PLUGINS`.

The BUILD target sets `CPU_PLUGIN` and `MOCK_PLUGIN` to the plugin paths.
"""

from std.os import getenv, setenv

from max._driver import Host
from max._driver.max_license.hal.status import HALError, STATUS_INVALID_ARG
from std.testing import assert_equal, assert_true


def _host_driver_name(plugins_spec: String) raises -> String:
    _ = setenv("MODULAR_DRIVER_PLUGINS", plugins_spec)
    var host = Host()
    assert_equal(host.driver_count(), 1)
    return host.driver(0)[].name()


def test_accelerator_listed_after_cpu() raises:
    var cpu = getenv("CPU_PLUGIN")
    var mock = getenv("MOCK_PLUGIN")
    assert_equal(_host_driver_name(String(t"CPU@{cpu};Mock@{mock}")), "Mock")


def test_accelerator_listed_before_cpu() raises:
    var cpu = getenv("CPU_PLUGIN")
    var mock = getenv("MOCK_PLUGIN")
    assert_equal(_host_driver_name(String(t"Mock@{mock};CPU@{cpu}")), "Mock")


def test_cpu_name_is_case_insensitive() raises:
    var cpu = getenv("CPU_PLUGIN")
    var mock = getenv("MOCK_PLUGIN")
    assert_equal(_host_driver_name(String(t"cpu@{cpu};Mock@{mock}")), "Mock")


def test_unnamed_entry_is_accelerator() raises:
    var cpu = getenv("CPU_PLUGIN")
    var mock = getenv("MOCK_PLUGIN")
    assert_equal(_host_driver_name(String(t"CPU@{cpu};{mock}")), mock)


def _assert_host_rejects(plugins_spec: String) raises:
    _ = setenv("MODULAR_DRIVER_PLUGINS", plugins_spec)
    var rejected = False
    try:
        _ = Host()
    except e:
        assert_equal(e.status, STATUS_INVALID_ARG)
        rejected = True
    assert_true(rejected, String(t"Host accepted {plugins_spec}"))


def test_cpu_only_raises() raises:
    var cpu = getenv("CPU_PLUGIN")
    _assert_host_rejects(String(t"CPU@{cpu}"))


def test_empty_spec_raises() raises:
    _assert_host_rejects(";")


def main() raises:
    test_accelerator_listed_after_cpu()
    test_accelerator_listed_before_cpu()
    test_cpu_name_is_case_insensitive()
    test_unnamed_entry_is_accelerator()
    test_cpu_only_raises()
    test_empty_spec_raises()
