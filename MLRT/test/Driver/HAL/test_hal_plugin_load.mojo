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
"""Tests for loading a HAL plugin: version negotiation and a failed
`M_driver_create`.

The mock plugin reads its knobs from the environment at each call, so one
process covers every case.
"""

from std.ffi import OwnedDLHandle
from std.os import getenv, setenv, unsetenv
from max._driver import Driver
from max._driver.max_license.hal.plugins import HOST_DRIVER_VERSION
from max._driver.max_license.hal.status import STATUS_INCOMPATIBLE_VERSION
from std.testing import assert_equal, assert_false, assert_true, TestSuite

# The mock's `MOCK_STATUS_CREATE_FAILED`.
comptime _MOCK_STATUS_CREATE_FAILED: Int64 = -65536


def _mock_path() -> String:
    return getenv("MOCK_PLUGIN")


def _mock_spec() -> String:
    return "Mock@" + _mock_path()


def _reset_mock() raises:
    for name in [
        "MOCK_VERSION_MAJOR",
        "MOCK_VERSION_MINOR",
        "MOCK_VERSION_UNREPORTED",
        "MOCK_CREATE_FAIL",
    ]:
        _ = unsetenv(name)


def _load_error(spec: String) raises -> Optional[Int64]:
    """Loads `spec` and returns the status it failed with, if any."""
    try:
        _ = Driver.create(spec)
    except e:
        return e.status
    return None


def _load_message(spec: String) raises -> String:
    try:
        _ = Driver.create(spec)
    except e:
        return e.message
    raise Error(String(t"loading '{spec}' succeeded"))


def test_compatible_version_loads() raises:
    _reset_mock()
    var driver = Driver.create(_mock_spec())
    assert_equal(driver[].device_count(), 2)


def test_newer_minor_version_loads() raises:
    _reset_mock()
    _ = setenv("MOCK_VERSION_MINOR", String(HOST_DRIVER_VERSION.minor + 1))
    assert_false(_load_error(_mock_spec()))


def test_older_minor_version_is_rejected() raises:
    _reset_mock()
    assert_true(HOST_DRIVER_VERSION.minor > 0, "no older minor version exists")
    var minor = HOST_DRIVER_VERSION.minor - 1
    _ = setenv("MOCK_VERSION_MINOR", String(minor))
    var status = _load_error(_mock_spec())
    assert_equal(status.value(), STATUS_INCOMPATIBLE_VERSION)
    var message = _load_message(_mock_spec())
    assert_true(_mock_path() in message, message)
    assert_true(String(t"{HOST_DRIVER_VERSION}") in message, message)
    assert_true(
        String(t"{HOST_DRIVER_VERSION.major}.{minor}.") in message, message
    )


def test_wrong_major_version_is_rejected() raises:
    _reset_mock()
    var major = HOST_DRIVER_VERSION.major + 1
    _ = setenv("MOCK_VERSION_MAJOR", String(major))
    var status = _load_error(_mock_spec())
    assert_equal(status.value(), STATUS_INCOMPATIBLE_VERSION)
    var message = _load_message(_mock_spec())
    assert_true(_mock_path() in message, message)
    assert_true(
        String(t"implements driver interface {major}.") in message, message
    )


def test_unreported_version_is_named_as_such() raises:
    _reset_mock()
    _ = setenv("MOCK_VERSION_UNREPORTED", "1")
    var status = _load_error(_mock_spec())
    assert_equal(status.value(), STATUS_INCOMPATIBLE_VERSION)
    var message = _load_message(_mock_spec())
    assert_true(_mock_path() in message, message)
    assert_true(
        "without reporting the version it implements" in message, message
    )


def test_create_failure_reports_plugin_message_and_destroys_handle() raises:
    _reset_mock()
    # Keeps the mock resident so its counter survives the loader's dlclose.
    var lib = OwnedDLHandle(_mock_path())
    _ = setenv("MOCK_CREATE_FAIL", "1")
    var status = _load_error(_mock_spec())
    assert_equal(status.value(), _MOCK_STATUS_CREATE_FAILED)
    var message = _load_message(_mock_spec())
    assert_true("mock create failed on request" in message, message)
    assert_true(_mock_path() in message, message)
    var live = lib.get_function[Int64]("M_driver_mock_live_failed_drivers")()
    assert_equal(live, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
