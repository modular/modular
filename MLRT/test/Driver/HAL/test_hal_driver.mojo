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
"""Basic smoke test for the Mojo HAL driver.

Requires MODULAR_DRIVER_PLUGINS env var set to a plugin spec, e.g.:
    CUDA@/path/to/cuda_plugin.so
"""

from std.os import getenv

from max._driver import Host
from max._driver import Driver
from max._driver.max_license.hal.extensions.completion_flag import (
    FlagCondition,
)
from max._driver.max_license.hal.extensions.cuda import CUDAHALExtensions
from max._driver.max_license.hal.extensions.hip import HIPHALExtensions
from max._driver.max_license.hal.extensions.metal import MetalHALExtensions
from max._driver.max_license.hal.extensions.traits import (
    SupportsCompletionFlagAddressV1,
    SupportsCompletionFlagResetV1,
    SupportsCompletionFlagV1,
    SupportsPeerAccessV1,
)
from max._driver.max_license.hal.status import HALError, STATUS_INVALID_ARG
from std.testing import assert_equal, assert_false, assert_true


def test_driver_create() raises HALError:
    var host = Host()
    print("Driver count:", host.driver_count())
    print("Driver0 name:", host.driver(0)[].name())
    print("Accelerator count:", host.accelerator_count())
    print("----------------------")


def test_device_get() raises:
    var host = Host()
    var driver = host.driver(0)
    var device_count = driver[].device_count()
    for i in range(device_count):
        var device = driver[].get_device_dynamic(Int64(i))
        var name = device[].name()
        assert_true(name.byte_length() > 0, "device has an empty name")
        print("Device", i, name)
    print("----------------------")

    var rejected = False
    try:
        _ = driver[].get_device_dynamic(device_count)
    except e:
        assert_equal(e.status, STATUS_INVALID_ARG)
        rejected = True
    assert_true(rejected, "a device ID past the count was accepted")


def test_peer_access() raises:
    var host = Host()
    if host.accelerator_count() < 2:
        print("single device; skipping")
        return
    var dev0 = host.accelerator(0)
    var dev1 = host.accelerator(1)
    comptime Ext = type_of(dev0[].extensions())
    comptime if conforms_to(Ext, SupportsPeerAccessV1):
        ref ext = dev0[].extensions()
        var raw = dev0[].raw_driver()
        assert_false(
            ext.can_access_peer(raw[], dev0[].handle(), dev0[].handle())
        )
        var forward = ext.can_access_peer(
            raw[], dev0[].handle(), dev1[].handle()
        )
        assert_equal(
            forward,
            ext.can_access_peer(raw[], dev1[].handle(), dev0[].handle()),
        )
        if not forward:
            print("pair has no peer access; skipping enable")
            return
        var ctx0 = dev0[].get_context()
        var ctx1 = dev1[].get_context()
        ext.enable_peer_access(raw[], ctx0[].handle(), ctx1[].handle())
        ext.enable_peer_access(raw[], ctx0[].handle(), ctx1[].handle())
    else:
        print("extension set has no peer access; skipping")


def test_completion_flag_tiers() raises:
    # Pins each in-tree vendor set's flag tiers, so dropping one fails here.
    comptime assert conforms_to(
        CUDAHALExtensions, SupportsCompletionFlagResetV1
    )
    comptime assert conforms_to(HIPHALExtensions, SupportsCompletionFlagResetV1)
    comptime assert conforms_to(
        MetalHALExtensions, SupportsCompletionFlagResetV1
    )
    comptime assert not conforms_to(
        MetalHALExtensions, SupportsCompletionFlagAddressV1
    )
    comptime assert conforms_to(
        CUDAHALExtensions, SupportsCompletionFlagAddressV1
    )
    comptime assert conforms_to(
        HIPHALExtensions, SupportsCompletionFlagAddressV1
    )
    comptime assert FlagCondition.eq in CUDAHALExtensions.FlagConditions
    comptime assert FlagCondition.eq in HIPHALExtensions.FlagConditions
    comptime assert FlagCondition.eq not in MetalHALExtensions.FlagConditions


def test_completion_flag_round_trip() raises:
    var host = Host()
    var dev = host.accelerator(0)
    comptime Ext = type_of(dev[].extensions())
    comptime if conforms_to(Ext, SupportsCompletionFlagV1):
        ref ext = dev[].extensions()
        var flags: Ext.FlagGroup
        if case .Some(f) = ext.completion_flag():
            flags = f
        else:
            raise Error("in-tree plugin lacks completion flags")
        var raw = dev[].raw_driver()
        var ctx = dev[].get_context()
        var queue = ctx[].create_queue()

        # The loaded plugin reports the conditions its vendor set promises.
        var conds = ext.flag_conditions(raw[], dev[].handle())
        assert_true(FlagCondition.geq in conds)
        assert_equal(
            FlagCondition.eq in conds,
            FlagCondition.eq in Ext.FlagConditions,
        )

        var flag = ext.create_flag(raw[], ctx[].handle())
        assert_equal(flags.load(raw[], flag), 0)
        flags.signal(raw[], flag, 5)
        flags.wait(raw[], queue[]._handle, flag, 3, FlagCondition.geq)
        queue[].synchronize()
        assert_equal(flags.load(raw[], flag), 5)

        comptime if conforms_to(Ext, SupportsCompletionFlagResetV1):
            if case .Some(reset) = ext.completion_flag_reset():
                reset.reset(raw[], flag)
                assert_equal(flags.load(raw[], flag), 0)
            else:
                raise Error("reset tier lacks the reset symbol")
        comptime if conforms_to(Ext, SupportsCompletionFlagAddressV1):
            if case .Some(address) = ext.completion_flag_address():
                assert_true(address.device_address(raw[], flag) != 0)
            else:
                raise Error("address tier lacks the address symbol")

        ext.destroy_flag(raw[], ctx[].handle(), flag)
    else:
        print("extension set has no completion flags; skipping")


def main() raises:
    try:
        test_driver_create()
    except e:
        raise Error(String(e))
    test_device_get()
    test_peer_access()
    test_completion_flag_tiers()
    test_completion_flag_round_trip()
