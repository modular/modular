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
from std.collections import OptionalReg
from std.ffi import _DLHandle
from std.memory import MaybeUninit

from ..plugin import RawDriver
from ..plugins import (
    ContextHandle,
    DeviceHandle,
    FlagHandle,
    HALFunction,
    OutParam,
    PluginResultCode,
    QueueHandle,
    RawPluginComponent,
)
from ..status import STATUS_SUCCESS, STATUS_UNKNOWN_SYMBOL, HALError


__enum FlagCondition(Equatable, ImplicitlyCopyable):
    """A comparison a queue wait applies to a flag's value."""

    case geq
    """The flag holds at least the target."""
    case eq
    """The flag holds exactly the target."""
    case gt
    """The flag holds more than the target."""
    case lt
    """The flag holds less than the target."""
    case leq
    """The flag holds at most the target."""

    def bit(self) -> UInt32:
        """Returns the condition's bit in `M_driver_flag_condition_t`.

        Returns:
            The bit the C ABI uses for this condition.
        """
        match self:
            case .geq:
                return 1 << 0
            case .eq:
                return 1 << 1
            case .gt:
                return 1 << 2
            case .lt:
                return 1 << 3
            case .leq:
                return 1 << 4


@fieldwise_init
struct FlagConditionSet(Equatable, TrivialRegisterPassable):
    """A set of `FlagCondition`s, as `M_driver_flag_conditions` reports."""

    var mask: UInt32

    def __init__(out self, *conds: FlagCondition):
        """Creates a set holding `conds`.

        Args:
            conds: The conditions.
        """
        self.mask = 0
        for cond in conds:
            self.mask |= cond.bit()

    def __contains__(self, cond: FlagCondition) -> Bool:
        """Returns whether the set holds `cond`.

        Args:
            cond: The condition to look for.

        Returns:
            True if `cond` is in the set.
        """
        return (self.mask & cond.bit()) != 0


trait CompletionFlagV1(TrivialRegisterPassable):
    """A 64-bit flag the host signals and a queue waits on.

    A flag starts at 0 and never decreases without a reset. A wait with a
    condition the plugin doesn't report fails with `STATUS_UNIMPLEMENTED`.
    """

    # Status messages are driver-scoped, hence the `RawDriver` argument.
    def create(
        self, driver: RawDriver, context: ContextHandle
    ) raises HALError -> FlagHandle:
        ...

    def destroy(
        self, driver: RawDriver, context: ContextHandle, flag: FlagHandle
    ) raises HALError:
        ...

    def signal(
        self, driver: RawDriver, flag: FlagHandle, value: UInt64
    ) raises HALError:
        """Sets `flag` to `value` from the host, releasing satisfied waits.

        Args:
            driver: The driver, for status messages.
            flag: The flag to signal.
            value: The new value. Keep it below 2^63: HIP compares flag values
                as signed.

        Raises:
            If the plugin rejects the signal.
        """
        ...

    def load(
        self, driver: RawDriver, flag: FlagHandle
    ) raises HALError -> UInt64:
        ...

    def conditions(
        self, driver: RawDriver, device: DeviceHandle
    ) raises HALError -> FlagConditionSet:
        ...

    def wait(
        self,
        driver: RawDriver,
        queue: QueueHandle,
        flag: FlagHandle,
        value: UInt64,
        cond: FlagCondition,
    ) raises HALError:
        """Makes later work on `queue` wait until `flag` meets `cond`.

        Args:
            driver: The driver, for status messages.
            queue: The queue that waits.
            flag: The flag to test.
            value: The target. Keep it below 2^63: HIP compares flag values as
                signed.
            cond: The comparison, one the plugin reports.

        Raises:
            If the plugin refuses the wait.
        """
        ...


def _raise_on_error(
    driver: RawDriver, status: PluginResultCode, what: StaticString
) raises HALError:
    if status != STATUS_SUCCESS:
        var err = driver.get_status_message(status)
        raise HALError(
            err.status, message=String(t"failed to {what}: {err.message}")
        )


def load_optional[
    T: RawPluginComponent
](lib: _DLHandle, so_path: StringSpan) raises HALError -> OptionalReg[T]:
    """Resolves a capability group, or None when `lib` lacks any of its symbols.

    Parameters:
        T: The group.

    Args:
        lib: The plugin library.
        so_path: The plugin's path, for error messages.

    Returns:
        The group, or None.

    Raises:
        If loading fails for a reason other than a missing symbol.
    """
    try:
        return T(lib, so_path)
    except e:
        if e.status == STATUS_UNKNOWN_SYMBOL:
            return None
        raise e^


struct RawPluginCompletionFlag(CompletionFlagV1, RawPluginComponent):
    """Core entry points of `M_driver_flag.h`, shared by CUDA, HIP and Metal."""

    var flag_create: HALFunction[
        "M_driver_flag_create",
        def(
            context: ContextHandle, out_flag: OutParam[FlagHandle, _]
        ) thin abi("C") -> PluginResultCode,
        required=True,
    ]
    var flag_destroy: HALFunction[
        "M_driver_flag_destroy",
        def(
            context: ContextHandle, flag: FlagHandle
        ) thin abi("C") -> PluginResultCode,
        required=True,
    ]
    var flag_signal: HALFunction[
        "M_driver_flag_signal",
        def(flag: FlagHandle, value: UInt64) thin abi("C") -> PluginResultCode,
        required=True,
    ]
    var flag_load: HALFunction[
        "M_driver_flag_load",
        def(
            flag: FlagHandle, out_value: OutParam[UInt64, _]
        ) thin abi("C") -> PluginResultCode,
        required=True,
    ]
    var flag_conditions: HALFunction[
        "M_driver_flag_conditions",
        def(
            device: DeviceHandle, out_mask: OutParam[UInt32, _]
        ) thin abi("C") -> PluginResultCode,
        required=True,
    ]
    var queue_wait_flag: HALFunction[
        "M_driver_queue_wait_flag",
        def(
            queue: QueueHandle,
            flag: FlagHandle,
            value: UInt64,
            condition: UInt32,
        ) thin abi("C") -> PluginResultCode,
        required=True,
    ]

    def create(
        self, driver: RawDriver, context: ContextHandle
    ) raises HALError -> FlagHandle:
        var flag = MaybeUninit[FlagHandle]()
        _raise_on_error(
            driver,
            self.flag_create.get()(context, OutParam[FlagHandle](to=flag)),
            "create completion flag",
        )
        return flag.unsafe_assume_init()

    def destroy(
        self, driver: RawDriver, context: ContextHandle, flag: FlagHandle
    ) raises HALError:
        _raise_on_error(
            driver,
            self.flag_destroy.get()(context, flag),
            "destroy completion flag",
        )

    def signal(
        self, driver: RawDriver, flag: FlagHandle, value: UInt64
    ) raises HALError:
        _raise_on_error(
            driver,
            self.flag_signal.get()(flag, value),
            "signal completion flag",
        )

    def load(
        self, driver: RawDriver, flag: FlagHandle
    ) raises HALError -> UInt64:
        var value = MaybeUninit[UInt64]()
        _raise_on_error(
            driver,
            self.flag_load.get()(flag, OutParam[UInt64](to=value)),
            "load completion flag",
        )
        return value.unsafe_assume_init()

    def conditions(
        self, driver: RawDriver, device: DeviceHandle
    ) raises HALError -> FlagConditionSet:
        var mask = MaybeUninit[UInt32]()
        _raise_on_error(
            driver,
            self.flag_conditions.get()(device, OutParam[UInt32](to=mask)),
            "query completion flag conditions",
        )
        return FlagConditionSet(mask=mask.unsafe_assume_init())

    def wait(
        self,
        driver: RawDriver,
        queue: QueueHandle,
        flag: FlagHandle,
        value: UInt64,
        cond: FlagCondition,
    ) raises HALError:
        _raise_on_error(
            driver,
            self.queue_wait_flag.get()(queue, flag, value, cond.bit()),
            "wait on completion flag",
        )


trait CompletionFlagResetV1(TrivialRegisterPassable):
    """Resets a flag to 0 in place."""

    def reset(self, driver: RawDriver, flag: FlagHandle) raises HALError:
        """Sets `flag` back to 0, keeping its handle.

        Don't reset while a wait on the flag is pending; doing so is undefined
        behavior.

        Args:
            driver: The driver, for status messages.
            flag: The flag to reset.

        Raises:
            If the plugin fails.
        """
        ...


trait CompletionFlagAddressV1(TrivialRegisterPassable):
    """Exposes a flag's device-visible 64-bit slot."""

    def device_address(
        self, driver: RawDriver, flag: FlagHandle
    ) raises HALError -> UInt64:
        """Returns the device address of `flag`'s 64-bit slot.

        Args:
            driver: The driver, for status messages.
            flag: The flag.

        Returns:
            The address, valid while `flag` lives.

        Raises:
            If the plugin fails.
        """
        ...


struct RawPluginCompletionFlagReset(CompletionFlagResetV1, RawPluginComponent):
    """`M_driver_flag_reset` (CUDA, HIP, Metal)."""

    var flag_reset: HALFunction[
        "M_driver_flag_reset",
        def(flag: FlagHandle) thin abi("C") -> PluginResultCode,
        required=True,
    ]

    def reset(self, driver: RawDriver, flag: FlagHandle) raises HALError:
        _raise_on_error(
            driver, self.flag_reset.get()(flag), "reset completion flag"
        )


struct RawPluginCompletionFlagAddress(
    CompletionFlagAddressV1, RawPluginComponent
):
    """`M_driver_flag_device_address` (CUDA, HIP)."""

    var flag_device_address: HALFunction[
        "M_driver_flag_device_address",
        def(
            flag: FlagHandle, out_address: OutParam[UInt64, _]
        ) thin abi("C") -> PluginResultCode,
        required=True,
    ]

    def device_address(
        self, driver: RawDriver, flag: FlagHandle
    ) raises HALError -> UInt64:
        var address = MaybeUninit[UInt64]()
        _raise_on_error(
            driver,
            self.flag_device_address.get()(flag, OutParam[UInt64](to=address)),
            "get completion flag address",
        )
        return address.unsafe_assume_init()
