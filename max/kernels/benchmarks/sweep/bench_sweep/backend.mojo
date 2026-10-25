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
"""The target-specific queries a sweep runs on: fill, timing, device limits.

The library covers the targets it knows through `ctx`'s portable APIs: grid-
stride fill kernels, device-event timing, `DeviceAttribute` limits, and the
management tools' throttle queries. A plugin-backed target supplies its own
`SweepBackend` from its plugin directory instead, so the shared library holds
nothing target-confidential. Every query has an `auto` answer, so a backend
overrides only what its target answers differently.
"""

from std.os import getenv
from std.subprocess import run

from max.gpu.host import DeviceAttribute, DeviceContext

# Covers MI355's 256 MiB Infinity Cache twice over, which `L2_CACHE_SIZE` does
# not report; also the fallback where the L2 size is unavailable.
comptime _MIN_AUTO_EVICT_BYTES = 512 * 1024 * 1024


def device_fill_supported(ctx: DeviceContext) raises -> Bool:
    """Returns whether the generators' fill kernels run on this target.

    The kernels are written for the APIs the library targets directly; any
    other target falls back to host fill and copy, which produces the same
    bits through the same formulas, so a plugin-backed accelerator needs no
    fill kernels of its own. The generators dispatch on this, and
    `SweepBackend.auto` reports it.
    """
    var api = ctx.api()
    return api == "cuda" or api == "hip" or api == "metal"


def auto_evict_bytes(ctx: DeviceContext) raises -> Int:
    """Returns the bytes a cold iteration must evict on a known target.

    Twice the reported L2, with a floor where the target reports none -- and
    on AMD, whose runtime does not report the Infinity Cache.
    """
    var l2 = 0
    try:
        l2 = ctx.get_attribute(DeviceAttribute.L2_CACHE_SIZE)
    except:
        pass
    if l2 == 0 or ctx.api() == "hip":
        return max(2 * l2, _MIN_AUTO_EVICT_BYTES)
    return 2 * l2


def auto_device_events(ctx: DeviceContext) raises -> Bool:
    """Returns whether timing comes from device events on a known target."""
    var api = ctx.api()
    return api == "cuda" or api == "hip"


@fieldwise_init
struct SweepBackend(Copyable, Movable):
    """What a sweep's target provides, as the queries the library makes of it.

    A plugin-backed target builds one from its own directory -- the answers
    its runtime and tools give -- and passes it to its `SweepHarness`, so the
    benchmarks and the library stay target-agnostic. `auto()` answers for
    the targets the library knows; any query a backend leaves at its default
    keeps that answer.
    """

    var device_fill: def(DeviceContext) thin raises -> Bool
    """Whether the generators' fill kernels run on the target.

    False fills on the host instead and copies, through the bit-exact CPU
    references; the generators make the same decision for themselves through
    `device_fill_supported`, so a backend's answer agrees with what its
    benchmark's fills do."""

    var device_events: def(DeviceContext) thin raises -> Bool
    """Whether timing comes from device events rather than a host clock.

    Reported in the harness's first `SWEEPINFO` line, so a sweep measured on
    a host clock says so in its output."""

    var evict_bytes: def(DeviceContext) thin raises -> Int
    """The bytes a cold iteration must evict: twice the cache the target
    reports, or the target's fallback where it reports none."""

    var throttle_state: def(DeviceContext) thin raises -> String
    """`clear`, `active:<conditions>` or `unavailable:<reason>`; see
    `throttle_state`."""

    @staticmethod
    def auto() -> Self:
        """Returns the backend of the targets the library knows.

        Fill kernels and device events where they run, the reported cache
        twice over, and the management tools' throttle query.
        """
        return Self(
            device_fill_supported,
            auto_device_events,
            auto_evict_bytes,
            throttle_state,
        )


def _visible_index(ctx: DeviceContext, env: StaticString) raises -> String:
    """Returns the management-tool index of `ctx`'s device.

    The runtime numbers only the visible devices; the tool numbers all of
    them, so map through the visibility list when there is one.
    """
    var ordinal = Int(ctx.id())
    var visible = getenv(String(env), "")
    if visible.byte_length() == 0:
        return String(ordinal)
    var entries = visible.split(",")
    if ordinal >= len(entries):
        raise Error(env, "=", visible, " has no entry ", ordinal)
    return String(entries[ordinal].strip())


def throttle_state(ctx: DeviceContext) -> String:
    """Returns `clear`, `active:<conditions>` or `unavailable:<reason>`.

    Reads only hardware throttling (thermal, power brake, PROCHOT), the
    conditions `utils/check-gpu-throttle.sh` fails on; utilization-driven
    clock states are normal on an idle device. The `auto` backend's throttle
    query; a plugin replaces it with its target's own, or leaves it
    unavailable.
    """
    try:
        var cmd: String
        var marker: StaticString
        var api = ctx.api()
        # nvidia-smi: "HW Slowdown", "HW Thermal Slowdown" and "HW Power Brake
        # Slowdown", each "Active" or "Not Active".
        if api == "cuda":
            cmd = String(
                "nvidia-smi -q -d PERFORMANCE -i ",
                _visible_index(ctx, "CUDA_VISIBLE_DEVICES"),
                " 2>&1",
            )
            marker = "Slowdown"
        elif api == "hip":
            var env: StaticString = "ROCR_VISIBLE_DEVICES"
            if getenv("HIP_VISIBLE_DEVICES", "").byte_length() > 0:
                env = "HIP_VISIBLE_DEVICES"
            # amd-smi is a Python tool; a test's Python environment breaks
            # its imports.
            cmd = String(
                "env -u PYTHONPATH -u PYTHONHOME -u PYTHONSAFEPATH amd-smi",
                " metric -g ",
                _visible_index(ctx, env),
                " 2>&1",
            )
            marker = "VIOLATION_STATUS"
        else:
            return String("unavailable:no throttle query for ", api)
        var output = run(cmd)
        var seen = False
        var active = String()
        for line in output.splitlines():
            if marker not in line or (api == "cuda" and "HW " not in line):
                continue
            seen = True
            var hit: Bool
            if api == "cuda":
                hit = line.endswith(": Active") or ": Active " in line
            else:
                hit = (
                    (
                        "PROCHOT" in line
                        or "PPT" in line
                        or "SOCKET_THERMAL" in line
                        or "VR_THERMAL" in line
                        or "HBM_THERMAL" in line
                    )
                    and "NOT ACTIVE" not in line
                    and "N/A" not in line
                )
            if hit:
                if active.byte_length() > 0:
                    active += ";"
                active += String(line.strip()).replace(" ", "")
        if not seen:
            var first = String()
            for line in output.splitlines():
                if line.strip().byte_length() > 0:
                    first = String(line.strip())
                    break
            return String(
                "unavailable:no throttle fields in ", cmd, ": ", first
            )
        if active.byte_length() > 0:
            return "active:" + active
        return "clear"
    except e:
        return String("unavailable:", e)
