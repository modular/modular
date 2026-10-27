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
"""Sweep driver: runtime axes, config filtering, and fixture planning.

A sweep is the product of runtime axes, parsed from the command line, and
comptime axes, which the benchmark expands with `comptime for`. Comptime arms
belong innermost, so the arms of one runtime config run back to back on the
same clocks:

```mojo
for cfg in sweep.configs():
    comptime for h in [8, 64]:
        comptime if arm_enabled["sweep_h", h]():
            run_config[h](cfg)
```

Fixtures are planned before anything is allocated: each config states its
requirements in a `Footprint`, the driver takes the maximum over the configs,
and `plan_groups` splits the sweep where that maximum would not fit in device
memory.
"""

from std.sys import get_defined_dtype, get_defined_int, is_defined

from internal_utils import arg_parse
from max.gpu.host import DeviceContext


def arm_enabled[define: StaticString, value: Int]() -> Bool:
    """Returns whether comptime arm `value` is built.

    `-D <define>=<v>` pins the axis to `v`, so the same source still builds
    one executable per arm when parallel compiles are wanted. Unset, every
    arm is built.
    """
    comptime pinned = get_defined_int[define, 0]()
    return pinned == 0 or pinned == value


def dtype_arm_enabled[define: StaticString, dtype: DType]() -> Bool:
    """Returns whether comptime dtype arm `dtype` is built.

    The `DType` counterpart of `arm_enabled`: `-D <define>=<dtype>` pins the
    axis, and unset builds every arm.
    """
    comptime if is_defined[define]():
        return get_defined_dtype[define, dtype]() == dtype
    return True


def parse_int_list(s: String) raises -> List[Int]:
    """Parses a comma-separated list of integers, such as `6,8192`."""
    var out = List[Int]()
    for part in s.split(","):
        var trimmed = part.strip()
        if trimmed.byte_length() > 0:
            out.append(Int(trimmed))
    return out^


@fieldwise_init
struct Config(Copyable, Movable, Writable):
    """One point of a sweep: a value for each runtime axis."""

    var names: List[String]
    var values: List[Int]

    def __getitem__(self, name: StringSlice) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i]
        raise Error("config has no axis named ", name)

    def write_to(self, mut writer: Some[Writer]):
        for i in range(len(self.names)):
            if i > 0:
                writer.write(", ")
            writer.write(self.names[i], "=", self.values[i])


struct Sweep(Movable, Sized):
    """The runtime axes of a sweep, each a list of values.

    `axis("seq_len", "6,8192")` reads `--seq_len=...` from the command line,
    or from `KBENCH_ARG_seq_len` under kbench's environment mode, and falls
    back to the default. `zipped` pairs several axes element by element
    instead of crossing them, for a sweep whose shapes are a list of points.
    """

    var names: List[String]
    # One entry per group of axes that vary together: its points, each a value
    # for every name of the group. A plain axis is a group of one.
    var groups: List[List[List[Int]]]

    def __init__(out self):
        self.names = []
        self.groups = []

    def axis(mut self, name: String, default: String) raises:
        self.axis(name, parse_int_list(arg_parse(name, default)))

    def axis(mut self, name: String, var values: List[Int]) raises:
        if len(values) == 0:
            raise Error("sweep axis ", name, " has no values")
        var points = List[List[Int]](capacity=len(values))
        for v in values:
            points.append([v])
        self.names.append(name)
        self.groups.append(points^)

    def zipped(mut self, names: List[String], defaults: List[String]) raises:
        """Adds axes that vary together: point `i` takes value `i` of each.

        Each axis reads its own flag. A one-value list repeats for every
        point, so `--batch_size=8` alone keeps the default sequence lengths.
        """
        if len(names) != len(defaults) or len(names) == 0:
            raise Error("zipped axes need one default per name")
        var lists = List[List[Int]]()
        var n = 1
        for i in range(len(names)):
            var values = parse_int_list(arg_parse(names[i], defaults[i]))
            if len(values) == 0:
                raise Error("sweep axis ", names[i], " has no values")
            if len(values) > 1:
                if n > 1 and len(values) != n:
                    raise Error(
                        "zipped axis ",
                        names[i],
                        " has ",
                        len(values),
                        " values, not ",
                        n,
                    )
                n = len(values)
            lists.append(values^)
        var points = List[List[Int]](capacity=n)
        for p in range(n):
            var point = List[Int](capacity=len(names))
            for ref values in lists:
                point.append(values[p if len(values) > 1 else 0])
            points.append(point^)
        for ref name in names:
            self.names.append(name)
        self.groups.append(points^)

    def __len__(self) -> Int:
        var n = 1
        for ref g in self.groups:
            n *= len(g)
        return n

    def configs(self) -> List[Config]:
        """Returns every point of the product, first axis outermost."""
        var out = List[Config](capacity=len(self))
        var index = List[Int](length=len(self.groups), fill=0)
        for _ in range(len(self)):
            var values = List[Int](capacity=len(self.names))
            for g in range(len(self.groups)):
                for v in self.groups[g][index[g]]:
                    values.append(v)
            out.append(Config(self.names.copy(), values^))
            var g = len(self.groups) - 1
            while g >= 0:
                index[g] += 1
                if index[g] < len(self.groups[g]):
                    break
                index[g] = 0
                g -= 1
        return out^

    def configs[
        KeepFn: def(Config) raises -> Bool
    ](self, keep: KeepFn) raises -> List[Config]:
        """Returns the points `keep` accepts, in product order.

        Drop invalid points here, before planning, so they never size a
        fixture.
        """
        var out = List[Config]()
        for ref cfg in self.configs():
            if keep(cfg):
                out.append(cfg.copy())
        return out^


struct Footprint(Copyable, Movable, Writable):
    """Named sizes a fixture needs, each the maximum over the configs seen."""

    var names: List[String]
    var values: List[Int]

    def __init__(out self):
        self.names = []
        self.values = []

    def need(mut self, name: String, value: Int):
        """Raises requirement `name` to at least `value`."""
        for i in range(len(self.names)):
            if self.names[i] == name:
                self.values[i] = max(self.values[i], value)
                return
        self.names.append(name)
        self.values.append(value)

    def merge(mut self, other: Self):
        for i in range(len(other.names)):
            self.need(other.names[i], other.values[i])

    def __getitem__(self, name: StringSlice) -> Int:
        """Returns requirement `name`, or 0 when no config stated it."""
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i]
        return 0

    def write_to(self, mut writer: Some[Writer]):
        for i in range(len(self.names)):
            if i > 0:
                writer.write(" ")
            writer.write(self.names[i], "=", self.values[i])


@fieldwise_init
struct SweepGroup(Copyable, Movable):
    """Consecutive configs whose fixtures fit in memory together."""

    var configs: List[Config]
    var footprint: Footprint
    var bytes: Int


def device_budget(ctx: DeviceContext, fraction: Float64 = 0.9) -> Int:
    """Returns `fraction` of the device's free memory, in bytes.

    Returns `Int.MAX` where the device cannot report free memory, which
    leaves the sweep in one group.
    """
    try:
        var free = ctx.get_memory_info()[0]
        return Int(Float64(Int(free)) * fraction)
    except:
        return Int.MAX


def plan_groups[
    RequireFn: def(Config, mut Footprint) raises -> None,
    BytesFn: def(Footprint) raises -> Int,
](
    configs: List[Config],
    require: RequireFn,
    bytes: BytesFn,
    budget: Int = Int.MAX,
) raises -> List[SweepGroup]:
    """Partitions `configs` into groups whose fixtures fit in `budget` bytes.

    `require` raises a footprint to what one config needs; `bytes` prices a
    footprint. Groups keep the configs' order, so with no budget pressure the
    whole sweep is one group and every fixture is allocated and filled once.
    """
    var groups = List[SweepGroup]()
    var current = List[Config]()
    var footprint = Footprint()
    var current_bytes = 0
    for ref cfg in configs:
        var grown = footprint.copy()
        require(cfg, grown)
        var grown_bytes = bytes(grown)
        if grown_bytes > budget and len(current) > 0:
            groups.append(SweepGroup(current^, footprint^, current_bytes))
            current = List[Config]()
            grown = Footprint()
            require(cfg, grown)
            grown_bytes = bytes(grown)
        if grown_bytes > budget:
            raise Error(
                "config ",
                cfg,
                " needs ",
                grown_bytes,
                " bytes, over the budget of ",
                budget,
            )
        current.append(cfg.copy())
        footprint = grown^
        current_bytes = grown_bytes
    if len(current) > 0:
        groups.append(SweepGroup(current^, footprint^, current_bytes))
    return groups^
