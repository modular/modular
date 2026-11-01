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
"""Tests for the `layout._plugin` `TensorEngine` hooks.

`LayoutPluginHooks.TensorEngineFor` answers "which engine backs a tile in this
address space on this backend", keyed on the `LAYOUT_PLUGINS` registry. The
backend and the address space together pick out exactly one engine: some
targets want one engine for every space, others a different one per space.

In the default build every combination answers `DefaultEngine`, so these tests
are a guard: they pin that inertness, so an accidental arm cannot silently
retype every graph-compiler tensor argument, and they cover each
`stdlib_plugin` value an in-tree target actually sets as well as each address
space a tile is realistically placed in.
"""

from std._gpu.host.info import TargetAccelerator, TargetAcceleratorCollection
from std._plugin import CurrentPlugin, DefaultPluginHooks
from std.builtin.variadics import TypeList
from std.memory import AddressSpace
from std.sys.info import CompilationTarget
from std.testing import assert_equal, TestSuite

from max._plugin.cuda.targets import CudaTargets
from max._plugin.hip.targets import HipTargets
from max._plugin.metal.targets import MetalTargets

from layout import (
    CurrentLayoutPlugin,
    CurrentTensorEngine,
    DefaultEngine,
    DefaultLayoutPlugin,
    LayoutPluginForTarget,
    TensorEngineForTarget,
)

comptime _DEFAULT = DefaultEngine[element_width=1]

comptime _TARGETS = TypeList.of[
    Trait=TargetAcceleratorCollection, CudaTargets, HipTargets, MetalTargets
]()

comptime _target[arch: StaticString] = TargetAccelerator.from_arch[
    arch, _TARGETS
].mlir_target

# The spaces a tile actually lands in: graph-compiler arguments and device
# globals are GENERIC, on-chip staging is SHARED, and per-thread scratch is
# LOCAL. An accelerator is most likely to want a distinct engine across exactly
# this split, so all three are pinned rather than just the default.
comptime _SPACES = [
    AddressSpace.GENERIC,
    AddressSpace.GLOBAL,
    AddressSpace.SHARED,
    AddressSpace.LOCAL,
]


def test_current_target_resolves_to_default_engine() raises:
    """The host build has no `stdlib_plugin`, so it takes the defaults."""
    comptime assert (
        CurrentPlugin == DefaultPluginHooks[CompilationTarget.current()]
    ), "the host build should select DefaultPluginHooks"
    comptime assert (
        CurrentLayoutPlugin == DefaultLayoutPlugin
    ), "the host build should select DefaultLayoutPlugin"
    comptime assert CurrentTensorEngine[AddressSpace.GENERIC] == _DEFAULT, (
        "the default build must resolve to DefaultEngine; an arm has changed"
        " what every unannotated tile is backed by"
    )
    assert_equal(
        CurrentTensorEngine[AddressSpace.GENERIC]._BASE_TYPE_NAME,
        "DefaultEngine",
    )
    assert_equal(CurrentTensorEngine[AddressSpace.GENERIC].element_size, 1)


def test_every_address_space_resolves_to_default_engine() raises:
    """No in-tree backend splits its engine by address space."""
    comptime for space in _SPACES:
        comptime assert CurrentTensorEngine[space] == _DEFAULT, (
            "an address space has acquired its own engine; every in-tree"
            " backend is pointer-addressable in all of them"
        )


def test_unregistered_backends_fall_back_to_default_plugin() raises:
    """In-tree backends have no `LAYOUT_PLUGINS` entry of their own."""
    comptime assert (
        LayoutPluginForTarget[_target["sm_100a"]] == DefaultLayoutPlugin
    ), "an unregistered stdlib_plugin must take DefaultLayoutPlugin"


def test_cuda_targets_resolve_to_default_engine() raises:
    """`stdlib_plugin = "cuda"`, covering every NVIDIA target."""
    comptime assert (
        TensorEngineForTarget[_target["sm_100a"]] == _DEFAULT
    ), "B200 must resolve to DefaultEngine"
    comptime assert (
        TensorEngineForTarget[_target["sm_80"]] == _DEFAULT
    ), "A100 must resolve to DefaultEngine"


def test_hip_target_resolves_to_default_engine() raises:
    """`stdlib_plugin = "hip"`, covering every AMD target."""
    comptime assert (
        TensorEngineForTarget[_target["gfx942"]] == _DEFAULT
    ), "MI300X must resolve to DefaultEngine"


def test_metal_target_resolves_to_default_engine() raises:
    """`stdlib_plugin = "metal"`, covering every Apple target."""
    comptime assert (
        TensorEngineForTarget[_target["apple-m3"]] == _DEFAULT
    ), "Apple M3 must resolve to DefaultEngine"


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
