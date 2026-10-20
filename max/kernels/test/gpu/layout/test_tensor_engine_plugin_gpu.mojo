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
"""Confirms the `layout._plugin` engine hook is load-bearing in device codegen.

The host-side test (`test/layout/test_tensor_engine_plugin.mojo`) can only
reach the hook through `TensorEngineForTarget[<a named target>]`, evaluated
during a *host* compile. It cannot observe the path that actually matters:
`CurrentTensorEngine` resolving through `_current_target()` while the compiler
is generating code for the accelerator. That path is the whole reason the hook
needs no `target` parameter threaded to its call sites, so it is worth pinning
on real hardware.

The tile here is therefore built *inside* the kernel, which is where a
graph-compiler-emitted kernel will consume the hook:

- `_HookTile` names `Engine=CurrentTensorEngine` -- the hook, never a literal
  engine name -- and is instantiated in device code.
- The kernel asserts at comptime that the plugin selected during *device*
  codegen is a real backend rather than `DefaultPlugin`. Every accelerator
  target in tree sets `stdlib_plugin` (`cuda`, `hip` or `metal`), so resolving
  to `DefaultPlugin` here would mean `_current_target()` had returned the host
  target and the hook was answering for the wrong machine.
- It then writes through that tile, so a hook resolving to something unusable
  fails as a wrong answer rather than quietly compiling.

Today every in-tree backend answers `DefaultEngine`, so the numerics are
equivalent to a tile that named `DefaultEngine` outright. That is the point:
this pins *which mechanism supplied it*, so that when a hardware plugin
registers a different engine, the path is already known to carry it.
"""

from layout import CurrentTensorEngine, TileTensor, row_major

from std.memory import AddressSpace

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from std._plugin import CurrentPlugin, DefaultPlugin
from std.testing import assert_equal

# Non-square so a bug that swaps the extents cannot pass.
comptime _ROWS = 4
comptime _COLS = 8
comptime _NUM_ELEMENTS = _ROWS * _COLS

comptime _Layout = type_of(row_major[_ROWS, _COLS]())

comptime _HookTile = TileTensor[
    .float32,
    _Layout,
    MutAnyOrigin,
    Engine=CurrentTensorEngine[AddressSpace.GENERIC],
]
"""A tile whose engine comes from the hook rather than a literal engine name.

`GENERIC` matches the tile's own default address space, so the engine the hook
picks and the space the tile declares agree -- which is the pairing the hook
exists to keep consistent.
"""


def fill_through_hook_typed_tile_kernel(
    data: MutPointer[Float32, MutAnyOrigin]
):
    """Fills each element with its linear index through a hook-typed tile.

    The `comptime assert` is the real payload: it is evaluated during device
    codegen, so it witnesses that the plugin selector ran against the
    accelerator's `stdlib_plugin` field rather than the host's.
    """
    comptime assert CurrentPlugin != DefaultPlugin, (
        "device codegen resolved the host plugin; CurrentTensorEngine is"
        " answering for the wrong target"
    )

    if global_idx.x != 0:
        return
    var tile = _HookTile(ptr=data, layout=row_major[_ROWS, _COLS]())
    for i in range(_NUM_ELEMENTS):
        tile.raw_store[width=1](i, Float32(i))


def test_hook_typed_tile_round_trips_on_device(ctx: DeviceContext) raises:
    """A tile typed by the hook builds and writes correctly on real hardware."""
    print("== test_hook_typed_tile_round_trips_on_device")

    var buf = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    buf.enqueue_fill(Float32(-1))

    ctx.enqueue_function[fill_through_hook_typed_tile_kernel](
        buf, grid_dim=1, block_dim=1
    )

    with buf.map_to_host() as host:
        for i in range(_NUM_ELEMENTS):
            assert_equal(host[i], Float32(i))


def main() raises:
    with DeviceContext() as ctx:
        test_hook_typed_tile_round_trips_on_device(ctx)
