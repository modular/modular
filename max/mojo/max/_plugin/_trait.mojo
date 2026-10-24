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

from std.bit import count_trailing_zeros
from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.utils.index import IndexList
from std._gpu import lane_id
from std._gpu.host.info import (
    NoGPU,
    TargetAccelerator,
    TargetAcceleratorCollection,
    TargetAcceleratorType,
)
from std._plugin.selector import Plugin
from std.sys.info import CompilationTarget, _accelerator_arch

from max.gpu import PDLLevel
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.host.info import get_gpu_info
from max.gpu.memory.memory import (
    _AsyncCopyGroupPluginHookFnType,
    _AsyncCopyWaitGroupPluginHookFnType,
)
from max.gpu.primitives.warp import _FULL_MASK
from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)


trait MaxPlugin(Plugin):
    """Names the `MaxPluginHooks` a backend uses for each target.

    A target selects its MAX plugin from its `opaque_plugin` payload: the
    first type there that conforms to this trait. See `_impl.mojo` for the
    fallbacks when the payload names none.
    """

    comptime HookFor[target: CompilationTarget]: MaxPluginHooks
    """The hooks this plugin uses for `target`."""


trait MaxPluginHooks:
    """Defines compile-time hooks for target-specific MAX behavior."""

    @staticmethod
    def elementwise_fn[
        rank: Int,
        simd_width: Int,
        *,
        pdl_level: PDLLevel = PDLLevel.ON,
    ](
        func: Some[
            def[
                width: Int, rank: Int, alignment: Int = 1
            ](IndexList[rank]) -> None
        ],
        shape: IndexList[rank, element_type=_],
        ctx: DeviceContext,
    ) raises:
        """Per-target plugin hook for `elementwise[...]`.

        Parameters:
            rank: The rank of the work domain.
            simd_width: The SIMD lane count for bulk invocations.
            pdl_level: PDL level for overlap control.

        Args:
            func: The body closure to invoke per index.
            shape: The shape of the work domain.
            ctx: The device context to dispatch on.

        Only invoked when `_handles_elementwise` is `True`; the default
        is never called and is a no-op.
        """
        pass

    comptime _handles_elementwise: Bool = False
    """Whether this plugin overrides MAX elementwise dispatch."""

    comptime warp_size: Int
    """The number of lanes in a warp on the target, which `WARP_SIZE` reads.

    It has no default: the trait's warp primitives are sized by it, so a GPU
    plugin that left it out would still compile, with the warp size of
    whatever accelerator the build targets, and be wrong. A plugin whose
    targets have no warps of their own uses `_accelerator_warp_size()`.
    """

    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = None
    """Lowers `barrier()`, which synchronizes the threads of a block and
    makes their shared-memory writes visible to each other.

    A GPU target that leaves this `None` rejects `barrier()` at compile time.
    """

    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = None
    """Lowers `syncwarp()`, which synchronizes the lanes of a warp.

    A GPU target that leaves this `None` rejects `syncwarp()` at compile time.

    Args:
        mask: The lanes taking part; a target without lane masks ignores it.
    """

    comptime async_copy_commit_group_fn: OptionalReg[
        _AsyncCopyGroupPluginHookFnType
    ] = None
    """Lowers `async_copy_commit_group()`.

    A target without asynchronous copy groups leaves this and the two wait
    hooks `None`: its `async_copy()` completes before returning, so there is
    nothing to commit or wait for, and the calls compile to nothing.
    """

    comptime async_copy_wait_group_fn: OptionalReg[
        _AsyncCopyWaitGroupPluginHookFnType
    ] = None
    """Lowers `async_copy_wait_group()`; see `async_copy_commit_group_fn`.

    Args:
        n: The number of committed groups that may still be pending.
    """

    comptime async_copy_wait_all_fn: OptionalReg[
        _AsyncCopyGroupPluginHookFnType
    ] = None
    """Lowers `async_copy_wait_all()`; see `async_copy_commit_group_fn`."""

    @staticmethod
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        """Lowers `shuffle_idx()`, which copies `val` from lane `offset`.

        This is the one shuffle a GPU plugin has to provide: the up, down and
        XOR hooks default to reading their source lane through it. A target
        that doesn't override it rejects every shuffle at compile time. A
        plugin whose hardware only shuffles 32-bit words implements it with
        `_shuffle.shuffle_words`, which legalizes every other width onto that.

        Parameters:
            dtype: The element type of `val`.
            simd_width: The number of elements in `val`.

        Args:
            mask: The lanes taking part; a target without lane masks ignores it.
            val: The value each lane contributes.
            offset: The source lane.

        Returns:
            The value read from the source lane.
        """
        CompilationTarget.unsupported_target_error[operation="shuffle_idx"]()

    @inline(.always)
    @staticmethod
    def shuffle_up_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        """Lowers `shuffle_up()`, which copies `val` from the lane `offset`
        below the caller.

        The default reads that lane through `shuffle_idx_fn`; a lane whose
        source would fall below the first lane of the warp keeps its own
        value. A plugin with a native instruction overrides it.

        Parameters:
            dtype: The element type of `val`.
            simd_width: The number of elements in `val`.

        Args:
            mask: The lanes taking part; a target without lane masks ignores it.
            val: The value each lane contributes.
            offset: The number of lanes to look down.

        Returns:
            The value read from the source lane.
        """
        var lane = Int32(lane_id())
        var src_lane = lane - offset.cast[.int32]()
        return Self.shuffle_idx_fn(
            mask, val, UInt32(src_lane.lt(0).select(lane, src_lane))
        )

    @inline(.always)
    @staticmethod
    def shuffle_down_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        """Lowers `shuffle_down()`, which copies `val` from the lane `offset`
        above the caller.

        The default reads that lane through `shuffle_idx_fn`; a lane whose
        source would fall past the last lane of the warp keeps its own value.
        A plugin with a native instruction overrides it.

        Parameters:
            dtype: The element type of `val`.
            simd_width: The number of elements in `val`.

        Args:
            mask: The lanes taking part; a target without lane masks ignores it.
            val: The value each lane contributes.
            offset: The number of lanes to look up.

        Returns:
            The value read from the source lane.
        """
        var lane = UInt32(lane_id())
        return Self.shuffle_idx_fn(
            mask,
            val,
            (lane + offset).le(UInt32(WARP_SIZE - 1)).select(offset, 0) + lane,
        )

    @inline(.always)
    @staticmethod
    def shuffle_xor_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        """Lowers `shuffle_xor()`, which copies `val` from the lane whose ID is
        the caller's XOR `offset`.

        The default reads that lane through `shuffle_idx_fn`; a lane whose
        source would fall outside the warp keeps its own value. A plugin with
        a native instruction overrides it.

        Parameters:
            dtype: The element type of `val`.
            simd_width: The number of elements in `val`.

        Args:
            mask: The lanes taking part; a target without lane masks ignores it.
            val: The value each lane contributes.
            offset: The value XORed with the caller's lane ID.

        Returns:
            The value read from the source lane.
        """
        var lane = UInt32(lane_id())
        var src_lane = lane ^ offset
        return Self.shuffle_idx_fn(
            mask, val, src_lane.lt(UInt32(WARP_SIZE)).select(src_lane, lane)
        )

    @staticmethod
    def vote_fn[ret_type: DType](val: Bool) -> Scalar[ret_type]:
        """Lowers `vote()`, which returns the mask of warp lanes that voted
        `True`.

        A GPU plugin has to provide this. A target that doesn't override it
        rejects `vote()` at compile time.

        Parameters:
            ret_type: The mask type, `DType.uint32` or `DType.uint64`.

        Args:
            val: The calling lane's vote.

        Returns:
            The mask with bit `l` set for each lane `l` that voted `True`.
        """
        CompilationTarget.unsupported_target_error[operation="vote"]()

    @inline(.always)
    @staticmethod
    def match_any_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        """Lowers `match_any()`, which returns the mask of warp lanes whose
        bits equal the caller's.

        The default finds one group per round: the lowest lane not yet in a
        group broadcasts its bits through `shuffle_idx_fn`, and `vote_fn`
        collects the lanes holding them. That is one shuffle and one ballot
        per distinct value in the warp. A plugin with a match instruction, or
        a cheaper broadcast, overrides it.

        Parameters:
            dtype: The bits' type, `DType.uint32` or `DType.uint64`.
            mask_type: The mask type, `DType.uint32` or `DType.uint64`.

        Args:
            bits: The bits of the calling lane's value.

        Returns:
            The mask with bit `l` set for each lane `l` holding equal bits.
        """
        # `pending` is a ballot, so it's the same on every lane and the warp
        # runs each round converged, as the shuffle requires.
        var pending = Self.vote_fn[mask_type](True)
        var result = Scalar[mask_type](0)
        while pending != 0:
            var leader = UInt32(count_trailing_zeros(pending))
            var matches = Self.shuffle_idx_fn(_FULL_MASK, bits, leader) == bits
            var group = Self.vote_fn[mask_type](matches)
            if matches:
                result = group
            pending &= ~group
        return result

    @inline(.always)
    @staticmethod
    def match_all_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        """Lowers `match_all()`, which returns the mask of warp lanes if they
        all hold the caller's bits, and 0 otherwise.

        The default broadcasts the first lane's bits through `shuffle_idx_fn`
        and checks, through `vote_fn`, that the lanes holding them are all the
        active lanes. A plugin with a match instruction, or a cheaper
        broadcast, overrides it.

        Parameters:
            dtype: The bits' type, `DType.uint32` or `DType.uint64`.
            mask_type: The mask type, `DType.uint32` or `DType.uint64`.

        Args:
            bits: The bits of the calling lane's value.

        Returns:
            The mask of all lanes if they hold equal bits, otherwise 0.
        """
        var chosen = Self.shuffle_idx_fn(_FULL_MASK, bits, 0)
        var matched = Self.vote_fn[mask_type](bits == chosen)
        var active = Self.vote_fn[mask_type](True)
        return matched if matched == active else Scalar[mask_type](0)


struct DefaultMaxPlugin(MaxPlugin):
    """Provides the default MAX plugin behavior."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["default"]()
    comptime HookFor[target: CompilationTarget] = DefaultMaxPluginHooks[target]


struct DefaultMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime warp_size = _accelerator_warp_size()


def _accelerator_warp_size() -> Int:
    """Returns the warp size of the build's `--target-accelerator`.

    Host code sizes its launches with it, and it is 0 when the build targets
    no accelerator.

    Returns:
        The accelerator's warp size, or 0 without an accelerator.
    """
    comptime if _accelerator_arch() == "":
        return 0
    else:
        return get_gpu_info().warp_size


struct DefaultTargets(TargetAcceleratorCollection):
    """Holds the placeholder target that stands for "no accelerator".

    `default_accelerator()` still has to resolve when no
    `--target-accelerator` is given, so this collection maps the empty
    architecture to `NoGPU`. It stays first in `TARGETS`.
    """

    comptime vendor_name: String = "<none>"

    # TODO(MSTDL-3241): Refactor to avoid needing this dummy value.
    comptime RAW_TARGETS = TypeList.of[
        Trait=TargetAcceleratorType,
        TargetAccelerator[NoGPU, CompilationTarget._unknown()]._with_cli_values[
            [""]
        ],
    ]().values

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        return String(target_arch0)

    @staticmethod
    def supported_archs_message() -> String:
        # The placeholder isn't a supported architecture, so the
        # unsupported-arch error doesn't list it.
        return ""
