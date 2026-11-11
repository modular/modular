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
"""CDNA5 (gfx1250) Tensor Data Mover (TDM) descriptors and instructions.

CDNA5 deletes CDNA4's `buffer_load_lds` (buffer-to-LDS DMA) and replaces it with
`tensor_load_to_lds` / `tensor_store_from_lds`, which take a multi-register
Tensor DMA Descriptor describing a box transfer between global memory and LDS.
Completion of both directions is tracked by the wave's TENSORcnt counter
(`s_wait_tensorcnt`); CDNA5's STOREcnt counts vector stores, not TDM traffic.

Only the 2D dense form is implemented: the gather and iterate-mode bits stay
zero, and groups 2 and 3 (dimensions 2-4 with their strides and tile extents)
are issued as zeros, which describes a rank-2 tensor. A descriptor is meant to
be built once with compile-time box and padding fields, then advanced with
`with_addr` and `with_extents` instead of being rebuilt from scratch on every
iteration.

TODO(anand): lift the 2D ceiling -- a caller that needs three or four
dimensions has to fill groups 2 and 3, and a gather caller also has to change
the `type` field away from the dense encoding, so both wait on a caller that
wants them.

The field layout and widths come from AMD's public
`hip/amd_detail/amd_gfx1250_TDM.h`, whose bitfield structs describe every group,
and the `type` field and live-transfer count are the values that header's own
constructor writes. Two conventions it does not state, and that no public AMD
text was found to confirm:

* Dim 0 is the contiguous dimension with an implicit stride of one element, so
  the row pitch of a 2D box goes in the 48-bit field the header calls
  `tensorDim0Stride`; the field named `tensorDim1Stride` therefore belongs to
  the stride of dim 2, which a 2D transfer leaves zero.
* `pad_interval` and `pad_amount` are stored in 32-bit dword units, biased by
  one: `pad_interval_bytes` is `4 << (field + 1)` (a power of two in 8..1024)
  and `pad_amount_bytes` is `4 * (field + 1)` (a multiple of 4 in 4..512). The
  header gives the fields' widths, and the host test pins this packing, but
  nothing public gives their units.

Packing is plain integer arithmetic, so `TDMDescriptor.make_2d` and its mutators
are usable from host code and are covered by a host test; only the two
instruction wrappers are gated on the CDNA5 target.
"""

from std.bit import log2_floor, pop_count
from std.memory import bitcast
from std.sys.info import _cdna_5_or_newer, size_of
from std.sys.intrinsics import llvm_intrinsic


@fieldwise_init
struct TDMDescriptor(TrivialRegisterPassable):
    """A gfx1250 TDM descriptor: groups 0 and 1 of the 20-dword structure.

    Groups 2 and 3 describe dimensions 2-4 with their strides and tile extents,
    none of which a 2D dense transfer uses; the instruction wrappers issue them
    as zeros and no storage is kept for them. Keep the descriptor in uniform
    (SGPR) registers: the TDM instructions require a wave-uniform descriptor and
    the hardware does not broadcast a divergent one.
    """

    var d0: SIMD[DType.int32, 4]
    """Group 0: the count field, the LDS byte address, the global address, and
    the descriptor type."""

    var d1: SIMD[DType.int32, 8]
    """Group 1: the workgroup multicast mask, element size, padding fields,
    tensor extents, tile box extents, and the dim-1 row stride."""

    @staticmethod
    @inline(.nodebug)
    def _data_size_code[dtype: DType]() -> UInt32:
        """Returns the descriptor's `data_size` code for an element type.

        Parameters:
            dtype: The element type of the transfer.

        Returns:
            The base-2 log of the element size in bytes.

        Constraints:
            The element size must be 1, 2, 4 or 8 bytes.
        """
        comptime bytes = size_of[dtype]()
        comptime assert (
            bytes == 1 or bytes == 2 or bytes == 4 or bytes == 8
        ), "TDM data_size supports 1, 2, 4 or 8-byte elements"
        return UInt32(log2_floor(bytes))

    @staticmethod
    @inline(.nodebug)
    def _pack_group0(
        global_addr: Int64, lds_addr: Int32, count: UInt32
    ) -> SIMD[DType.int32, 4]:
        """Packs group 0.

        Args:
            global_addr: Byte address of the box origin in global memory.
            lds_addr: LDS byte offset of the transfer's LDS end; only its low
                32 bits are packed.
            count: The count field, 1 for a live transfer and 0 to disable it.

        Returns:
            The four words of group 0.
        """
        # The descriptor `type` field: 2 for a dense (non-gather) transfer. The
        # HIP header's constructor hard-codes 2 ("spg") and names no other
        # value, so this follows HIP rather than a published encoding.
        comptime spg_type = 2
        var g0 = SIMD[DType.uint32, 4](0)
        g0[0] = count
        g0[1] = UInt32(lds_addr)
        g0[2] = UInt32(global_addr & 0xFFFFFFFF)
        g0[3] = UInt32((global_addr >> 32) & 0x1FFFFFF) | UInt32(spg_type) << 30
        return bitcast[DType.int32, 4](g0)

    @staticmethod
    @inline(.nodebug)
    def _write_extents(
        mut d1: SIMD[DType.uint32, 8], tensor_dim0: Int32, tensor_dim1: Int32
    ):
        """Writes the remaining dim-0 and dim-1 extents into group 1.

        The extents share two of their words with other fields, the atomic
        barrier address and the dim-0 tile box extent, so this keeps those
        halves and overwrites only the extent bits.

        Args:
            d1: Group 1, as unsigned words.
            tensor_dim0: Remaining extent of dim 0, in elements. Must be
                non-negative.
            tensor_dim1: Remaining extent of dim 1, in elements. Must be
                non-negative.
        """
        d1[1] = (d1[1] & 0xFFFF) | UInt32(tensor_dim0 & 0xFFFF) << 16
        d1[2] = (
            UInt32((tensor_dim0 >> 16) & 0xFFFF)
            | UInt32(tensor_dim1 & 0xFFFF) << 16
        )
        d1[3] = (d1[3] & 0xFFFF0000) | UInt32((tensor_dim1 >> 16) & 0xFFFF)

    @staticmethod
    @inline(.nodebug)
    def make_2d[
        dtype: DType,
        tile_dim0: Int,
        tile_dim1: Int,
        *,
        pad_interval_bytes: Int = 0,
        pad_amount_bytes: Int = 0,
        multicast_mask: Int = 0,
    ](
        global_addr: Int64,
        lds_addr: Int32,
        tensor_dim0: Int32,
        tensor_dim1: Int32,
        dim1_stride: Int64,
    ) -> Self:
        """Builds a 2D load or store descriptor for a box of `tile_dim0` by
        `tile_dim1` elements.

        Dim 0 is the fastest-varying dimension with an implicit stride of one
        element; `dim1_stride` is the pitch of dim 1 in elements. The extents
        are what remains of the global tensor from the box origin, so the
        hardware clamps each dimension against them and zero-fills the
        out-of-bounds tail, which is how a TDM tile load handles ragged M/N/K. A
        zero extent masks the whole dimension.

        Both extents must be non-negative. They are packed as 16-bit halves of a
        two's-complement value, so a negative remaining extent reads back as an
        extent near 2^32, which the hardware clamps nothing against and the
        transfer then runs past the end of the tensor. Where an extent is
        computed as a difference, the caller clamps it to zero before the call;
        `with_extents` deliberately does not, because it runs once per TDM
        issue and its instruction count is budgeted.

        Padding makes the LDS write pattern physical offset
        `dense + (dense / pad_interval_bytes) * pad_amount_bytes`, applied to
        the dense byte stream starting at `lds_addr`. Valid intervals are powers
        of two in 8..1024 that are multiples of 4; valid amounts are in 4..512
        and a multiple of 4. TDM has no XOR swizzle, so padding is the only way
        to avoid LDS bank conflicts on the read side.

        Parameters:
            dtype: The element type of the transfer.
            tile_dim0: Extent of the box along dim 0, in elements.
            tile_dim1: Extent of the box along dim 1, in elements.
            pad_interval_bytes: LDS padding interval in bytes (defaults to 0,
                which disables padding).
            pad_amount_bytes: LDS padding amount in bytes (defaults to 0, which
                disables padding).
            multicast_mask: 16-bit multicast mask, one bit per cluster rank of
                the workgroups that receive this transfer (defaults to 0, which
                means no multicast). AMD's published guidance on the
                instruction says the mask must include the requesting workgroup
                and may name at most five destinations; neither rule is in an
                ISA description, so only the 16-bit width and the
                five-destination bound are asserted here.

        Args:
            global_addr: Byte address of the box origin in global memory.
            lds_addr: LDS byte offset of the transfer's LDS end. Only its low
                32 bits reach the descriptor, and a negative offset is not
                rejected.
            tensor_dim0: Remaining extent of dim 0, in elements. Must be
                non-negative.
            tensor_dim1: Remaining extent of dim 1, in elements. Must be
                non-negative.
            dim1_stride: Pitch of dim 1, in elements, in [0, 2^48). It is
                packed into 48 bits and is not checked here, because any
                runtime check would sit in every kernel that builds a
                descriptor; the bits above 48 are dropped.

        Returns:
            The packed descriptor.

        Constraints:
            `tile_dim0` and `tile_dim1` must be in 1..65535, the padding values
            must be in the ranges above, `pad_amount_bytes` requires a nonzero
            `pad_interval_bytes`, both extents must be non-negative,
            `dim1_stride` must fit in 48 bits, and `multicast_mask` must fit in
            16 bits with at most five of them set.
        """
        comptime pad_enabled = pad_interval_bytes != 0
        comptime assert (
            pad_enabled or pad_amount_bytes == 0
        ), "pad_amount_bytes requires pad_interval_bytes"
        comptime if pad_enabled:
            comptime assert (
                pad_interval_bytes >= 8
                and pad_interval_bytes <= 1024
                and Scalar[DType.int64](pad_interval_bytes).is_power_of_two()
            ), "pad_interval_bytes must be a power of two in 8..1024"
            comptime assert (
                pad_amount_bytes >= 4
                and pad_amount_bytes <= 512
                and pad_amount_bytes % 4 == 0
            ), "pad_amount_bytes must be in 4..512, %4"
        comptime pad_interval_code = (
            log2_floor(pad_interval_bytes // 4) - 1
        ) if pad_enabled else 0
        comptime pad_amount_code = (
            pad_amount_bytes // 4 - 1
        ) if pad_enabled else 0
        comptime field16_max = Int(UInt16.MAX)
        comptime assert (
            tile_dim0 >= 1 and tile_dim0 <= field16_max
        ), "tile_dim0 must be in 1..65535 elements"
        comptime assert (
            tile_dim1 >= 1 and tile_dim1 <= field16_max
        ), "tile_dim1 must be in 1..65535 elements"
        comptime assert (
            multicast_mask >= 0 and multicast_mask <= field16_max
        ), "multicast_mask must be a 16-bit workgroup mask"
        comptime assert (
            pop_count(multicast_mask) <= 5
        ), "multicast_mask may set at most 5 workgroup bits"

        # The count field of a live transfer, as the HIP header writes it; zero
        # disables the transfer (see `disabled`).
        comptime active_count = 1
        var g1 = SIMD[DType.uint32, 8](0)
        g1[0] = (
            UInt32(multicast_mask)
            | TDMDescriptor._data_size_code[dtype]() << 16
            | UInt32(1 if pad_enabled else 0) << 20
            | UInt32(pad_interval_code) << 22
            | UInt32(pad_amount_code) << 25
        )
        TDMDescriptor._write_extents(g1, tensor_dim0, tensor_dim1)
        g1[3] = (g1[3] & 0xFFFF) | UInt32(tile_dim0) << 16
        g1[4] = UInt32(tile_dim1)
        g1[5] = UInt32(dim1_stride & 0xFFFFFFFF)
        # The row stride fits 48 bits: its low 32 in sgpr5, the rest in
        # sgpr6[15:0], which it shares with the unused (zero) dim-2 stride.
        g1[6] = UInt32((dim1_stride >> 32) & 0xFFFF)

        return Self(
            TDMDescriptor._pack_group0(
                global_addr, lds_addr, UInt32(active_count)
            ),
            bitcast[DType.int32, 8](g1),
        )

    def with_addr(self, global_addr: Int64, lds_addr: Int32) -> Self:
        """Returns a copy of this descriptor aimed at a new global and LDS
        address.

        Everything else, including a cleared count field, is carried over.

        Args:
            global_addr: Byte address of the box origin in global memory.
            lds_addr: LDS byte offset of the transfer's LDS end. Only its low
                32 bits reach the descriptor, and a negative offset is not
                rejected.

        Returns:
            The re-addressed descriptor.
        """
        return Self(
            TDMDescriptor._pack_group0(
                global_addr, lds_addr, UInt32(self.d0[0]) & 0x3
            ),
            self.d1,
        )

    def with_extents(self, tensor_dim0: Int32, tensor_dim1: Int32) -> Self:
        """Returns a copy of this descriptor with new remaining extents.

        Everything else, including the tile box, the padding fields and the
        count field, is carried over.

        Args:
            tensor_dim0: Remaining extent of dim 0, in elements. Must be
                non-negative.
            tensor_dim1: Remaining extent of dim 1, in elements. Must be
                non-negative.

        Returns:
            The re-extended descriptor.
        """
        var d1 = bitcast[DType.uint32, 8](self.d1)
        TDMDescriptor._write_extents(d1, tensor_dim0, tensor_dim1)
        return Self(self.d0, bitcast[DType.int32, 8](d1))

    def disabled(self) -> Self:
        """Returns a copy of this descriptor with a cleared count field.

        A count-0 descriptor transfers nothing, which is how a pipeline turns
        the last prefetch of a loop into a no-op without branching. Whether the
        hardware counts such a copy against TENSORcnt, and how long it takes to
        release it, is unverified, so a pipeline that issues one should wait for
        the counter to reach zero rather than for a specific count.

        Returns:
            The disabled descriptor.
        """
        var d0 = bitcast[DType.uint32, 4](self.d0)
        d0[0] = d0[0] & 0xFFFFFFFC
        return Self(bitcast[DType.int32, 4](d0), self.d1)


@inline(.always)
def _issue_tensor_copy[
    intrinsic: StaticString, cache_policy: Int
](desc: TDMDescriptor):
    """Calls a TDM copy intrinsic on a 2D descriptor.

    Parameters:
        intrinsic: The LLVM intrinsic, the load or the store.
        cache_policy: The 5-bit cache-policy immediate.

    Args:
        desc: The descriptor to issue.
    """
    # Groups 2 and 3 carry dims 2-4, their strides and tiles, all zero for a
    # 2D box; the trailing group is reserved and read as zeros.
    comptime zero_group_2 = SIMD[DType.int32, 4](0)
    comptime zero_group_3 = SIMD[DType.int32, 4](0)
    comptime zero_group_4 = SIMD[DType.int32, 8](0)
    llvm_intrinsic[intrinsic, NoneType](
        desc.d0,
        desc.d1,
        zero_group_2,
        zero_group_3,
        zero_group_4,
        Int32(cache_policy),
    )


@inline(.always)
def tensor_load_to_lds[
    cache_policy: Int = 0,
](desc: TDMDescriptor):
    """Issues a TDM global-to-LDS copy described by `desc`.

    Completion is tracked by the wave's TENSORcnt counter; pair it with
    `s_wait_tensorcnt`.

    Parameters:
        cache_policy: 5-bit immediate, three temporal-hint bits followed by two
            scope bits (defaults to 0, the default policy).

    Args:
        desc: The descriptor to issue. It must be wave-uniform, i.e. live in
            scalar (SGPR) registers.

    Constraints:
        Requires CDNA5 (gfx1250) or newer, and `cache_policy` must be in 0..31.
    """
    comptime assert (
        _cdna_5_or_newer()
    ), "tensor_load_to_lds requires CDNA5 (gfx1250) or newer"
    comptime assert (
        cache_policy >= 0 and cache_policy <= 31
    ), "cache_policy must be a 5-bit immediate (3 temporal bits, 2 scope bits)"
    _issue_tensor_copy["llvm.amdgcn.tensor.load.to.lds", cache_policy](desc)


@inline(.always)
def tensor_store_from_lds[
    cache_policy: Int = 0,
](desc: TDMDescriptor):
    """Issues a TDM LDS-to-global copy described by `desc`.

    The LDS side is read as a dense stream, so the descriptor's padding fields
    are ignored by stores. Completion is tracked by the wave's TENSORcnt
    counter, not by the vector-store STOREcnt counter; pair it with
    `s_wait_tensorcnt`.

    Parameters:
        cache_policy: 5-bit immediate, three temporal-hint bits followed by two
            scope bits (defaults to 0, the default policy).

    Args:
        desc: The descriptor to issue. It must be wave-uniform, i.e. live in
            scalar (SGPR) registers.

    Constraints:
        Requires CDNA5 (gfx1250) or newer, and `cache_policy` must be in 0..31.
    """
    comptime assert (
        _cdna_5_or_newer()
    ), "tensor_store_from_lds requires CDNA5 (gfx1250) or newer"
    comptime assert (
        cache_policy >= 0 and cache_policy <= 31
    ), "cache_policy must be a 5-bit immediate (3 temporal bits, 2 scope bits)"
    _issue_tensor_copy["llvm.amdgcn.tensor.store.from.lds", cache_policy](desc)
