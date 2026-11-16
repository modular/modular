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

"""Host-side pins for the MLA decode SMEM stage counts.

The unified-gather path stages KV twice and gives the MMA-operand ring one
stage more than the gather ring, but only when the leftover SMEM covers it. The
sizing fails safe: when it does not fit, both rings keep the shared depth and
the kernel still produces the right answer, just slower. No correctness test
can tell the two apart, so the depth is pinned here rather than assumed.
"""

from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_sparse_qkv_fp8 import (
    sparse_qkv_fp8_carve_bytes,
)
from nn.attention.gpu.nvidia.sm100.nvfp4_kv_dequant import (
    nvfp4_mla_row_bytes,
    nvfp4_mla_scale_row_bytes,
)


def _config[
    page_size: Int,
    num_heads: Int,
    unified_gather: Bool,
    layout_g: Bool = False,
    half_rows: Bool = False,
]() -> MLA_SM100_Decode_Config:
    return MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=num_heads,
        depth=512,
        q_depth=576,
        dtype_size=1,
        kv_type_size=1,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        kv_mma_swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        page_size=page_size,
        decoding_warp_split_k=False,
        split_page_size=128,
        scale_block_size=0,
        native_fp8=True,
        per_token_scale_rope_aware=False,
        decode_layout_g=layout_g,
        native_fp8_unified_gather=unified_gather,
        half_rows=half_rows,
    )


def _check[page_size: Int, num_heads: Int]() raises:
    comptime cfg = _config[page_size, num_heads, True]()
    if cfg.num_kv_mma_stages != cfg.num_kv_stages + 1:
        raise Error(
            "unified gather lost its extra MMA-operand stage at page=",
            page_size,
            " heads=",
            num_heads,
            ": gather ring ",
            cfg.num_kv_stages,
            ", MMA ring ",
            cfg.num_kv_mma_stages,
        )
    # The extra stage and its barrier pair are charged to `smem_used` by hand.
    # Under-counting them overflows the carveout at launch, not here.
    if cfg.smem_used > MLA_SM100_Decode_Config.sm100_smem_carveout:
        raise Error(
            "SMEM over the carveout at page=",
            page_size,
            " heads=",
            num_heads,
            ": ",
            cfg.smem_used,
            " > ",
            MLA_SM100_Decode_Config.sm100_smem_carveout,
        )

    # Every other path shares one depth between the two rings.
    comptime shared = _config[page_size, num_heads, False]()
    if shared.num_kv_mma_stages != shared.num_kv_stages:
        raise Error(
            "non-unified path grew a second ring depth at page=",
            page_size,
            " heads=",
            num_heads,
            ": ",
            shared.num_kv_stages,
            " vs ",
            shared.num_kv_mma_stages,
        )


def _check_layout_g[page_size: Int, num_heads: Int]() raises:
    # Layout G halves Q and P but each KV stage is unchanged, so the unified
    # gather keeps Layout E's 2 gather and 3 MMA stages. The dense Layout G
    # kernel's pin of 4 would overflow the carveout here.
    comptime cfg = _config[page_size, num_heads, True, True]()
    if cfg.num_kv_stages != 2 or cfg.num_kv_mma_stages != 3:
        raise Error(
            "Layout G unified gather expected 2/3 stages at page=",
            page_size,
            " heads=",
            num_heads,
            ", got ",
            cfg.num_kv_stages,
            "/",
            cfg.num_kv_mma_stages,
        )
    if not comptime (cfg.supported()):
        raise Error("Layout G unified-gather config is not supported()")
    # The sparse dispatch's extra SMEM (idx_bars, TMEM pointer, idx_smem and
    # its alignment pad), as in launch_mla_sm100_decode_sparse_qkv_fp8.
    comptime sparse_extra = (
        2 * cfg.num_kv_stages * cfg.mbar_size
        + 4
        + 12
        + cfg.num_kv_stages * cfg.BN_QK * 4
    )
    if (
        cfg.smem_used + sparse_extra
        > MLA_SM100_Decode_Config.sm100_smem_carveout
    ):
        raise Error(
            "Layout G sparse SMEM over the carveout: ",
            cfg.smem_used + sparse_extra,
        )
    comptime o_round_cols = cfg.o_round_cols()
    if o_round_cols != 64:
        raise Error("Layout G O round stride ", o_round_cols, " != 64")

    comptime dense = _config[page_size, num_heads, False, True]()
    if dense.num_kv_stages != 4:
        raise Error("dense Layout G lost its 4-stage pin")


def _check_half_rows[page_size: Int]() raises:
    # The half-row softmax changes no SMEM, stage or TMEM math, applies at
    # <= 16 rows and needs Layout G.
    comptime off = _config[page_size, 8, True, True]()
    comptime on = _config[page_size, 8, True, True, True]()
    if on.smem_used != off.smem_used or (
        on.num_kv_stages != off.num_kv_stages
        or on.num_kv_mma_stages != off.num_kv_mma_stages
        or on.tmem_used != off.tmem_used
    ):
        raise Error("half_rows changed the SMEM/stage/TMEM carve")
    if not comptime (on.supported()):
        raise Error("half-row config is not supported()")
    if not comptime (on.half_row_softmax(16)) or comptime (
        on.half_row_softmax(17)
    ):
        raise Error("half_row_softmax row bound is not 16")
    if comptime (off.half_row_softmax(8)):
        raise Error("half_row_softmax applied with the knob off")
    comptime layout_e = _config[page_size, 8, True, False, True]()
    if comptime (layout_e.supported()):
        raise Error("half_rows accepted without Layout G")


def _config_p6[
    direct: Bool,
    split_k: Bool = False,
]() -> MLA_SM100_Decode_Config:
    return MLA_SM100_Decode_Config(
        num_q_heads=8,
        group=8,
        depth=512,
        q_depth=576,
        dtype_size=1,
        kv_type_size=1,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        kv_mma_swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        page_size=128,
        decoding_warp_split_k=split_k,
        split_page_size=128,
        scale_block_size=0,
        native_fp8=True,
        per_token_scale_rope_aware=False,
        decode_layout_g=True,
        native_fp8_unified_gather=True,
        half_rows=True,
        direct_gather=direct,
    )


def _config_nvfp4[
    num_heads: Int,
    packed: Bool,
    layout_g: Bool = False,
]() -> MLA_SM100_Decode_Config:
    """The sparse decode config a packed NVFP4 cache gets, as the dispatch
    builds it: FP8 Q over a `uint8` cache, the linear front end, and the
    packed linear gather slot when `packed` (`MLA_NVFP4_PACKED_LINEAR`).
    """
    comptime packed_stage = 64 * (
        nvfp4_mla_row_bytes[576]() + nvfp4_mla_scale_row_bytes[576]()
    )
    return MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=num_heads,
        depth=512,
        q_depth=576,
        input_q_depth=576,
        dtype_size=2,
        kv_type_size=1,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        kv_mma_swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        page_size=128,
        decoding_warp_split_k=False,
        split_page_size=128,
        scale_block_size=0,
        native_fp8=True,
        per_token_scale_rope_aware=False,
        native_fp8_unified_gather=True,
        decode_layout_g=layout_g,
        direct_gather=False,
        kv_convert=False,
        gather_stage_bytes=(packed_stage if packed else 0),
    )


def _sparse_extra[row_parts: Int = 1](cfg: MLA_SM100_Decode_Config) -> Int:
    """The sparse launcher's index ring, TMEM pointer and alignment pad.

    An NVFP4 index row holds the value rows then the scale rows, so its
    index ring carries `row_parts` slots per key.
    """
    return (
        2 * cfg.num_kv_stages * cfg.mbar_size
        + 4
        + 12
        + cfg.num_kv_stages * cfg.BN_QK * row_parts * 4
    )


def _p6_total(cfg: MLA_SM100_Decode_Config) -> Int:
    """`smem_used` plus the sparse launcher's extras and a stamp buffer."""
    return cfg.smem_used + _sparse_extra(cfg) + 16


def _check_carve[
    row_parts: Int = 1
](what: StringLiteral, cfg: MLA_SM100_Decode_Config) raises:
    """The kernel's carve ends inside what the launcher allocates.

    The stamp buffer is the last 16 B of the allocation, so a carve that ran
    past `smem_used` plus the launcher's extras would corrupt it, and SMEM
    allocation granularity would hide the overrun otherwise.
    """
    var carve = sparse_qkv_fp8_carve_bytes(cfg, sparse_row_parts=row_parts)
    var alloc = cfg.smem_used + _sparse_extra[row_parts](cfg)
    if carve > alloc:
        raise Error(what, ": carve ", carve, " B past the allocation ", alloc)
    print("  carve ", what, ": ", carve, " of ", alloc)


def _check_p6_ring[split_k: Bool, n: Int, m: Int]() raises:
    comptime cfg = _config_p6[True, split_k=split_k]()
    if cfg.num_kv_stages != n or cfg.num_kv_mma_stages != m:
        raise Error(
            "direct ring bn=",
            64,
            " split_k=",
            split_k,
            ": expected N/M ",
            n,
            "/",
            m,
            ", got ",
            cfg.num_kv_stages,
            "/",
            cfg.num_kv_mma_stages,
        )
    if not comptime (cfg.supported()):
        raise Error("direct ring config not supported(): bn=", 64)
    comptime total = _p6_total(cfg)
    if total > MLA_SM100_Decode_Config.sm100_smem_carveout:
        raise Error("direct ring SMEM over the carveout: ", total)
    _check_carve("direct ring", materialize[cfg]())
    print(
        "  direct ring bn=",
        64,
        " split_k=",
        split_k,
        " N/M=",
        n,
        "/",
        m,
        " SMEM ",
        total,
        " of ",
        MLA_SM100_Decode_Config.sm100_smem_carveout,
    )


def _check_p6() raises:
    # Direct gather into 640-wide SW128 rows: the S, P and index rings at 3,
    # the K/V ring the rest of SMEM.
    _check_p6_ring[False, 3, 5]()
    _check_p6_ring[True, 3, 5]()
    # Without the direct gather the 64-key Layout G carve is untouched.
    comptime off = _config_p6[False]()
    comptime base = _config[128, 8, True, True, True]()
    if (
        off.smem_used != base.smem_used
        or off.num_kv_stages != base.num_kv_stages
        or off.num_kv_mma_stages != base.num_kv_mma_stages
    ):
        raise Error("the direct-gather fields changed the Layout G carve")


def _check_shipped_carves() raises:
    # The routed configurations and the forced ones the BUILD file keeps.
    _check_carve("layout E h64", _config[128, 64, True]())
    _check_carve("layout E h128", _config[128, 128, True]())
    _check_carve("mode 1 linear h8", _config[128, 8, True, True, True]())
    _check_carve("mode 1 full rows h32", _config[128, 32, True, True]())
    _check_carve("mode 1 direct", _config_p6[True]())
    _check_carve("mode 1 direct split-K", _config_p6[True, split_k=True]())
    _check_carve[row_parts=2]("nvfp4 layout E h64", _config_nvfp4[64, True]())
    _check_carve[row_parts=2]("nvfp4 layout E h128", _config_nvfp4[128, True]())


def _check_nvfp4_layout_e[num_heads: Int]() raises:
    # The packed linear slot (`MLA_NVFP4_PACKED_LINEAR`) is what fits the
    # NVFP4 Layout E form: with it the doubled index ring, the TMEM pointer,
    # the alignment pad and the stamp all fit the carveout, and the deeper
    # ring runs 3 gather / 3 MMA stages.
    comptime cfg = _config_nvfp4[num_heads, True]()
    if cfg.num_kv_stages != 3 or cfg.num_kv_mma_stages != 3:
        raise Error(
            "nvfp4 layout E expected 3/3 stages at heads=",
            num_heads,
            ", got ",
            cfg.num_kv_stages,
            "/",
            cfg.num_kv_mma_stages,
        )
    comptime total = cfg.smem_used + _sparse_extra[row_parts=2](cfg) + 16
    if total > MLA_SM100_Decode_Config.sm100_smem_carveout:
        raise Error(
            "nvfp4 layout E SMEM over the carveout at heads=",
            num_heads,
            ": ",
            total,
            " > ",
            MLA_SM100_Decode_Config.sm100_smem_carveout,
        )
    # The full-size slot -- what `MLA_NVFP4_PACKED_LINEAR=0` builds -- must
    # keep missing the carveout: that is the knob-off behavior those head
    # counts rely on to fail the launcher's assert rather than launch.
    comptime full = _config_nvfp4[num_heads, False]()
    comptime full_total = (
        full.smem_used + _sparse_extra[row_parts=2](full) + 16
    )
    if full_total <= MLA_SM100_Decode_Config.sm100_smem_carveout:
        raise Error(
            "nvfp4 layout E full-size slot fits the carveout at heads=",
            num_heads,
            ": ",
            full_total,
        )
    print(
        "  nvfp4 layout E heads=",
        num_heads,
        " SMEM ",
        total,
        " of ",
        MLA_SM100_Decode_Config.sm100_smem_carveout,
        " (full slot ",
        full_total,
        ")",
    )


def _check_nvfp4_layout_g[num_heads: Int]() raises:
    # The packed slot applies at Layout G too: the freed 14,336 B per slot
    # deepens the gather ring to 3/3 stages, while the full-size slot (the
    # knob-off build) keeps the measured 2 gather / 3 MMA rings -- and must
    # still fit, so `-D MLA_NVFP4_PACKED_LINEAR=0` stays a runnable escape
    # hatch rather than a compile error at every head count.
    comptime on = _config_nvfp4[num_heads, True, True]()
    comptime off = _config_nvfp4[num_heads, False, True]()
    if on.num_kv_stages != 3 or on.num_kv_mma_stages != 3:
        raise Error(
            "nvfp4 layout G expected 3/3 stages at heads=",
            num_heads,
            ", got ",
            on.num_kv_stages,
            "/",
            on.num_kv_mma_stages,
        )
    if off.num_kv_stages != 2 or off.num_kv_mma_stages != 3:
        raise Error(
            "nvfp4 layout G full-size slot expected 2/3 stages at heads=",
            num_heads,
            ", got ",
            off.num_kv_stages,
            "/",
            off.num_kv_mma_stages,
        )
    comptime on_total = on.smem_used + _sparse_extra[row_parts=2](on) + 16
    if on_total > MLA_SM100_Decode_Config.sm100_smem_carveout:
        raise Error(
            "nvfp4 layout G SMEM over the carveout at heads=",
            num_heads,
            ": ",
            on_total,
        )
    comptime off_total = off.smem_used + _sparse_extra[row_parts=2](off) + 16
    if off_total > MLA_SM100_Decode_Config.sm100_smem_carveout:
        raise Error(
            "nvfp4 layout G full-size slot over the carveout at heads=",
            num_heads,
            ": ",
            off_total,
        )
    if not comptime (on.supported()):
        raise Error("nvfp4 layout G packed config is not supported()")
    _check_carve[row_parts=2]("nvfp4 layout G", materialize[on]())
    print(
        "  nvfp4 layout G heads=",
        num_heads,
        " SMEM ",
        on_total,
        " (full slot ",
        off_total,
        ")",
    )


def _check_wide[num_heads: Int, nvfp4: Bool, cta_group: Int = 1]() raises:
    comptime cfg = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=num_heads,
        depth=512,
        q_depth=576,
        input_q_depth=576,
        dtype_size=2,
        kv_type_size=1,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        kv_mma_swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        page_size=128,
        decoding_warp_split_k=False,
        native_fp8=True,
        native_fp8_unified_gather=True,
        direct_gather=not nvfp4 or cta_group == 2,
        kv_convert=nvfp4 and cta_group == 1,
        cta_group=cta_group,
        key_tile=128,
    )
    if not comptime (cfg.supported()):
        raise Error("128-key configuration is unsupported")
    if comptime (cfg.BN_QK != 128 or cfg.s_slot_stride() != 64):
        raise Error("128-key score geometry changed")
    if cfg.num_kv_stages < 2 or cfg.num_kv_mma_stages < 2:
        raise Error("128-key configuration lost pipeline overlap")
    if comptime (cfg.num_kv_stages > cfg.max_s_slots()):
        raise Error("128-key score slots overlap correction TMEM")
    comptime carve = sparse_qkv_fp8_carve_bytes(
        cfg, sparse_row_parts=2 if nvfp4 else 1
    )
    if carve + 16 > cfg.sm100_smem_carveout:
        raise Error("128-key configuration exceeds SMEM: ", carve)
    print(
        "  128-key heads=",
        num_heads,
        " nvfp4=",
        nvfp4,
        " cta_group=",
        cta_group,
        " SMEM=",
        carve,
        " stages=",
        cfg.num_kv_stages,
        "/",
        cfg.num_kv_mma_stages,
    )


def main() raises:
    _check_wide[64, False]()
    _check_wide[64, True]()
    _check_wide[128, False]()
    _check_wide[128, True]()
    _check_wide[128, False, 2]()
    _check_wide[128, True, 2]()
    _check_p6()
    _check_shipped_carves()
    _check_half_rows[64]()
    _check_half_rows[128]()
    _check_layout_g[64, 8]()
    _check_layout_g[128, 8]()
    _check_layout_g[128, 32]()
    _check[64, 8]()
    _check[128, 8]()
    _check[64, 16]()
    _check[128, 128]()
    _check_nvfp4_layout_e[64]()
    _check_nvfp4_layout_e[128]()
    _check_nvfp4_layout_g[8]()
    _check_nvfp4_layout_g[32]()
    print("decode config stage pins OK")
