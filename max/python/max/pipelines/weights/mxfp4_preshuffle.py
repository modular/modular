# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Shared MXFP4 expert-weight preshuffle for the AMD preb grouped-matmul kernel.

CPU byte-permutation of per-expert MXFP4 ``B`` weights (and their E8M0
``B``-scales) into the 5D / 4D-cell layouts the AMD
``mxfp4_grouped_matmul_amd_preb`` kernel reads, so it can issue coalesced
DRAM->VGPR loads. Model weight adapters call
:func:`preshuffle_mxfp4_b_experts` + :func:`preshuffle_mxfp4_b_scales` in
lockstep and flip ``QuantConfig.mxfp4_preshuffled_b`` so ``MoEQuantized``
dispatches to the preb path.

Matches expert weights named ``...layers.N.mlp.experts.IDX.{gate,up,down}_proj``;
weights whose dtype/dims aren't MXFP4-packed and tile-aligned are silently
skipped (they fall through to the row-major kernel). Used by the Kimi K2.5 and
MiniMax-M3 adapters.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData

logger = logging.getLogger("max.pipelines")

# The layer prefix is optional: Kimi keeps a `language_model.` prefix, while
# MiniMax-M3 strips it, leaving keys like `layers.N.mlp.experts.J...`.
_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>(?:.+\.)?layers\.\d+\.mlp\.experts)"
    r"\.(?P<idx>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

_EXPERT_SCALE_RE = re.compile(
    r"^(?P<prefix>(?:.+\.)?layers\.\d+\.mlp\.experts)"
    r"\.(?P<idx>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight_scale$"
)


def _as_shuffleable_mxfp4_b(wd: WeightData) -> np.ndarray | None:
    """Return ``wd`` as a numpy view if it's a shuffleable MXFP4 B weight.

    A weight is shuffleable when its dtype is packed-MXFP4 (uint8) and its
    dims are MFMA-tile-aligned: ``N % 16 == 0`` (NLane=16) and
    ``K_BYTES % 64 == 0`` (4 KLane * 16 KPack). The shuffle reshape
    hardcodes those factors, so non-aligned dims would crash on reshape.
    Returns ``None`` when the weight isn't shuffleable.
    """
    if wd.dtype != DType.uint8:
        return None
    arr = np.from_dlpack(wd.data)
    if arr.ndim != 2 or arr.shape[0] % 16 != 0 or arr.shape[1] % 64 != 0:
        return None
    return arr


def _as_shuffleable_mxfp4_b_scale(wd: WeightData) -> np.ndarray | None:
    """Return ``wd`` as a uint8 view if it's a shuffleable MXFP4 B scale.

    Shuffleable when dtype is E8M0 and dims are cell-aligned for
    ``Shuffler.scale_4d_grouped_layout``: ``N % 32 == 0`` (S_MN_BLOCK) and
    ``K_SCALES % 8 == 0`` (S_K_BLOCK). The 2D src reshape used by
    :func:`_shuffle_scale_4d` hardcodes those factors. Returns ``None``
    when not shuffleable. E8M0 bytes are reinterpreted as uint8 for byte
    permutation; the dtype is restored to E8M0 by the caller.
    """
    if wd.dtype != DType.float8_e8m0fnu:
        return None
    try:
        arr = np.from_dlpack(wd.data)
    except (TypeError, BufferError):
        return None
    except RuntimeError:
        arr = np.from_dlpack(
            wd.to_buffer().view(DType.uint8, wd.shape.static_dims)
        )
    if arr.dtype != np.uint8:
        arr = arr.view(np.uint8)
    if arr.ndim != 2 or arr.shape[0] % 32 != 0 or arr.shape[1] % 8 != 0:
        return None
    return arr


def _shuffle_b_5d(src: np.ndarray, dst: np.ndarray) -> None:
    """Permute MXFP4 expert B bytes into ``Shuffler.b_5d_grouped_layout``.

    Reshape ``[N, K_BYTES]`` row-major into the 5D tile structure
    ``(N0, NLane=16, K0, KLane=4, KPack=16)`` and transpose into
    ``(N0, K0, KLane, NLane, KPack)`` so C-order strides match
    ``b_5d_grouped_layout`` in ``mxfp4_preshuffle_layouts.mojo``. ``dst``
    is a contiguous ``(N, K_BYTES)`` slot the caller owns.
    """
    N, K_BYTES = src.shape
    src_v = src.reshape(N // 16, 16, K_BYTES // 64, 4, 16).transpose(
        0, 2, 3, 1, 4
    )
    dst_v = dst.reshape(N // 16, K_BYTES // 64, 4, 16, 16)
    np.copyto(dst_v, src_v)


def _shuffle_scale_4d(src: np.ndarray, dst: np.ndarray) -> None:
    """Permute MXFP4 B-scale bytes into ``Shuffler.scale_4d_grouped_layout``.

    Reshape ``[MN, K_SCALES]`` row-major into the 6D decomposition
    ``(MN_block, MN_pack=2, MN_lane=16, K_block, K_pack=2, K_lane=4)``
    and transpose into the dst axis order
    ``(MN_block, K_block, K_lane, MN_lane, K_pack, MN_pack)`` so C-order
    strides match the 4D-cell byte layout addressed by
    ``Shuffler.scale_4d_byte_off``. Within each i32 cell the bytes land
    in ``(mn_pack, k_pack) = {(0,0), (1,0), (0,1), (1,1)}`` order at
    byte offsets ``{0, 1, 2, 3}`` — what the preb kernel's OPSEL byte
    selector reads.
    """
    MN, K_SCALES = src.shape
    src_v = src.reshape(MN // 32, 2, 16, K_SCALES // 8, 2, 4).transpose(
        0, 3, 5, 2, 4, 1
    )
    dst_v = dst.reshape(MN // 32, K_SCALES // 8, 4, 16, 2, 2)
    np.copyto(dst_v, src_v)


def preshuffle_mxfp4_b_experts(
    state_dict: dict[str, WeightData],
) -> None:
    """MXFP4 B preshuffle of all per-expert weights in-place on CPU.

    Walks ``state_dict``, groups expert weights by ``(prefix, proj)``,
    rewrites each group's WeightData entries with the bytes laid out in
    ``b_5d_grouped_layout`` so the AMD ``mxfp4_grouped_matmul_amd_preb``
    kernel reads them with coalesced DRAM->VGPR loads. Experts whose
    dtype/shape isn't MXFP4-packed uint8 with tile-aligned dims are
    silently skipped (they fall through to the row-major kernel).

    One numpy buffer per ``(prefix, proj)`` group keeps allocation count
    at ~180. Per-expert allocations would mean ~70k mmap chunks, blowing
    past glibc's M_MMAP_MAX (65536).
    """
    groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for name in state_dict:
        if m := _EXPERT_WEIGHT_RE.match(name):
            groups[m["prefix"], m["proj"]].append(name)

    if not groups:
        return

    t0 = time.perf_counter()
    n_total = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for names in groups.values():
            shuffleable = [
                (name, arr)
                for name in names
                if (arr := _as_shuffleable_mxfp4_b(state_dict[name]))
                is not None
            ]
            if not shuffleable:
                continue

            kept_names, srcs = zip(*shuffleable, strict=True)
            N, K_BYTES = srcs[0].shape
            buf = np.empty((len(srcs), N, K_BYTES), dtype=np.uint8)
            list(pool.map(_shuffle_b_5d, srcs, buf))
            for name, slot in zip(kept_names, buf, strict=True):
                state_dict[name] = WeightData.from_numpy(
                    slot, name=state_dict[name].name
                )
            n_total += len(srcs)

    logger.info(
        "MXFP4 B preshuffle: %d experts across %d groups in %.1fs",
        n_total,
        len(groups),
        time.perf_counter() - t0,
    )


def preshuffle_mxfp4_b_scales(
    state_dict: dict[str, WeightData],
) -> None:
    """MXFP4 B-scale preshuffle of all per-expert scales in-place on CPU.

    Walks ``state_dict``, groups expert scales by ``(prefix, proj)``,
    rewrites each group's WeightData entries with bytes laid out in
    ``scale_4d_grouped_layout`` so the AMD preb grouped-matmul kernel
    can issue direct-VGPR i32 scale loads (one 2x2 cell per lane).
    Scales whose dtype isn't E8M0 or whose dims aren't cell-aligned
    (``N % 32 == 0`` and ``K_SCALES % 8 == 0``) are silently skipped.

    Companion to :func:`preshuffle_mxfp4_b_experts`; should be called
    immediately after it so weight and scale layouts stay in sync.
    """
    groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for name in state_dict:
        if m := _EXPERT_SCALE_RE.match(name):
            groups[m["prefix"], m["proj"]].append(name)

    if not groups:
        return

    t0 = time.perf_counter()
    n_total = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for names in groups.values():
            shuffleable = [
                (name, arr)
                for name in names
                if (arr := _as_shuffleable_mxfp4_b_scale(state_dict[name]))
                is not None
            ]
            if not shuffleable:
                continue

            kept_names, srcs = zip(*shuffleable, strict=True)
            MN, K_SCALES = srcs[0].shape
            buf = np.empty((len(srcs), MN, K_SCALES), dtype=np.uint8)
            list(pool.map(_shuffle_scale_4d, srcs, buf))
            for name, slot in zip(kept_names, buf, strict=True):
                # from_numpy infers uint8 from the slab dtype; restore the
                # E8M0 metadata so downstream graph-compiler dtype checks
                # (e.g. grouped_dynamic_scaled_mxfp4_matmul) still pass.
                state_dict[name] = dataclasses.replace(
                    WeightData.from_numpy(slot, name=state_dict[name].name),
                    dtype=DType.float8_e8m0fnu,
                )
            n_total += len(srcs)

    logger.info(
        "MXFP4 B-scale preshuffle: %d experts across %d groups in %.1fs",
        n_total,
        len(groups),
        time.perf_counter() - t0,
    )
