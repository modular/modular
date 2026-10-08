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

# MiniMax-M3 decode MoE FFN CHAIN benchmark (MAX only, gfx950/MI355X).
#
# Times the per-rank decode kernel chain gemm1 -> SwiGLU-OAI + requant -> gemm2
# end to end: gemm1-start -> gemm2-end kernel time via device-graph replay.
# Router, EP dispatch, weight preshuffle, and host logic are OUTSIDE the timed
# region. Variants live in a registry (one builder contract over identical
# inputs), so new kernels -- the fused gemm1+SwiGLU epilogue, additional
# chain shapes -- slot in as additional builders.
#
# Measurement discipline:
#   * Timing is CUDA-event based on a captured device graph; no CPU timers on
#     the timed path.
#   * Cache busting: every replay rotates through `ncopies` distinct weight
#     sets sized so the working set exceeds 1.5x L2 (256 MB), covering gemm1 +
#     gemm2 active bytes COMBINED so gemm2 also replays L2-cold.
#   * Every variant replays the SAME graph shape with the SAME ncopies, so
#     graph-replay overhead is paid symmetrically.
#   * The decode grid cap (and rows) is always set; uncapped is not a decode
#     measurement.
#
# Baseline chain (production shape, MXFP8 a8w8):
#   gemm1: grouped_dynamic_block_scaled_matmul_amd (gate_up, N=6144 K=6144)
#   act:   ep.fused_silu_quantized (OAI-clamped SwiGLU + E8M0 requant,
#          max_padded_M=stride slot-fold)  [3-launch variant adds the
#          standalone A-scale preshuffle instead]
#   gemm2: grouped_dynamic_block_scaled_matmul_amd (down, N=6144 K=3072)
#
# Variants:
#   plain+preshuffle  4 launches (gemm1, fused_silu, A-scale preshuffle, gemm2)
#   plain+fold        3 launches (gemm1, fused_silu w/ slot scales, gemm2)
#                     load). No activation kernel and no A-scale traffic: the
#                     down-proj scales are computed in-kernel and never reach
#                     memory. Takes the gate_up weight sigma-permuted so gate
#                     and up land in one output tile. Exposes no packed
#                     intermediate, so --check skips its byte-identity gate.
#
# Run (bazel, gpu):
#   bazel run //max/kernels/benchmarks/misc/comparison:bench_amd_m3_moe_chain \
#     -- --M 96 --mtp 4 --rank worst --decode-grid-m-cap 96 --check
# Dry-run (no GPU): validates shapes/routing.
#   python bench_amd_m3_moe_chain.py --dry-run --M 96 --mtp 4

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from max.driver import Buffer
    from max.engine import Model
    from max.graph import TensorValue
    from max.nn.quant_config import QuantConfig

    # A builder wires one chain shape into the graph and returns
    # (down, packed_intermediate | None, intermediate_scales | None). The
    # probe pair is exposed only by variants that can, and only for --check.
    ChainOut = tuple[TensorValue, TensorValue | None, TensorValue | None]
    ChainBuilder = Callable[..., ChainOut]

if "/usr/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = (
        "/usr/bin:/bin:/usr/local/bin:/opt/rocm/bin:"
        + os.environ.get("PATH", "")
    )

_L2_CACHE_SIZE_BYTES = int(256e6)
_MI355_NAMEPLATE_TBPS = 8.0
_SCALE_BLOCK = 32  # MX block size along K (elements per E8M0 scale)
_MAX_CAPTURE_KEY = 0
_SWIGLU_ALPHA = 1.702
_SWIGLU_LIMIT = 7.0
_CHAIN_LAUNCHES = {
    "plain+preshuffle": 4,
    "plain+fold": 3,
    "fused_mxfp8": 2,
}
# Variants in baseline -> most-fused order.
_CHAIN_VARIANTS = (
    "plain+preshuffle",
    "plain+fold",
    "fused_mxfp8",
)
# --check gate per variant: worst absolute difference from the first variant,
# as a fraction of the output's own scale. A variant with no entry must agree
# bit for bit. fused_mxfp8 has none: its epilogue keeps the unfused path's bf16
# round-trip before the SwiGLU, and its in-kernel block maxima match the
# standalone quantizer's, so any difference means the fusion changed the math.
_CHECK_REL_TOL: dict[str, float] = {}


@dataclass(frozen=True)
class M3Config:
    """MiniMax-M3 decode MoE dimensions (EP8, full per-rank shapes)."""

    hidden: int = 6144
    inter: int = 3072
    experts: int = 128
    topk: int = 4
    ep: int = 8
    n_shared: int = 1  # always-active shared expert, fused as group 0


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _decode_grid_m_rows(decode_grid_m_cap: int, cfg: M3Config) -> int:
    """Rows grid.y must cover per expert on the capped decode band
    (production moe_fp8.py rows derivation)."""
    return _ceil_div(decode_grid_m_cap * cfg.ep, cfg.topk)


def rank_counts(
    m: int,
    cfg: M3Config,
    routing: str,
    seed: int = 0,
    active: int = 0,
    mtp: int = 1,
    rank: str = "worst",
) -> np.ndarray:
    """Per-local-expert routed token counts for ONE EP rank.

    `routing` picks the model:

    - ``uniform`` spreads ``m * topk`` routes evenly over all experts. It is
      the worst case, not the typical one: below ``experts`` routes it still
      lights up every local expert with one row each, so the chain streams all
      of them regardless of ``m``.
    - ``multinomial`` draws the routes at random, which is what the router
      actually does. Experts that draw nothing are dropped from the offsets and
      their weights are never read -- and since this chain is weight-stream
      bound, that is the whole cost model.

    `mtp` is tokens per request. The draw is per REQUEST, not per token: the
    MTP tokens of one request are consecutive positions in the same sequence
    and route to the same experts. So `m` tokens are only `m // mtp`
    independent draws, and the extra tokens land on already-active experts
    where they cost nothing -- one tile holds BM rows either way. Drawing per
    token instead roughly doubles the active-expert count at BS24xMTP4, which
    is the entire cost model, so this is not a detail.
    """
    n_local = cfg.experts // cfg.ep
    routes = m * cfg.topk
    if active > 0:
        # Design-curve mode: hold the token count fixed and vary only how many
        # experts the routes land on. The chain is weight-stream bound, so this
        # is the axis its cost actually moves along; concentrating the same
        # routes is also what a hot-expert router does.
        if active > n_local:
            raise ValueError(f"active {active} exceeds {n_local} local experts")
        per_rank = routes // cfg.ep
        counts = np.zeros(n_local, dtype=np.int64)
        counts[:active] += per_rank // active
        counts[: per_rank - (per_rank // active) * active] += 1
        return counts
    if routing == "uniform":
        all_counts = np.zeros(cfg.experts, dtype=np.int64)
        all_counts += routes // cfg.experts
        all_counts[: routes - (routes // cfg.experts) * cfg.experts] += 1
        return all_counts[:n_local].copy()
    if routing != "multinomial":
        raise ValueError(f"unknown routing model {routing!r}")
    # Each token picks `topk` DISTINCT experts, so draw per token without
    # replacement rather than multinomial over the flattened route count.
    rng = np.random.default_rng(seed)
    all_counts = np.zeros(cfg.experts, dtype=np.int64)
    requests, tail = divmod(m, mtp)
    for r in range(requests + (1 if tail else 0)):
        rows = mtp if r < requests else tail
        for e in rng.choice(cfg.experts, size=cfg.topk, replace=False):
            all_counts[e] += rows
    per_rank = all_counts.reshape(cfg.ep, n_local)
    if rank == "first":
        return per_rank[0].copy()
    if rank != "worst":
        raise ValueError(f"unknown rank selector {rank!r}")
    # EP is straggler-gated: the step waits on the slowest rank, and cost
    # tracks active experts (weight stream), not rows. So the rank worth
    # modelling is the one with the most active experts.
    active_per_rank = (per_rank > 0).sum(axis=1)
    return per_rank[
        int(np.lexsort((per_rank.sum(axis=1), active_per_rank))[-1])
    ].copy()


def weight_bytes(active_experts: int, n: int, k: int) -> int:
    """MXFP8 packed weight bytes + E8M0 scale bytes for ACTIVE experts."""
    packed = active_experts * n * k
    scales = active_experts * n * (k // _SCALE_BLOCK)
    return packed + scales


def _cold_target_bytes() -> float:
    return 1.5 * _L2_CACHE_SIZE_BYTES


def _auto_ncopies(active_weight_bytes: int) -> int:
    return max(2, math.ceil(_cold_target_bytes() / max(1, active_weight_bytes)))


def _chain_quant_config() -> QuantConfig:
    from max.dtype import DType
    from max.nn.quant_config import (
        InputScaleSpec,
        QuantConfig,
        QuantFormat,
        ScaleGranularity,
        ScaleOrigin,
        WeightScaleSpec,
    )

    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 32),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 32),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.MXFP8,
    )


def bench_chain(
    m: int,
    cfg: M3Config,
    num_iters: int,
    ncopies: int,
    check: bool,
    decode_grid_m_cap: int,
    model_cache: dict,
    routing: str = "multinomial",
    active_override: int = 0,
    mtp: int = 1,
    rank: str = "worst",
) -> list[dict]:
    """One EP-rank chain measurement per variant on IDENTICAL inputs."""
    import torch
    from max.driver import Accelerator, Buffer
    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, TensorType, ops
    from max.nn.comm.ep.ep_kernels import fused_silu_quantized
    from max.nn.kernels import (
        block_scaled_preshuffle_b_5d,
        grouped_dynamic_block_scaled_matmul_amd,
        grouped_dynamic_block_scaled_matmul_amd_swiglu_quant,
    )

    est_total_m = m
    decode_grid_m_rows = _decode_grid_m_rows(decode_grid_m_cap, cfg)
    n1, k1 = cfg.hidden, cfg.hidden  # gate_up N=6144 K=6144
    n2, k2 = cfg.hidden, cfg.inter  # down N=6144 K=3072

    routed_counts = [
        int(c)
        for c in rank_counts(
            m, cfg, routing, active=active_override, mtp=mtp, rank=rank
        )
    ]
    # A zero-row group costs nothing: the kernel derives M from the offsets and
    # returns before touching B. So active-expert count, not row count, is what
    # the weight stream is priced on.
    active_routed = sum(1 for c in routed_counts if c > 0)
    shared_rows = _ceil_div(m, cfg.ep)
    group_rows = [shared_rows] + routed_counts
    n_groups = len(group_rows)
    total_rows = int(sum(group_rows))
    active = int(sum(1 for r in group_rows if r > 0))

    combined_wbytes = weight_bytes(active, n1, k1) + weight_bytes(
        active, n2, k2
    )
    if ncopies <= 0:
        ncopies = _auto_ncopies(combined_wbytes)
    cold_ws = ncopies * combined_wbytes
    assert cold_ws > _cold_target_bytes(), (
        f"cold working set {cold_ws} <= 1.5*L2 ({_cold_target_bytes():.0f}); "
        "increase --ncopies"
    )

    max_group_rows = max(group_rows)
    stride = _ceil_div(max_group_rows, _SCALE_BLOCK) * _SCALE_BLOCK
    storage_rows = max(total_rows, stride)
    k1_bytes, k2_bytes = k1, k2  # MXFP8: one byte per element
    k1_scales = k1 // _SCALE_BLOCK

    def _rand_fp8_bytes(shape: tuple[int, ...], seed: int) -> torch.Tensor:
        """Random e4m3 bytes EXCLUDING NaN encodings 0x7F/0xFF."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        r = torch.randint(
            0, 254, shape, dtype=torch.uint8, device="cuda", generator=g
        )
        return r + (r >= 0x7F).to(torch.uint8)

    # Inputs shared by all variants. Activations arrive already quantized
    # (production: EP dispatch output). A-scales uniform E8M0 127 in the slot
    # layout: value-invariant under the slot swizzle.
    act = torch.zeros(storage_rows, k1_bytes, dtype=torch.uint8, device="cuda")
    act[:total_rows] = _rand_fp8_bytes((total_rows, k1_bytes), 7)
    a_slot_scales = torch.full(
        (n_groups * stride, k1_scales), 127, dtype=torch.uint8, device="cuda"
    )
    row_off = torch.zeros(n_groups + 1, dtype=torch.uint32)
    acc = 0
    for gi, r in enumerate(group_rows):
        acc += r
        row_off[gi + 1] = acc
    expert_ids = torch.from_numpy(np.arange(n_groups, dtype=np.int32))
    usage = torch.tensor([max_group_rows, n_groups], dtype=torch.uint32)

    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()

    # Claim torch headroom before MAX's allocator claims free VRAM.
    _ballast = torch.empty(
        max(8 << 30, 2 * ncopies * combined_wbytes + (4 << 30)),
        dtype=torch.uint8,
        device="cuda",
    )

    skey = ("chain-session",)
    if skey not in model_cache:
        model_cache[skey] = InferenceSession(devices=[Accelerator()])
    session = model_cache[skey]

    def _preshuffle_model(n: int, k_bytes: int) -> Model:
        pkey = ("preshuffle", n_groups, n, k_bytes)
        if pkey not in model_cache:
            w_t = TensorType(DType.uint8, (n_groups, n, k_bytes), device=gpu)
            with Graph(
                f"m3_b_preshuffle_{n}x{k_bytes}", input_types=[w_t]
            ) as g:
                g.output(block_scaled_preshuffle_b_5d(g.inputs[0].tensor))
            model_cache[pkey] = session.load(g)
        return model_cache[pkey]

    def _shuffle(pre_model: Model, w: torch.Tensor) -> torch.Tensor:
        out = pre_model.execute(Buffer.from_dlpack(w))
        torch.cuda.synchronize()
        return torch.from_dlpack(out[0]).clone()

    def _gen_weight(
        e: int, n: int, k: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = _rand_fp8_bytes((e, n, k), seed)
        ws = torch.full(
            (e, n, k // _SCALE_BLOCK), 127, dtype=torch.uint8, device="cuda"
        )
        return w, ws

    # sigma(2i) = i, sigma(2i+1) = H + i: the N-axis permutation that turns
    # the [gate || up] weight into the interleaved (gate_i, up_i) pairs the
    # fused-SwiGLU epilogue needs, since gate and up must land in the same
    # output tile. Production applies it in the weight adapter at load time.
    sigma = torch.empty(n1, dtype=torch.long, device="cuda")
    sigma[0::2] = torch.arange(n1 // 2, device="cuda")
    sigma[1::2] = torch.arange(n1 // 2, device="cuda") + n1 // 2

    wcopies = []
    for c in range(ncopies):
        gu_w, gu_ws = _gen_weight(n_groups, n1, k1, seed=100 + c)
        dn_w, dn_ws = _gen_weight(n_groups, n2, k2, seed=500 + c)
        wcopies.append(
            {
                "gu": _shuffle(_preshuffle_model(n1, k1_bytes), gu_w),
                "gu_sig": _shuffle(
                    _preshuffle_model(n1, k1_bytes), gu_w[:, sigma, :]
                ),
                # The scales are a uniform E8M0 127, so sigma is the identity
                # on them; both gate_up weights share this tensor.
                "gu_ws": gu_ws,
                "down": _shuffle(_preshuffle_model(n2, k2_bytes), dn_w),
                "dn_ws": dn_ws,
            }
        )
    del _ballast

    qcfg = _chain_quant_config()

    def _matmul(
        a: TensorValue,
        w: TensorValue,
        a_sc: TensorValue,
        b_sc: TensorValue,
        off: TensorValue,
        ids: TensorValue,
        usage_in: TensorValue,
        est_m: TensorValue,
        slot: bool,
    ) -> TensorValue:
        return grouped_dynamic_block_scaled_matmul_amd(
            a,
            w,
            a_sc,
            b_sc,
            off,
            ids,
            usage_in,
            out_type=DType.bfloat16,
            estimated_total_m=est_m,
            preshuffled_b=True,
            a_scales_preshuffled=slot,
            a_scales_max_padded_m=stride if slot else 0,
            decode_grid_m_cap=decode_grid_m_cap,
            decode_grid_m_rows=decode_grid_m_rows,
        )

    # `gu_sig` is the same weight under the sigma N-permutation; only the
    # fused-SwiGLU variant reads it.

    def _est_m() -> TensorValue:
        return ops.constant(est_total_m, dtype=DType.uint32, device=cpu)

    def _build_plain(fold: bool) -> ChainBuilder:
        def build(
            act_in: TensorValue,
            a_sc: TensorValue,
            off: TensorValue,
            ids: TensorValue,
            usage_in: TensorValue,
            gu_w: TensorValue,
            gu_sig: TensorValue,
            gu_ws: TensorValue,
            dn_w: TensorValue,
            dn_ws: TensorValue,
        ) -> ChainOut:
            est_m = _est_m()
            gate_up = _matmul(
                act_in, gu_w, a_sc, gu_ws, off, ids, usage_in, est_m, slot=True
            )
            c_packed, c_scales = fused_silu_quantized(
                gate_up,
                off,
                qcfg,
                DType.float8_e4m3fn,
                max_padded_M=stride if fold else 0,
                clamp_activation=True,
                swiglu_alpha=_SWIGLU_ALPHA,
                swiglu_limit=_SWIGLU_LIMIT,
            )
            down = _matmul(
                c_packed,
                dn_w,
                c_scales,
                dn_ws,
                off,
                ids,
                usage_in,
                est_m,
                slot=fold,
            )
            return down, c_packed, c_scales

        return build

    def _build_fused_mxfp8() -> ChainBuilder:
        """Two launches: gemm1 folds the SwiGLU AND the MXFP8 requantize into
        its epilogue, emitting packed E4M3 + E8M0 scales; gemm2 is a plain
        w8a8 matmul over them. No activation kernel and no bf16 intermediate,
        and no regime gate, since A is quantized once here rather than once
        per gemm2 output tile."""

        def build(
            act_in: TensorValue,
            a_sc: TensorValue,
            off: TensorValue,
            ids: TensorValue,
            usage_in: TensorValue,
            gu_w: TensorValue,
            gu_sig: TensorValue,
            gu_ws: TensorValue,
            dn_w: TensorValue,
            dn_ws: TensorValue,
        ) -> ChainOut:
            est_m = _est_m()
            c_packed, c_scales = (
                grouped_dynamic_block_scaled_matmul_amd_swiglu_quant(
                    act_in,
                    gu_sig,
                    a_sc,
                    gu_ws,
                    off,
                    ids,
                    usage_in,
                    estimated_total_m=est_m,
                    decode_grid_m_cap=decode_grid_m_cap,
                    decode_grid_m_rows=decode_grid_m_rows,
                    out_scales_max_padded_m=stride,
                )
            )
            down = _matmul(
                c_packed,
                dn_w,
                c_scales,
                dn_ws,
                off,
                ids,
                usage_in,
                est_m,
                slot=True,
            )
            return down, c_packed, c_scales

        return build

    builders = {
        "plain+preshuffle": _build_plain(fold=False),
        "plain+fold": _build_plain(fold=True),
        "fused_mxfp8": _build_fused_mxfp8(),
    }
    variants = _CHAIN_VARIANTS
    if _VARIANT_SEL:
        missing = set(_VARIANT_SEL) - set(variants)
        if missing:
            raise SystemExit(f"unknown variant(s) {sorted(missing)}")
        variants = tuple(v for v in variants if v in _VARIANT_SEL)
    assert set(variants) <= set(builders), sorted(set(variants) - set(builders))

    def _load_chain_model(variant: str) -> tuple[Model, bool]:
        key = (
            "chain",
            variant,
            total_rows,
            storage_rows,
            ncopies,
            est_total_m,
            n_groups,
            stride,
            decode_grid_m_cap,
        )
        if key in model_cache:
            return model_cache[key]
        ab_dtype = DType.float8_e4m3fn
        a_t = TensorType(ab_dtype, (storage_rows, k1_bytes), device=gpu)
        a_sc_t = TensorType(
            DType.float8_e8m0fnu, (n_groups * stride, k1_scales), device=gpu
        )
        off_t = TensorType(DType.uint32, (n_groups + 1,), device=gpu)
        ids_t = TensorType(DType.int32, (n_groups,), device=gpu)
        usage_t = TensorType(DType.uint32, (2,), device=cpu)
        gu_w_t = TensorType(ab_dtype, (n_groups, n1, k1_bytes), device=gpu)
        gu_ws_t = TensorType(
            DType.float8_e8m0fnu, (n_groups, n1, k1_scales), device=gpu
        )
        dn_w_t = TensorType(ab_dtype, (n_groups, n2, k2_bytes), device=gpu)
        dn_ws_t = TensorType(
            DType.float8_e8m0fnu, (n_groups, n2, k2 // _SCALE_BLOCK), device=gpu
        )
        _build = builders[variant]
        with Graph(
            f"m3_moe_chain_{variant.replace('+', '_')}",
            input_types=[
                a_t,
                a_sc_t,
                off_t,
                ids_t,
                usage_t,
                *([gu_w_t, gu_w_t, gu_ws_t, dn_w_t, dn_ws_t] * ncopies),
            ],
        ) as graph:
            ins = graph.inputs
            act_in, a_sc, off, ids, usage_in = (ins[i].tensor for i in range(5))
            _x = 5 + 5 * ncopies
            downs = []
            probe = []
            has_packed = False
            for c in range(ncopies):
                base = 5 + 5 * c
                down, c_packed, c_scales = _build(
                    act_in,
                    a_sc,
                    off,
                    ids,
                    usage_in,
                    ins[base].tensor,
                    ins[base + 1].tensor,
                    ins[base + 2].tensor,
                    ins[base + 3].tensor,
                    ins[base + 4].tensor,
                )
                downs.append(down)
                if c == 0:
                    has_packed = c_packed is not None
                    probe = [t for t in (c_packed, c_scales) if t is not None]
            graph.output(*downs, *probe)
        model_cache[key] = (session.load(graph), has_packed)
        return model_cache[key]

    def _dl(t: torch.Tensor) -> Buffer:
        return Buffer.from_dlpack(t)

    dev = Accelerator()

    def _bind_inputs() -> tuple[Buffer, ...]:
        fixed = [
            _dl(act).view(DType.float8_e4m3fn),
            _dl(a_slot_scales).view(DType.float8_e8m0fnu),
            _dl(row_off).to(dev),
            _dl(expert_ids).to(dev),
            _dl(usage),
        ]
        weights = []
        for wc in wcopies:
            weights += [
                _dl(wc["gu"]).view(DType.float8_e4m3fn),
                _dl(wc["gu_sig"]).view(DType.float8_e4m3fn),
                _dl(wc["gu_ws"]).view(DType.float8_e8m0fnu),
                _dl(wc["down"]).view(DType.float8_e4m3fn),
                _dl(wc["dn_ws"]).view(DType.float8_e8m0fnu),
            ]
        return (*fixed, *weights)

    models = {v: _load_chain_model(v) for v in variants}
    inputs = _bind_inputs()

    check_note = ""
    if check:
        ref = None
        compared = 0
        for variant in variants:
            model, _has_packed = models[variant]
            outs = model.execute(*inputs)
            torch.cuda.synchronize()
            down0 = (
                torch.from_dlpack(outs[0])[:total_rows]
                .clone()
                .to(torch.float32)
            )
            if not torch.isfinite(down0).all():
                raise RuntimeError(f"{variant}: non-finite down output")
            if ref is None:
                ref = down0
                continue
            compared += 1
            tol = _CHECK_REL_TOL.get(variant)
            if tol is None:
                if not torch.equal(down0, ref):
                    raise RuntimeError(
                        f"{variant}: down output differs from reference "
                        f"(chain variants must be byte-identical on identical "
                        "inputs)"
                    )
                continue
            # Scale-relative, not element-relative: the output has entries
            # arbitrarily close to zero, against which any absolute difference
            # reads as enormous.
            scale = ref.abs().max().clamp_min(1e-6)
            rel = ((down0 - ref).abs().max() / scale).item()
            print(f"    {variant}: worst |delta| / max|ref| = {rel:.5f}")
            if os.environ.get("CHAIN_DEBUG_ROWS"):
                per_row = (down0 - ref).abs().max(dim=1).values / scale
                bad = (per_row > tol).nonzero().flatten().tolist()
                print(
                    f"      shared_rows={shared_rows} total={total_rows} "
                    f"nbad={len(bad)} first={bad[:8]} last={bad[-8:]}"
                )
            if rel > tol:
                raise RuntimeError(
                    f"{variant}: down output differs from reference by "
                    f"{rel:.5f} > {tol} of the output's scale"
                )
        check_note = "/check-ok" if compared else "/finite-only"

    results = []
    for variant in variants:
        model, _ = models[variant]
        med_s, best_s = _time_max_replay(
            model, inputs, num_iters, ncopies, torch
        )
        notes = (
            f"chain/{variant}/mxfp8/gemm1+act+gemm2/"
            f"launches={_CHAIN_LAUNCHES.get(variant, 0)}/uniform-a-scales/"
            f"stride={stride}/cap={decode_grid_m_cap}/rows={decode_grid_m_rows}"
            f"/routing={routing}/mtp={mtp}/rank={rank}"
            f"/active={1 + active_routed}"
            f"{check_note}/fused-shared"
        )
        results.append(
            {
                "variant": variant,
                "m": m,
                "median_s": med_s,
                "best_s": best_s,
                "total_rows": total_rows,
                "active_groups": active,
                "active_routed": active_routed,
                "routing": routing,
                "shared_rows": shared_rows,
                "routed_rows": total_rows - shared_rows,
                "ncopies": ncopies,
                "cold_ws": cold_ws,
                "launches": _CHAIN_LAUNCHES.get(variant, 0),
                "notes": notes,
            }
        )
    return results


def _time_max_replay(
    model: Model,
    graph_inputs: tuple[Buffer, ...],
    num_iters: int,
    ncopies: int,
    torch: ModuleType,
) -> tuple[float, float]:
    """Capture the chained graph and time back-to-back replay with CUDA
    events. Per-chain time = whole-graph / ncopies. No CPU timers."""
    global _MAX_CAPTURE_KEY

    nrun = max(num_iters, 200)
    capture_key = _MAX_CAPTURE_KEY
    _MAX_CAPTURE_KEY += 1
    try:
        model.capture(capture_key, *graph_inputs)
        model.debug_verify_replay(capture_key, *graph_inputs)
        run = lambda: model.replay(capture_key, *graph_inputs)
    except Exception as e:
        print(f"    (device-graph capture failed, eager: {e})")
        run = lambda: model.execute(*graph_inputs)
    torch.cuda.synchronize()
    for _ in range(50):
        run()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    reps = 5
    for _ in range(reps):
        start.record()
        for _ in range(nrun // reps):
            run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3 / (nrun // reps) / ncopies)
    times.sort()
    return times[len(times) // 2], times[0]


def dry_run(args: argparse.Namespace, cfg: M3Config) -> None:
    for m in [int(x) for x in args.M.split(",")]:
        counts = rank_counts(m, cfg, args.routing, mtp=args.mtp, rank=args.rank)
        shared = _ceil_div(m, cfg.ep)
        total = shared + int(counts.sum())
        active = 1 + int((counts > 0).sum())
        wbytes = weight_bytes(active, cfg.hidden, cfg.hidden) + weight_bytes(
            active, cfg.hidden, cfg.inter
        )
        ncopies = _auto_ncopies(wbytes)
        print(
            f"M={m}: shared={shared} routed={int(counts.sum())} "
            f"total_rows={total} active_groups={active} "
            f"ncopies={ncopies} cold_ws={ncopies * wbytes / 1e6:.0f}MB "
            f"grid_cap={args.decode_grid_m_cap} "
            f"grid_rows={_decode_grid_m_rows(args.decode_grid_m_cap, cfg)}"
        )
    print("\nDry-run OK: routing -> shape mapping validated.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="MiniMax-M3 decode MoE FFN chain bench (MAX, gfx950)"
    )
    p.add_argument("--M", "--m", default="16,48,128")
    p.add_argument("--num-iters", "--num_iters", type=int, default=200)
    p.add_argument(
        "--ncopies",
        type=int,
        default=0,
        help="0 = auto-size from 1.5x L2 over combined chain bytes",
    )
    p.add_argument(
        "--decode-grid-m-cap",
        type=int,
        default=None,
        help="Capped decode grid; default = min(max(M), 96)",
    )
    p.add_argument(
        "--routing",
        choices=("multinomial", "uniform"),
        default="multinomial",
        help=(
            "router model. multinomial = real draw, leaves experts empty. "
            "uniform = worst case, every local expert active at any M."
        ),
    )
    p.add_argument(
        "--variants",
        default="",
        help=(
            "comma-separated subset of chain variants to build; empty = all. "
            "Prefill graphs are slow to compile, so narrow this."
        ),
    )
    p.add_argument(
        "--rank",
        choices=("worst", "first"),
        default="worst",
        help=(
            "which EP rank to model. EP is straggler-gated, so the default is "
            "the rank with the most active experts, not an average one."
        ),
    )
    p.add_argument(
        "--mtp",
        type=int,
        default=1,
        help=(
            "tokens per request; routing is drawn per request, so M/mtp "
            "independent draws. Production decode is BS24 x mtp 4 = M 96."
        ),
    )
    p.add_argument(
        "--active-experts",
        type=int,
        default=0,
        help=(
            "force exactly N routed experts active, holding tokens fixed; "
            "0 = let --routing decide"
        ),
    )
    p.add_argument(
        "--check", action="store_true", help="byte-identity A/B across variants"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-o", "--output", default="chain.csv")
    args = p.parse_args()

    cfg = M3Config()
    if args.dry_run:
        if args.decode_grid_m_cap is None:
            args.decode_grid_m_cap = min(
                max(int(x) for x in args.M.split(",")), 96
            )
        dry_run(args, cfg)
        return

    global _VARIANT_SEL
    _VARIANT_SEL = [v for v in args.variants.split(",") if v]
    m_values = [int(x) for x in args.M.split(",")]
    if args.decode_grid_m_cap is None:
        args.decode_grid_m_cap = min(max(m_values), 96)

    model_cache: dict = {}
    import csv

    rows = []
    for m in m_values:
        print(f"[workload] M={m} cap={args.decode_grid_m_cap}")
        rows.extend(
            bench_chain(
                m,
                cfg,
                args.num_iters,
                args.ncopies,
                args.check,
                args.decode_grid_m_cap,
                model_cache,
                args.routing,
                args.active_experts,
                args.mtp,
                args.rank,
            )
        )

    fields = [
        "variant",
        "m",
        "median_s",
        "best_s",
        "total_rows",
        "active_groups",
        "active_routed",
        "routing",
        "shared_rows",
        "routed_rows",
        "ncopies",
        "cold_ws",
        "launches",
        "notes",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {len(rows)} rows to {args.output}")

    print("\n=== chain (gemm1-start -> gemm2-end, per rank) ===")
    for r in rows:
        us = r["median_s"] * 1e6
        best_us = r["best_s"] * 1e6
        print(
            f"M={r['m']:4d} {r['variant']:18s} "
            f"median={us:8.2f}us best={best_us:8.2f}us "
            f"launches={r['launches']} rows={r['total_rows']}"
        )


if __name__ == "__main__":
    main()
