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
"""Checks MiMo-V2's serving-mode (W4A8) MoE on one production-width layer.

One layer of 256 random MXFP4 experts (hidden 4096, width 2048) on one GPU,
run on a 4,096-token prefill, with the routing given as an input so that it
can be uniform or all on 8 experts:

* the adapter's E8M0 interleave equals the kernel library's own interleave;
* each grouped matmul's usage stats: element 1, the loop bound, is the
  constant the strategy builds from the 256 expert slots, and element 0, read
  back from the graph, is tokens x 8;
* with element 0 forced to 1, the expert outputs match tokens x 8's within
  BF16 tolerance and no row is zero, so element 0 picks a tile configuration
  and does not bound the rows each expert computes; the two run times show
  that element 0 reached the kernel;
* against an exact host reference (weights dequantized, float32 GEMMs, BF16
  activations) on a sample of rows, the error that MXFP8 activations add.

It needs a B200, so it runs by hand, never in CI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorType
from max.nn.kernels import block_scales_interleave
from max.nn.moe import quant_strategy
from max.pipelines.architectures.mimo_v2.layers.moe import MiMoV2MoE
from max.pipelines.architectures.mimo_v2.quant import parse_quant_scheme
from max.pipelines.architectures.mimo_v2.weight_adapters import (
    interleave_e8m0,
)
from transformers.configuration_utils import PretrainedConfig

EXPERTS, TOP_K, HIDDEN, WIDTH = 256, 8, 4096, 2048


def _quant_config() -> Any:
    config = PretrainedConfig(
        num_hidden_layers=2,
        moe_layer_freq=[0, 1],
        n_routed_experts=EXPERTS,
        conversion_metadata={"qkv_layout": "global_q_k_v"},
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "quantized_layers": {
                f"model.layers.1.mlp.experts.{e}.{p}": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
                for e in range(EXPERTS)
                for p in ("gate_proj", "up_proj", "down_proj")
            },
        },
    )
    return parse_quant_scheme(config).experts


def _weights(
    rng: np.random.Generator,
) -> dict[str, npt.NDArray[Any]]:
    """Random MXFP4 experts and router, with row-major E8M0 scales."""

    def e8m0(rows: int, cols: int) -> npt.NDArray[np.uint8]:
        # 2^-6 to 2^-3: E2M1 values up to 0.75, as in the real checkpoint.
        return rng.integers(121, 125, (EXPERTS, rows, cols), dtype=np.uint8)

    return {
        "experts_gate_up_proj": rng.integers(
            0, 256, (EXPERTS, 2 * WIDTH, HIDDEN // 2), dtype=np.uint8
        ),
        "experts_gate_up_proj_scale": e8m0(2 * WIDTH, HIDDEN // 32),
        "experts_down_proj": rng.integers(
            0, 256, (EXPERTS, HIDDEN, WIDTH // 2), dtype=np.uint8
        ),
        "experts_down_proj_scale": e8m0(HIDDEN, WIDTH // 32),
        "gate.gate_score.weight": rng.standard_normal(
            (EXPERTS, HIDDEN), dtype=np.float32
        )
        * 0.02,
        "gate.e_score_correction_bias": np.zeros(EXPERTS, dtype=np.float32),
    }


def _state(weights: dict[str, npt.NDArray[Any]]) -> dict[str, Buffer]:
    """The serving-mode weights, scales interleaved as the adapter stores
    them."""
    state = {}
    for name, array in weights.items():
        buffer = Buffer.from_numpy(np.ascontiguousarray(array))
        if name.endswith("_scale"):
            interleaved = np.stack([interleave_e8m0(e) for e in array])
            buffer = Buffer.from_numpy(interleaved).view(DType.float8_e8m0fnu)
        state[name] = buffer
    return state


def _moe(
    weights: dict[str, npt.NDArray[Any]],
) -> tuple[MiMoV2MoE, dict[str, Any]]:
    """The serving-mode layer, loaded, and its weights registry."""
    moe = MiMoV2MoE(
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_dim=WIDTH,
        norm_topk_prob=True,
        quant_config=_quant_config(),
        devices=[DeviceRef.GPU(0)],
    )
    moe.sharding_strategy = ShardingStrategy.tensor_parallel(1)
    moe.load_state_dict(_state(weights), weight_alignment=1, strict=True)
    return moe, moe.state_dict(auto_initialize=False)


def _bf16(buffer: Any) -> npt.NDArray[np.float32]:
    bits = buffer.to(CPU()).view(DType.uint16).to_numpy().astype(np.uint32)
    return (bits << 16).view(np.float32)


def check_interleave(
    session: InferenceSession, weights: dict[str, npt.NDArray[Any]]
) -> dict[str, Any]:
    """The adapter's interleave against the kernel library's, per stack."""
    result = {}
    for name in ("experts_gate_up_proj_scale", "experts_down_proj_scale"):
        block = weights[name][7]
        with Graph(
            "interleave",
            input_types=[
                TensorType(
                    DType.float8_e8m0fnu, block.shape, device=DeviceRef.GPU(0)
                )
            ],
        ) as graph:
            graph.output(block_scales_interleave(graph.inputs[0].tensor, 32))
        model = session.load(graph)
        device_input = (
            Buffer.from_numpy(block)
            .view(DType.float8_e8m0fnu)
            .to(session.devices[0])
        )
        (out,) = model.execute(device_input)
        got = out.to(CPU()).view(DType.uint8).to_numpy()
        want = interleave_e8m0(block)
        result[name] = {
            "shape": list(got.shape),
            "bytes_differ": int(np.count_nonzero(got != want)),
        }
    return result


# The token count is symbolic, as in the served graph: the grouped matmul
# must see element 0 as a run-time value, not a folded constant.
_INPUTS = [
    TensorType(DType.bfloat16, ["tokens", HIDDEN], device=DeviceRef.GPU(0)),
    TensorType(DType.int32, ["tokens", TOP_K], device=DeviceRef.GPU(0)),
]


def build_serving(moe: MiMoV2MoE) -> tuple[Graph, list[dict[str, Any]]]:
    """The W4A8 expert rows with element 0 derived and with element 0 given
    as an input, and the derived element 0, as graph outputs.

    Also returns what each grouped matmul was given for element 1, which is a
    graph constant: the printed constant op and the expert-slot count the
    strategy builds it from.
    """
    with mock.patch.object(
        quant_strategy,
        "grouped_matmul_block_scaled",
        wraps=quant_strategy.grouped_matmul_block_scaled,
    ) as spy:
        with Graph(
            "mimo_v2_w4a8_moe",
            input_types=[
                *_INPUTS,
                TensorType(DType.uint32, [], device=DeviceRef.CPU()),
            ],
        ) as graph:
            shard = moe.shard([DeviceRef.GPU(0)])[0]
            x, idx, element_0 = (v.tensor for v in graph.inputs)
            derived = shard._w4a8_experts(x, idx)
            labels = ["derived"] * spy.call_count
            one = shard._w4a8_experts(x, idx, estimated_total_m=element_0)
            labels += ["one"] * (spy.call_count - len(labels))
            calls = [
                {
                    "variant": label,
                    "expert_slots": int(call.args[6].shape[0]),
                    "usage_stats_op": str(call.args[8]._mlir_value.owner),
                }
                for label, call in zip(labels, spy.call_args_list, strict=True)
            ]
            graph.output(
                derived, one, spy.call_args_list[0].kwargs["estimated_total_m"]
            )
    return graph, calls


_E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)


def _dequantize(
    codes: npt.NDArray[np.uint8], e8m0: npt.NDArray[np.uint8]
) -> npt.NDArray[np.float32]:
    """One expert's ``[N, K/2]`` E2M1 codes, low nibble first, times its
    ``[N, K/32]`` E8M0 scales."""
    values = np.stack([_E2M1[codes & 0xF], _E2M1[codes >> 4]], axis=-1)
    values = values.reshape(codes.shape[0], -1, 32)
    scales = np.exp2(e8m0.astype(np.float32) - 127)[..., None]
    return (values * scales).reshape(codes.shape[0], -1)


def host_reference(
    weights: dict[str, npt.NDArray[Any]],
    x: npt.NDArray[np.float32],
    idx: npt.NDArray[np.int64],
    rows: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """The exact expert outputs of ``rows``, ``[len(rows) * 8, hidden]`` in
    flat routing order, with the SiLU input and output rounded to BF16 as
    the served graph rounds them."""
    out = np.zeros((len(rows), TOP_K, HIDDEN), dtype=np.float32)
    for expert in np.unique(idx[rows]):
        where = np.argwhere(idx[rows] == expert)
        inputs = x[rows[where[:, 0]]]
        gate_up = _bf16_round(
            inputs
            @ _dequantize(
                weights["experts_gate_up_proj"][expert],
                weights["experts_gate_up_proj_scale"][expert],
            ).T
        )
        gate, up = gate_up[:, :WIDTH], gate_up[:, WIDTH:]
        hidden = _bf16_round(gate / (1 + np.exp(-gate)) * up)
        out[where[:, 0], where[:, 1]] = (
            hidden
            @ _dequantize(
                weights["experts_down_proj"][expert],
                weights["experts_down_proj_scale"][expert],
            ).T
        )
    return out.reshape(-1, HIDDEN)


def _bf16_round(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def build_timed(moe: MiMoV2MoE, given_element_0: bool) -> Graph:
    """The W4A8 expert rows alone, element 0 derived or given."""
    extra = [TensorType(DType.uint32, [], device=DeviceRef.CPU())]
    with Graph(
        "mimo_v2_w4a8_timed",
        input_types=[*_INPUTS, *(extra if given_element_0 else [])],
    ) as graph:
        shard = moe.shard([DeviceRef.GPU(0)])[0]
        x, idx, *element_0 = (v.tensor for v in graph.inputs)
        graph.output(
            shard._w4a8_experts(
                x, idx, estimated_total_m=element_0[0] if element_0 else None
            )
        )
    return graph


def _milliseconds(model: Any, feeds: list[Buffer], runs: int = 20) -> float:
    """Median wall time of one execute, synchronized by a copy to the host."""
    times = []
    for _ in range(runs + 3):
        t0 = time.perf_counter()
        (out,) = model.execute(*feeds)
        out.to(CPU())
        times.append(time.perf_counter() - t0)
    return float(np.median(times[3:]) * 1e3)


def _row_stats(
    got: npt.NDArray[np.float32], want: npt.NDArray[np.float32]
) -> dict[str, float]:
    a, b = got.astype(np.float64), want.astype(np.float64)
    norm = np.linalg.norm(b, axis=-1)
    rel = np.linalg.norm(a - b, axis=-1) / np.where(norm > 0, norm, 1)
    cos = 1 - (a * b).sum(-1) / np.maximum(
        np.linalg.norm(a, axis=-1) * norm, 1e-30
    )
    return {
        "max_abs": float(np.abs(a - b).max()),
        "rel_l2_mean": float(rel.mean()),
        "rel_l2_max": float(rel.max()),
        "cos_dist_mean": float(cos.mean()),
        "cos_dist_max": float(cos.max()),
        "bit_identical": bool(
            (got.view(np.uint32) == want.view(np.uint32)).all()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokens = args.tokens
    rng = np.random.default_rng(0)
    weights = _weights(rng)
    session = InferenceSession(devices=[Accelerator(0)])
    results: dict[str, Any] = {"tokens": tokens}
    results["interleave"] = check_interleave(session, weights)
    print("interleave:", json.dumps(results["interleave"]), flush=True)

    t0 = time.time()
    w4a8, w4a8_registry = _moe(weights)
    graph, calls = build_serving(w4a8)
    results["grouped_matmul_calls"] = calls
    print("calls:", json.dumps(calls), flush=True)
    serving = session.load(graph, weights_registry=w4a8_registry)
    # A module's shards belong to the first graph that uses them.
    timed = {
        given: session.load(
            build_timed(_moe(weights)[0], given),
            weights_registry=w4a8_registry,
        )
        for given in (False, True)
    }
    print(f"compiled in {time.time() - t0:.0f} s", flush=True)

    device = session.devices[0]
    x = _bf16_round(rng.standard_normal((tokens, HIDDEN), dtype=np.float32))
    x_bf16 = Buffer.from_numpy(
        (x.view(np.uint32) >> 16).astype(np.uint16)
    ).view(DType.bfloat16)
    sample = np.sort(rng.choice(tokens, size=256, replace=False))
    flat_sample = (sample[:, None] * TOP_K + np.arange(TOP_K)).reshape(-1)
    routings = {
        # Each token's 8 distinct experts, uniform over the 256.
        "uniform": np.argsort(rng.random((tokens, EXPERTS)), axis=-1)[
            :, :TOP_K
        ],
        # Every token on experts 0-7: 4,096 rows each, 248 experts empty.
        "all_on_8": np.tile(np.arange(TOP_K), (tokens, 1)),
    }
    for name, idx in routings.items():
        feeds = [
            x_bf16.to(device),
            Buffer.from_numpy(idx.astype(np.int32)).to(device),
        ]
        one_as_element_0 = Buffer.from_numpy(np.array(1, dtype=np.uint32))
        derived, one, total_m = serving.execute(*feeds, one_as_element_0)
        derived_rows, one_rows = _bf16(derived), _bf16(one)
        t0 = time.time()
        exact = host_reference(weights, x, idx, sample)
        print(f"host reference in {time.time() - t0:.0f} s", flush=True)
        derived_total_m = int(total_m.to(CPU()).to_numpy())
        results[name] = {
            "ms_element_0_derived": _milliseconds(timed[False], feeds),
            "ms_element_0_one": _milliseconds(
                timed[True], [*feeds, one_as_element_0]
            ),
            "element_0_derived": derived_total_m,
            "element_0_derived_is_tokens_x_8": derived_total_m
            == tokens * TOP_K,
            "element_1_slots_are_256": all(
                c["expert_slots"] == EXPERTS for c in calls
            ),
            "zero_rows_derived": int((np.abs(derived_rows).max(-1) == 0).sum()),
            "zero_rows_element_0_one": int(
                (np.abs(one_rows).max(-1) == 0).sum()
            ),
            "element_0_one_vs_derived": _row_stats(one_rows, derived_rows),
            "w4a8_vs_exact_sampled_rows": _row_stats(
                derived_rows[flat_sample], exact
            ),
        }
        print(f"{name}:", json.dumps(results[name]), flush=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
