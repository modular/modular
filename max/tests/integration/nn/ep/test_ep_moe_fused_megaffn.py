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
"""EP8 A/B gate: the one-launch fused EP MoE against the shipping NVFP4 chain.

Arm A is the shipping expert-parallel NVFP4 MoE (``ep.dispatch_async`` ->
side-stream shared expert -> ``ep.dispatch_wait`` -> ``mega_ffn.ep_combine_send``
-> ``ep.combine_wait``). Arm B is ``mega_ffn.ep_fused`` (dispatch, MegaFFN and
the weighted combine in ONE launch per rank) with the shared expert inline on
the default stream. Same weights, same routing, same inputs, same devices.

UNRUN DRAFT (needs 8 GPUs). This is not a qualified EP8 gate and not a model
benchmark; review its construction before relying on it:

* It chains two ``forward_moe_sharded_layers`` calls on ONE module. In
  ``test_ep_moe_fused_megaffn_ep2.py`` that pattern reused the shared
  expert's memoized weight concat across side-stream regions (an MLIR
  dominance error), so the EP2 test builds one module per layer (its
  ``_Stack``).
* Its shipping reference runs both layers in one graph. On an older main the
  NVFP4 two-layer shipping graph returned non-finite rows on its second
  execution; the EP2 test's two-layer shipping check passes on the current
  main, which has not been tried at EP8.

Geometry is GLM-5.3's MoE (hidden 6144, 256 routed experts, moe 2048, top-k 8)
at EP8 (32 local experts) with the recipe's decode bound of 24 tokens per rank.
Two MoE layers are chained in one graph so consecutive fused launches share
one workspace (the production layout), and every cell runs several executions
so the EP generation parity alternates and the mailbox generation advances.
The ``bs1_q6`` cell is the batch-1 MTP verify step: six ranks hold one token
and two ranks hold none, which the fused kernel must still serve.

The second test replays ONE real GLM-5.3 MoE layer (its 256 NVFP4 routed
experts with their real E4M3 block scales, per-tensor ``weight_scale_2`` and
``input_scale``, its BF16 shared expert and its router matrix) from the local
``RadixArk/GLM-5.3-NVFP4`` snapshot named by ``GLM53_NVFP4_SNAPSHOT`` (layer
``GLM53_LAYER``, default 10). Routing there uses the generic top-k gate on the
real router matrix (not GLM's sigmoid + correction-bias router), and the
hidden states are synthetic; both arms see the same ones.

Synthetic weights and routing in the first test; this is a runtime-integration
gate, not a model-quality test.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Shape, ShardingStrategy, TensorType
from max.graph.weights import WeightData
from max.nn.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.moe import MoEQuantized
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from test_common.graph_utils import is_b100_b200

HIDDEN_DIM = 6144
NUM_EXPERTS = 256
MOE_DIM = 2048
TOP_K = 8
N_DEVICES = 8
MAX_TOKENS_PER_RANK = 24
N_LAYERS = 2
N_GENERATIONS = 6

CELLS = {
    "dense24": [24] * N_DEVICES,
    "bs1_q6": [1, 1, 1, 1, 1, 1, 0, 0],
    "ragged": [1, 3, 24, 5, 7, 12, 2, 24],
}


def _nvfp4_weights() -> dict[str, torch.Tensor]:
    """Random NVFP4 experts + shared expert, generated on cuda:0 (fast)."""
    torch.manual_seed(20260928)
    device = "cuda:0"

    def _proj(
        w: dict[str, torch.Tensor],
        prefix: str,
        out_dim: int,
        in_dim: int,
        weight_scale_2: torch.Tensor | None = None,
        input_scale: float = 1.0,
    ) -> torch.Tensor:
        w[f"{prefix}.weight"] = torch.randint(
            0, 256, (out_dim, in_dim // 2), dtype=torch.uint8, device=device
        )
        w[f"{prefix}.weight_scale"] = (
            torch.rand(out_dim, in_dim // 16, device=device) * 100.0 + 50.0
        ).to(torch.float8_e4m3fn)
        if weight_scale_2 is None:
            weight_scale_2 = torch.rand((), device=device) * 1e-4
        w[f"{prefix}.weight_scale_2"] = weight_scale_2
        w[f"{prefix}.input_scale"] = torch.full(
            (), input_scale, dtype=torch.float32, device=device
        )
        return weight_scale_2

    w: dict[str, torch.Tensor] = {
        "gate.gate_score.weight": torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * 1e-3
    }
    for e in range(NUM_EXPERTS):
        s2 = _proj(w, f"experts.{e}.gate_proj", MOE_DIM, HIDDEN_DIM)
        _proj(
            w,
            f"experts.{e}.up_proj",
            MOE_DIM,
            HIDDEN_DIM,
            weight_scale_2=s2,
        )
        # Non-unit, per-expert down input scales: the fused op must invert
        # them exactly as the shipping chain does.
        _proj(
            w,
            f"experts.{e}.down_proj",
            HIDDEN_DIM,
            MOE_DIM,
            input_scale=0.75 + 0.5 * (e % 4) / 3,
        )
    # GLM-5.3's shared expert is BF16 (the modelopt export ignores it).
    for name, shape in (
        ("gate_proj", (MOE_DIM, HIDDEN_DIM)),
        ("up_proj", (MOE_DIM, HIDDEN_DIM)),
        ("down_proj", (HIDDEN_DIM, MOE_DIM)),
    ):
        w[f"shared_experts.{name}.weight"] = (
            torch.randn(*shape, dtype=torch.bfloat16, device=device) * 1e-2
        )
    return w


def _glm_layer_weights(snapshot: Path, layer: int) -> dict[str, torch.Tensor]:
    """One real GLM-5.3 NVFP4 MoE layer, renamed to the MoE module's keys."""
    import json

    from safetensors import safe_open

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.layers.{layer}.mlp."
    names = [n for n in index if n.startswith(prefix)]
    assert names, f"no MoE tensors for layer {layer} in {snapshot}"
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(index[n], []).append(n)
    w: dict[str, torch.Tensor] = {}
    for fname, tensors in by_file.items():
        with safe_open(str(snapshot / fname), framework="pt") as f:
            for n in tensors:
                key = n[len(prefix) :]
                if key == "gate.weight":
                    key = "gate.gate_score.weight"
                elif key.startswith("gate."):
                    continue  # e_score_correction_bias: unused by MoEGate
                w[key] = f.get_tensor(n)
    return w


def _wrap(
    weights: dict[str, torch.Tensor],
) -> dict[str, WeightData | torch.Tensor]:
    out: dict[str, WeightData | torch.Tensor] = {}
    for key, value in weights.items():
        value = value.cpu()
        if value.dtype == torch.float8_e4m3fn:
            out[key] = WeightData(
                Buffer.from_dlpack(value.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),
                key,
                DType.float8_e4m3fn,
                Shape(value.shape),
            )
        else:
            out[key] = value
    return out


def _quant_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.NVFP4,
        can_use_fused_swiglu=True,
    )


def _build_arm(
    fused: bool,
    wrapped: dict[str, WeightData | torch.Tensor],
    session: InferenceSession,
    devices_ref: list[DeviceRef],
) -> tuple[Model, EPCommInitializer, str]:
    quant = _quant_config()
    ep_config = EPConfig(
        dispatch_dtype=DType.uint8,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN_DIM,
        top_k=TOP_K,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=MAX_TOKENS_PER_RANK,
        n_gpus_per_node=N_DEVICES,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
        dispatch_quant_config=quant,
        fused_shared_expert=False,
        fuse_ffn_combine_send=True,
        moe_dim=MOE_DIM,
    )
    ep_comm_init = EPCommInitializer(ep_config)
    ep_batch_manager = EPBatchManager(ep_config)
    moe = MoEQuantized(
        devices=[DeviceRef.CPU()] + devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        ep_size=N_DEVICES,
        dtype=DType.uint8,
        ep_batch_manager=ep_batch_manager,
        quant_config=quant,
        shared_experts_dtype=DType.bfloat16,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(N_DEVICES)
    moe_shards = moe.shard(devices_ref)
    moe.load_state_dict(wrapped)
    with mock.patch.dict(
        os.environ, {"MODULAR_EP_FUSED_MOE": "1" if fused else "0"}
    ):
        ep_comm_init.ep_init(session)
    assert ep_config.fused_moe_ready == fused

    with Graph(
        "EPMoE_FusedMegaFFN" if fused else "EPMoE_Shipping",
        input_types=[
            *(
                TensorType(
                    DType.bfloat16,
                    (f"input_len_{i}", HIDDEN_DIM),
                    DeviceRef.GPU(i),
                )
                for i in range(N_DEVICES)
            ),
            *ep_batch_manager.input_types(),
        ],
    ) as graph:
        xs = [x.tensor for x in graph.inputs[:N_DEVICES]]
        ep_batch_manager.fetch_buffers(graph.inputs[N_DEVICES:])
        # Two chained MoE layers (same weights) on ONE module, the pattern the
        # module docstring's UNRUN DRAFT note asks to review.
        for _ in range(N_LAYERS):
            xs = forward_moe_sharded_layers(moe_shards, xs)
        graph.output(*xs)

    ir = str(graph)
    compiled = session.load(graph, weights_registry=moe.state_dict())
    return compiled, ep_comm_init, ir


def _run_ab(wrapped: dict[str, WeightData | torch.Tensor]) -> None:
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)

    shipping, ship_init, ship_ir = _build_arm(
        False, wrapped, session, devices_ref
    )
    fused, fused_init, fused_ir = _build_arm(
        True, wrapped, session, devices_ref
    )

    # Structure: arm B launches exactly one fused op per (layer, device) and
    # none of the shipping EP/FFN ops; its shared expert is not on a side
    # stream (the exclusive-launch contract). Arm A is the shipping chain.
    assert (
        fused_ir.count('symbol = "mega_ffn.ep_fused"') == N_LAYERS * N_DEVICES
    )
    for op in ("ep.dispatch_async", "ep.dispatch_wait", "ep.combine_wait"):
        assert f'symbol = "{op}' not in fused_ir, op
    assert 'symbol = "mega_ffn.ep_combine_send"' not in fused_ir
    assert "mo.sequence" not in fused_ir
    assert 'symbol = "mega_ffn.ep_fused"' not in ship_ir
    assert "mo.sequence" in ship_ir

    exact = 0
    total = 0
    for cell, lengths in CELLS.items():
        for gen in range(N_GENERATIONS):
            torch.manual_seed(1000 * gen + len(cell))
            inputs = [
                Buffer.from_dlpack(
                    torch.randn(n, HIDDEN_DIM, dtype=torch.bfloat16)
                ).to(devices[i])
                for i, n in enumerate(lengths)
            ]
            ref = shipping.execute(*inputs, *ship_init.model_inputs())
            out = fused.execute(*inputs, *fused_init.model_inputs())
            for i, (r, o) in enumerate(zip(ref, out, strict=True)):
                r_t = torch.from_dlpack(r).cpu().float()
                o_t = torch.from_dlpack(o).cpu().float()
                assert r_t.shape == o_t.shape == (lengths[i], HIDDEN_DIM)
                if lengths[i] == 0:
                    continue
                assert torch.isfinite(o_t).all(), (cell, gen, i)
                assert o_t.abs().max() > 0, (cell, gen, i)
                cos = torch.nn.functional.cosine_similarity(r_t, o_t, dim=-1)
                assert cos.min() > 0.9999, (
                    f"{cell} gen {gen} rank {i}: cosine {cos.min().item():.6f}"
                )
                exact += int(torch.equal(r_t, o_t))
                total += 1
    # Informational: the fused FFN is the native NVFP4 math and the tail
    # reuses the shipping reduce, so bit-identity is expected but not gated.
    print(f"fused == shipping bit-exactly on {exact}/{total} (cell, gen, rank)")


_needs_ep8 = [
    pytest.mark.skipif(
        accelerator_api() == "hip", reason="NVFP4 fused EP MoE is NVIDIA-only"
    ),
    pytest.mark.skipif(not is_b100_b200(), reason="requires B100/B200 (SM100)"),
    pytest.mark.skipif(
        accelerator_count() < N_DEVICES, reason="requires 8 GPUs (EP8)"
    ),
]


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
def test_ep_fused_megaffn_matches_shipping() -> None:
    _run_ab(_wrap(_nvfp4_weights()))


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.skipif(
    "GLM53_NVFP4_SNAPSHOT" not in os.environ,
    reason="set GLM53_NVFP4_SNAPSHOT to a local RadixArk/GLM-5.3-NVFP4 snapshot",
)
def test_ep_fused_megaffn_real_glm_layer() -> None:
    snapshot = Path(os.environ["GLM53_NVFP4_SNAPSHOT"])
    layer = int(os.environ.get("GLM53_LAYER", "10"))
    _run_ab(_wrap(_glm_layer_weights(snapshot, layer)))
