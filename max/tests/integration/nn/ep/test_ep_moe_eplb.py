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
"""Checks that an EPLB placement never changes what an EP MoE computes.

Two MoE layers share one compiled subgraph, as in a real model. A placement
only decides which GPU runs each expert: the combine sums a token's top-k
outputs in top-k order in fp32, and no expert kernel depends on its
neighbors, so outputs must be bit-identical to the same checkpoint served
without EPLB. A control loads layer 0's placement into both layers, which is
what subgraph reuse did before placements moved into the weights, and must
not match.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    Weight,
    ops,
)
from max.graph.value import TensorValue, Value
from max.nn.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.kernels import moe_router_group_limited
from max.nn.layer import LayerList, Module
from max.nn.moe import MoE, MoEGate, apply_eplb_placement
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.transformer.transformer import forward_sequential_layers
from max.pipelines.lib.eplb_stats import EplbPlacement, EplbTopology

HIDDEN_DIM = 7168
# The single-group router runs one thread per expert in whole warps, so the
# expert count must be a multiple of AMD's 64-wide wavefront.
# MOE_DIM is halved to keep the doubled expert count in the same memory.
MOE_DIM = 1024
NUM_EXPERTS = 64
TOP_K = 8
NUM_LAYERS = 2
TOKENS_PER_DEVICE = 64
WEIGHTS_STDDEV = 0.01

N_DEVICES = min(accelerator_count(), 8)

pytestmark = pytest.mark.skipif(
    N_DEVICES < 2, reason="EP needs at least two GPUs"
)


class _SigmoidRouter(MoEGate):
    """Single-group router shaped like MiniMax-M3's.

    fp32 gate weights and correction bias, sigmoid scores, top-k over one
    group, normalized and scaled. It defines ``compute_scores``, so EP
    dispatch runs it as the fused single-group router with the EPLB remap.
    """

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        is_sharding: bool = False,
    ) -> None:
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=DType.float32,
            is_sharding=is_sharding,
        )
        self.norm_topk_prob = True
        self.routed_scaling_factor = 2.0
        self.e_score_correction_bias = Weight(
            "e_score_correction_bias",
            DType.float32,
            [num_experts],
            device=devices[0],
        )

    def compute_scores(self, hidden_states: TensorValue) -> TensorValue:
        return ops.sigmoid(self.gate_score(hidden_states.cast(DType.float32)))

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        return moe_router_group_limited(
            self.compute_scores(hidden_states),
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            n_groups=1,
            topk_group=1,
            norm_weights=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        super()._set_sharding_strategy(strategy)
        self.e_score_correction_bias.sharding_strategy = (
            ShardingStrategy.replicate(strategy.num_devices)
        )

    def shard(self, devices: Iterable[DeviceRef]) -> list[_SigmoidRouter]:
        devices = list(devices)
        gate_shards = self.gate_score.shard(devices)
        bias_shards = self.e_score_correction_bias.shard(devices)
        shards = []
        for i, device in enumerate(devices):
            sharded = _SigmoidRouter(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                is_sharding=True,
            )
            sharded.gate_score = gate_shards[i]
            sharded.e_score_correction_bias = bias_shards[i]
            shards.append(sharded)
        return shards


class _Block(Module):
    """One residual EP MoE layer, called through a shared subgraph."""

    def __init__(
        self,
        layer_idx: int,
        num_physical: int,
        devices: list[DeviceRef],
        ep_batch_manager: EPBatchManager,
        sigmoid_router: bool,
    ) -> None:
        super().__init__()
        self.mlp = MoE(
            devices=devices,
            hidden_dim=HIDDEN_DIM,
            num_experts=num_physical,
            num_experts_per_token=TOP_K,
            moe_dim=MOE_DIM,
            num_logical_experts=NUM_EXPERTS,
            ep_size=len(devices),
            dtype=DType.bfloat16,
            gate_cls=_SigmoidRouter if sigmoid_router else MoEGate,
            ep_batch_manager=ep_batch_manager,
        )
        self.mlp.layer_idx = layer_idx
        self.mlp.sharding_strategy = ShardingStrategy.expert_parallel(
            len(devices)
        )
        self.mlp_shards = self.mlp.shard(devices)
        self._ep_batch_manager = ep_batch_manager

    def __call__(
        self,
        xs: list[TensorValue],
        ep_inputs: list[Value[Any]],
        layer_idx_per_device: list[TensorValue],
    ) -> list[TensorValue]:
        self._ep_batch_manager.fetch_buffers(ep_inputs)
        outs = forward_moe_sharded_layers(
            self.mlp_shards, xs, None, layer_idx_per_device
        )
        return [x + out for x, out in zip(xs, outs, strict=True)]


class _Model(Module):
    def __init__(
        self,
        num_physical: int,
        devices: list[DeviceRef],
        ep_batch_manager: EPBatchManager,
        sigmoid_router: bool,
    ) -> None:
        super().__init__()
        self.layers = LayerList(
            [
                _Block(
                    i, num_physical, devices, ep_batch_manager, sigmoid_router
                )
                for i in range(NUM_LAYERS)
            ]
        )

    def __call__(self, x):  # noqa: ANN001
        return x


def _logical_checkpoint(sigmoid_router: bool) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    device = "cuda:0"

    def rand(*shape: int) -> torch.Tensor:
        return (
            torch.randn(*shape, dtype=torch.bfloat16, device=device)
            * WEIGHTS_STDDEV
        ).cpu()

    state_dict: dict[str, torch.Tensor] = {}
    for layer in range(NUM_LAYERS):
        prefix = f"layers.{layer}.mlp."
        state_dict[f"{prefix}gate.gate_score.weight"] = rand(
            NUM_EXPERTS, HIDDEN_DIM
        )
        if sigmoid_router:
            state_dict[f"{prefix}gate.gate_score.weight"] = state_dict[
                f"{prefix}gate.gate_score.weight"
            ].float()
            state_dict[f"{prefix}gate.e_score_correction_bias"] = (
                torch.randn(NUM_EXPERTS, dtype=torch.float32) * 0.1
            )
        for e in range(NUM_EXPERTS):
            state_dict[f"{prefix}experts.{e}.gate_proj.weight"] = rand(
                MOE_DIM, HIDDEN_DIM
            )
            state_dict[f"{prefix}experts.{e}.up_proj.weight"] = rand(
                MOE_DIM, HIDDEN_DIM
            )
            state_dict[f"{prefix}experts.{e}.down_proj.weight"] = rand(
                HIDDEN_DIM, MOE_DIM
            )
    return state_dict


def _placements(
    redundant_experts_per_gpu: int,
) -> tuple[EplbPlacement, EplbPlacement]:
    """Returns a per-layer placement and a layer-invariant one."""
    topology = EplbTopology(
        ep_size=N_DEVICES,
        n_nodes=1,
        n_groups=1,
        redundant_experts_per_gpu=redundant_experts_per_gpu,
    )
    # Each layer has a different hot set, so the per-layer planner must move
    # different experts in each layer.
    rng = np.random.default_rng(0)
    load = rng.integers(1, 100, size=(NUM_LAYERS, NUM_EXPERTS))
    load[0, :4] *= 1000
    load[1, -4:] *= 1000
    per_layer = EplbPlacement.from_rebalance(load, topology)
    shared = EplbPlacement.from_rebalance(
        load.sum(axis=0, keepdims=True), topology
    )
    shared = EplbPlacement(
        phy2log=np.repeat(shared.phy2log, NUM_LAYERS, axis=0),
        log2phy=np.repeat(shared.log2phy, NUM_LAYERS, axis=0),
        logcnt=np.repeat(shared.logcnt, NUM_LAYERS, axis=0),
    )
    assert not np.array_equal(per_layer.phy2log[0], per_layer.phy2log[1])
    return per_layer, shared


def _pad_log2phy(log2phy: np.ndarray, max_replicas: int) -> np.ndarray:
    pad = max_replicas - log2phy.shape[-1]
    return np.pad(
        log2phy, ((0, 0), (0, 0), (0, pad)), constant_values=-1
    ).astype(np.int32)


class _Harness:
    """One EP deployment that compiles a model per placement."""

    def __init__(
        self, redundant_experts_per_gpu: int, sigmoid_router: bool = False
    ) -> None:
        self.sigmoid_router = sigmoid_router
        self.devices = [Accelerator(i) for i in range(N_DEVICES)]
        self.devices_ref = [DeviceRef(d.label, d.id) for d in self.devices]
        self.session = InferenceSession(devices=self.devices)
        self.per_layer, self.shared = _placements(redundant_experts_per_gpu)
        self.max_replicas = max(
            self.per_layer.max_replicas, self.shared.max_replicas
        )
        self.base_config = EPConfig(
            dispatch_dtype=DType.bfloat16,
            combine_dtype=DType.bfloat16,
            hidden_size=HIDDEN_DIM,
            top_k=TOP_K,
            n_experts=self.per_layer.num_phy,
            max_tokens_per_rank=TOKENS_PER_DEVICE,
            n_gpus_per_node=N_DEVICES,
            n_nodes=1,
            eplb_enabled=True,
            num_moe_layers=NUM_LAYERS,
            num_logical_experts=NUM_EXPERTS,
            max_replicas=self.max_replicas,
        )
        self.ep_comm_init = EPCommInitializer(self.base_config)
        self.ep_comm_init.ep_init(self.session)
        self._set_tables(self.per_layer)
        self.checkpoint = _logical_checkpoint(sigmoid_router)

    def _set_tables(self, tables: EplbPlacement) -> None:
        log2phy = _pad_log2phy(tables.log2phy, self.max_replicas)
        self.ep_comm_init.set_eplb_placement(
            [Buffer.from_numpy(log2phy).to(d) for d in self.devices],
            [Buffer.from_numpy(tables.logcnt).to(d) for d in self.devices],
        )

    def compile(
        self,
        tables: EplbPlacement | None,
        weights_phy2log: np.ndarray | None = None,
        fold_router: bool = False,
    ) -> Model:
        """Compiles with ``tables`` routing and weights laid out by
        ``weights_phy2log``; ``tables=None`` compiles without EPLB."""
        if tables is None:
            assert self.base_config.n_experts == NUM_EXPERTS
            config = dataclasses.replace(self.base_config, eplb_enabled=False)
            num_physical = NUM_EXPERTS
        else:
            config = dataclasses.replace(
                self.base_config,
                eplb_phy2log_plan=tables.phy2log,
                eplb_router_folded=fold_router,
            )
            num_physical = tables.num_phy
        ep_batch_manager = EPBatchManager(config)
        model = _Model(
            num_physical,
            self.devices_ref,
            ep_batch_manager,
            self.sigmoid_router,
        )
        if tables is None:
            model.load_state_dict(self.checkpoint, strict=True)
        else:
            assert weights_phy2log is not None
            model.load_state_dict(
                apply_eplb_placement(
                    model,
                    self.checkpoint,
                    weights_phy2log,
                    fold_router=fold_router,
                ),
                strict=True,
            )

        input_types = [
            TensorType(DType.bfloat16, (f"tokens_{i}", HIDDEN_DIM), ref)
            for i, ref in enumerate(self.devices_ref)
        ]
        with Graph(
            "EplbMoE",
            input_types=[*input_types, *ep_batch_manager.input_types()],
        ) as graph:
            xs = [v.tensor for v in graph.inputs[:N_DEVICES]]
            ep_inputs = list(graph.inputs[N_DEVICES:])

            def inputs_for_layer(
                idx: int, h: list[TensorValue]
            ) -> Sequence[Any]:
                return [
                    h,
                    ep_inputs,
                    [
                        ops.constant(idx, DType.int32, device=ref)
                        for ref in self.devices_ref
                    ],
                ]

            out = forward_sequential_layers(
                list(model.layers),
                inputs_for_layer=inputs_for_layer,
                initial_hidden_states=xs,
                subgraph_layer_groups=[list(range(NUM_LAYERS))],
                weight_prefix_for_layer=lambda i: f"layers.{i}.",
            )
            graph.output(*out)
        return self.session.load(graph, weights_registry=model.state_dict())

    def run(
        self,
        model: Model,
        tables: EplbPlacement | None,
        xs: list[torch.Tensor],
    ) -> torch.Tensor:
        ep_inputs = self.ep_comm_init.model_inputs()
        if tables is None:
            # A graph without EPLB takes no log2phy / logcnt tables.
            ep_inputs = ep_inputs[: -2 * N_DEVICES]
        else:
            self._set_tables(tables)
            ep_inputs = self.ep_comm_init.model_inputs()
        inputs = [
            Buffer.from_dlpack(x).to(d)
            for x, d in zip(xs, self.devices, strict=True)
        ]
        outs = model.execute(*inputs, *ep_inputs)
        return torch.cat([torch.from_dlpack(o).to("cpu") for o in outs])


def _inputs(seed: int) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    return [
        torch.randn(TOKENS_PER_DEVICE, HIDDEN_DIM, dtype=torch.bfloat16)
        for _ in range(N_DEVICES)
    ]


def test_per_layer_placement_is_bit_identical_to_no_eplb() -> None:
    harness = _Harness(redundant_experts_per_gpu=0)
    xs = _inputs(1)
    per_layer = harness.per_layer

    want = harness.run(harness.compile(None), None, xs)
    got = harness.run(
        harness.compile(per_layer, per_layer.phy2log), per_layer, xs
    )
    assert torch.equal(got, want), (
        f"{int((got != want).any(dim=1).sum())} of {got.shape[0]} tokens "
        "differ from the model served without EPLB"
    )

    # Layer 0's weight layout under per-layer routing: the pre-fix behavior.
    stale = np.repeat(per_layer.phy2log[:1], NUM_LAYERS, axis=0)
    wrong = harness.run(harness.compile(per_layer, stale), per_layer, xs)
    assert (wrong.float() - want.float()).abs().max() > 0.1, (
        "the stale-placement control matched; the test cannot detect a "
        "wrong expert layout"
    )


def test_replicated_placement_is_bit_identical() -> None:
    """Which replica serves a token must not change its output."""
    harness = _Harness(redundant_experts_per_gpu=1)
    xs = _inputs(1)
    per_layer = harness.per_layer
    shared = harness.shared
    assert int(per_layer.logcnt.max()) > 1

    got = harness.run(
        harness.compile(per_layer, per_layer.phy2log), per_layer, xs
    )
    want = harness.run(harness.compile(shared, shared.phy2log), shared, xs)
    assert torch.equal(got, want), (
        f"{int((got != want).any(dim=1).sum())} of {got.shape[0]} tokens "
        "differ between two replicated placements"
    )


def test_folded_permutation_matches_no_eplb() -> None:
    """A permutation folded into the router is bit-identical up to ties."""
    harness = _Harness(redundant_experts_per_gpu=0)
    xs = _inputs(2)
    per_layer = harness.per_layer

    want = harness.run(harness.compile(None), None, xs)
    got = harness.run(
        harness.compile(per_layer, per_layer.phy2log, fold_router=True),
        per_layer,
        xs,
    )

    # Folding reorders the router's experts, and top-k breaks exact score
    # ties by expert index, so a token whose k-th and (k+1)-th bf16 scores
    # tie may pick the other, equally scored expert. Every other token must
    # be bit-identical. A bf16 ulp is ~0.4% of a score and the 8th and 9th
    # of 64 scores sit a few percent apart, so a few percent of tokens tie
    # in each of the two layers (3% on B200, 6% on MI355 with two GPUs). A
    # wrong fold moves nearly every token, so 10% still catches one.
    differs = (got != want).any(dim=1)
    assert int(differs.sum()) <= max(2, got.shape[0] // 10), (
        f"{int(differs.sum())} of {got.shape[0]} tokens differ; more than "
        "top-k ties explain"
    )


def test_fused_single_group_router_is_bit_identical_to_no_eplb(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fused router+remap kernel changes no bit of the output."""
    harness = _Harness(redundant_experts_per_gpu=0, sigmoid_router=True)
    xs = _inputs(3)
    per_layer = harness.per_layer

    want = harness.run(harness.compile(None), None, xs)
    with caplog.at_level("INFO"):
        fused = harness.compile(per_layer, per_layer.phy2log)
    assert "mo.moe.single.group.router.eplb" in caplog.text
    got = harness.run(fused, per_layer, xs)
    assert torch.equal(got, want), (
        f"{int((got != want).any(dim=1).sum())} of {got.shape[0]} tokens "
        "differ from the model served without EPLB"
    )


def test_fused_single_group_router_with_replicas_is_bit_identical() -> None:
    harness = _Harness(redundant_experts_per_gpu=1, sigmoid_router=True)
    xs = _inputs(3)
    per_layer = harness.per_layer
    shared = harness.shared
    assert int(per_layer.logcnt.max()) > 1

    got = harness.run(
        harness.compile(per_layer, per_layer.phy2log), per_layer, xs
    )
    want = harness.run(harness.compile(shared, shared.phy2log), shared, xs)
    assert torch.equal(got, want), (
        f"{int((got != want).any(dim=1).sum())} of {got.shape[0]} tokens "
        "differ between two replicated placements"
    )


def test_fold_with_fp32_router_is_bit_identical_to_no_eplb() -> None:
    """With fp32 router scores, folding leaves no ties to break differently."""
    harness = _Harness(redundant_experts_per_gpu=0, sigmoid_router=True)
    xs = _inputs(4)
    per_layer = harness.per_layer

    want = harness.run(harness.compile(None), None, xs)
    got = harness.run(
        harness.compile(per_layer, per_layer.phy2log, fold_router=True),
        per_layer,
        xs,
    )
    assert torch.equal(got, want), (
        f"{int((got != want).any(dim=1).sum())} of {got.shape[0]} tokens "
        "differ from the model served without EPLB"
    )
