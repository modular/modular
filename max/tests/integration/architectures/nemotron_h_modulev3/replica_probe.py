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
"""Test-only Nemotron-H pieces for the TP4 bitwise replica check.

:class:`NemotronHReplicaTrace` runs MoE decoder layers and returns every
device's tensors at each allreduce boundary. It scales each device's
partial sum by a per-step power of two just before the reduction, so a step
can plant an operand spread on any subset of the allreduces without
recompiling. :func:`replica_weights` gives the trace its weights, and
:func:`stage_replica_trace` stages it with them.

The trace is not the model, but it stages each block the way the backbone
does and runs the real mixer and the real ``NemotronHRouter.forward``: the
router's taps are the only addition, read through ``gate._route`` and
returned alongside the block's output. The boundary matters, because
``NemotronHBackbone`` wraps a whole ``NemotronHBlock`` in ``as_subgraph``, so
a mixer output's reduction and the residual add sit inside the subgraph and
the next block's norm sits behind another call. PatternFusion therefore never
sees the ``allreduce -> add -> rms_norm`` chain, and production takes the
plain ``mo.distributed.allreduce.sum`` for every reduction on this path.
Staging only the mixer, as this trace first did, keeps the chain in one
region and fuses it, which would check a kernel production never runs:
:func:`assert_allreduce_form` pins the form the trace's dump takes.

The schedule holds MoE layers only, and every layer's mixer output reduces
across the ranks, so the residual stream and the router input inherit one
allreduce per layer. A Mamba or attention layer would add its state or
paged-KV inputs to the trace without changing the reduction the check reads,
and a dense MLP layer is replicated, so it reduces nothing. The row counts
cover the three forms the reduction's kernel dispatches to, so the check
covers what the router input depends on and nothing about attention or Mamba
arithmetic.
"""

from __future__ import annotations

import re
from unittest.mock import Mock

import ml_dtypes
import numpy as np
from max import functional as F
from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.pipelines.architectures.nemotron_h_modulev3.layers.moe import (
    NemotronHMoE,
)
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    NemotronHConfig,
)
from max.pipelines.architectures.nemotron_h_modulev3.nemotron_h import (
    NemotronH,
    NemotronHBlock,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor
from max.tensor.compilation import StagedGraph, as_subgraph
from transformers import NemotronHConfig as HFNemotronHConfig

ALLREDUCES_PER_LAYER = 1
"""The mixer output's reduction; the router input inherits every earlier one."""

LAYERS = ["moe", "moe", "moe", "moe"]
"""The traced schedule: each MoE layer reduces its mixer output, which is the
residual the next layer's router reads."""

assert all(kind == "moe" for kind in LAYERS), LAYERS

HIDDEN = 4096
MOE_INTERMEDIATE = 2048
INTERMEDIATE = 2048
NUM_EXPERTS = 4
EXPERTS_PER_TOK = 2

COMPOSITE_ALLREDUCE = (
    "mo.composite.distributed.allreduce_add_rms_norm_quant_fp8"
)
"""The fused allreduce, residual add, RMSNorm and FP8 quantize."""
# Every op whose name mentions allreduce, so a reduction that took a form the
# check does not expect still counts. Op lines only: a full-width dump also
# names the composite in its kernel declaration's registration string.
_ALLREDUCE_FAMILY_OP = re.compile(r"= (mo\.[\w.]*allreduce[\w.]*)\([^\n]*")
# The form the block subgraph boundary leaves production with: the plain op,
# whose kernels add peers in global rank order. A bundled collective adds its
# peers from inside a `mo.parallel` region instead.
_PLAIN_ALLREDUCE = "mo.distributed.allreduce.sum"
_BUNDLED_OP = re.compile(r"= mo\.(?:composite\.)?bundled\.")

_ROUTER_STD = 0.02
_EXPERT_STD = 0.02


def allreduce_ops(post_fusion_ir: str) -> list[str]:
    """Returns the names of a post-PatternFusion dump's allreduce-family ops.

    Plain and composite ops alike, in program order, so ``layer`` is that
    layer's mixer output reduction.
    """
    return [m.group(1) for m in _ALLREDUCE_FAMILY_OP.finditer(post_fusion_ir)]


def fused_allreduces(post_fusion_ir: str) -> list[int]:
    """Returns the :func:`allreduce_ops` indices that fused."""
    return [
        index
        for index, op in enumerate(allreduce_ops(post_fusion_ir))
        if op == COMPOSITE_ALLREDUCE
    ]


def assert_allreduce_form(post_fusion_ir: str) -> None:
    """Asserts every reduction in the dump is the form production's take.

    Local routing needs the kernels' global rank order, and the plain op is
    the only form the residual stream reduces with in production, because the
    backbone's block subgraph keeps PatternFusion's allreduce-add-norm chain
    from forming across its boundary. A dump that fuses, or that bundles a
    collective, fails here and names the op, rather than in a numerics check
    that would only report a mismatch.
    """
    unexpected = [
        op for op in allreduce_ops(post_fusion_ir) if op != _PLAIN_ALLREDUCE
    ]
    assert not unexpected, (
        "an allreduce took a form production does not:\n"
        + "\n".join(unexpected)
    )
    bundled = _BUNDLED_OP.findall(post_fusion_ir)
    assert not bundled, "the graph bundled a collective: " + ", ".join(bundled)


def replica_hf_config() -> HFNemotronHConfig:
    """Returns a Nemotron-H config for :data:`LAYERS` at the trace's width.

    The width is wide enough that a row is 8 KiB, which separates the three
    forms of the 4x B200 allreduce dispatch over the row counts the test runs.
    Nemotron-3.5-Lightning itself is narrower (2688), so the row counts land
    in those forms at other row numbers than in production; the form of every
    reduction is pinned by the post-fusion IR check instead.
    The fields of the mixers the schedule does not build still have to divide
    across the devices.
    """
    return HFNemotronHConfig(
        layers_block_type=LAYERS,
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        mamba_num_heads=8,
        mamba_head_dim=64,
        n_groups=4,
        ssm_state_size=16,
        conv_kernel=4,
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=EXPERTS_PER_TOK,
        moe_intermediate_size=MOE_INTERMEDIATE,
        moe_shared_expert_intermediate_size=MOE_INTERMEDIATE,
        routed_scaling_factor=2.5,
    )


def replica_config(devices: list[DeviceRef]) -> NemotronHConfig:
    """Returns the config of the trace for a mesh on ``devices``.

    SM100 GPUs put the router's gate matmul on the tensor cores, as the
    production config does, so the traced router is the one a B200 serves.
    """
    pipeline = Mock()
    pipeline.model.data_parallel_degree = 1
    pipeline.speculative = None
    hf = replica_hf_config()
    kv_params = NemotronHConfig.construct_kv_params(
        huggingface_config=hf,
        pipeline_config=pipeline,
        devices=devices,
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
    )
    config = NemotronHConfig.from_huggingface(
        hf,
        kv_params=kv_params,
        devices=devices,
        max_seq_len=1024,
    )
    config.bf16_router_gate = True
    return config


def _bf16(values: np.ndarray) -> Buffer:
    return Buffer.from_numpy(
        values.astype(ml_dtypes.bfloat16).view(np.uint16)
    ).view(DType.bfloat16, list(values.shape))


def _filled(value: float, shape: tuple[int, ...], dtype: DType) -> Buffer:
    values = np.full(shape, value, np.float32)
    if dtype == DType.bfloat16:
        return _bf16(values)
    return Buffer.from_numpy(values.astype(dtype.to_numpy()))


def replica_weights(
    trace: NemotronHReplicaTrace, seed: int = 17
) -> dict[str, DLPackArray]:
    """Returns a host buffer for every trace parameter.

    The router gate, the routed experts and the shared expert get random
    values, so each layer's output has a spread and the router's top-k
    compares scores as close to each other as a checkpoint's do. The norms
    get ones and the selection bias zeros, the values the model starts from.
    """
    rng = np.random.default_rng(seed)
    weights: dict[str, DLPackArray] = {}
    for name, parameter in trace.parameters:
        shape = tuple(int(dim) for dim in parameter.shape)
        if name.endswith("norm.weight"):
            weights[name] = _filled(1.0, shape, parameter.dtype)
        elif name.endswith("gate.weight"):
            weights[name] = _bf16(rng.normal(0.0, _ROUTER_STD, shape))
        elif name.endswith("e_score_correction_bias"):
            weights[name] = _filled(0.0, shape, parameter.dtype)
        elif name.endswith(("up_weight", "down_weight", "proj.weight")):
            weights[name] = _bf16(rng.normal(0.0, _EXPERT_STD, shape))
        else:
            raise AssertionError(f"no replica weight rule for {name!r}")
    return weights


def stage_replica_trace(
    trace: NemotronHReplicaTrace, devices: list[DeviceRef]
) -> tuple[StagedGraph[..., list[Tensor]], dict[str, Tensor]]:
    """Stages the trace with :func:`replica_weights`.

    The inputs are ``h0`` and the plant on the first device, and nothing
    else: the schedule has no attention or Mamba layer to take a cache.
    """
    norm = trace.layers[0].norm.weight
    return trace.stage(
        TensorType(norm.dtype, ["rows", norm.shape[0]], device=devices[0]),
        TensorType(
            norm.dtype,
            [ALLREDUCES_PER_LAYER * len(trace.layers), len(devices)],
            device=devices[0],
        ),
        weights=replica_weights(trace),
    )


def _layer_prefix(layer: NemotronHBlock) -> str:
    """Returns the weight-name prefix ``layer``'s weights are stored under.

    The block is staged inside a wrapper, so its weights sit one attribute
    deeper than the names they load from; the prefix has to say where the
    block itself starts.
    """
    name = layer.norm.weight.external_name
    assert name is not None, "the trace's weights must be named before staging"
    return name[: -len("norm.weight")]


class _ReplicaBlock(Module[..., list[Tensor]]):
    """One block, staged as a subgraph the way the backbone stages it.

    ``NemotronHBlock.forward``, with the plant applied to the mixer's shards
    just before the reduction and the router's taps returned beside the
    output, so the region PatternFusion sees is the one production gives it.
    """

    def __init__(self, layer: NemotronHBlock, row: int) -> None:
        self.layer = layer
        self.row = row

    def forward(self, h: Tensor, plant: Tensor) -> list[Tensor]:
        """Returns the residual, then each device's router taps.

        Args:
            h: The block's input rows.
            plant: ``[allreduces, devices]`` power-of-two factors. Row
                ``row`` scales each device's mixer output just before its
                reduction.
        """
        mixer = self.layer.mixer
        assert isinstance(mixer, NemotronHMoE)
        router_input = self.layer.norm(h)
        gate = mixer.gate
        devices = router_input.mesh.num_devices
        ids: list[Tensor] = []
        weights: list[Tensor] = []
        for rank in range(devices):
            expert_ids, routing_weights = gate._route(
                router_input.local_shards[rank],
                gate.weight.local_shards[rank],
                gate.e_score_correction_bias.local_shards[rank],
            )
            ids.append(expert_ids)
            weights.append(routing_weights)
        mixed = mixer(router_input)
        # A power-of-two factor is exact in bf16, and a factor of one leaves
        # the reduction's operands as the model computes them.
        planted = Tensor.from_shard_values(
            [
                shard
                * plant.local_shards[device][
                    self.row : self.row + 1, device : device + 1
                ]
                for device, shard in enumerate(mixed.local_shards)
            ],
            mixed.mapping,
        )
        h = h + planted.to(h.mapping)
        return [
            h,
            *router_input.local_shards,
            *ids,
            *weights,
            *h.local_shards,
        ]


class NemotronHReplicaTrace(Module[..., list[Tensor]]):
    """Runs Nemotron-H decoder layers and returns per-rank boundary tensors.

    For each layer it returns, in order: every device's router input
    (``norm(h)``, the tensor the MoE layer's router reads), every device's
    expert ids, every device's routing weights, and every device's residual
    after the mixer output's reduction.
    """

    def __init__(self, model: NemotronH) -> None:
        self.layers = model.backbone.layers

    def forward(self, h0: Tensor, plant: Tensor) -> list[Tensor]:
        """Returns each layer's boundary tensors on every device.

        Args:
            h0: The first layer's input rows, on the first device.
            plant: ``[allreduces, devices]`` power-of-two factors, on the
                first device. Row ``layer`` scales each device's mixer
                output just before its reduction.
        """
        mesh = self.layers[0].norm.weight.mesh
        h = h0.to(mesh)
        if mesh.num_devices > 1:
            plant = F.distributed_broadcast(plant, mesh)
        traces: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            # The backbone shares one subgraph between the blocks of a kind.
            # Each layer here has its own name, so that a layer's taps need
            # no layer index, and the weights it holds give it its prefix.
            block = as_subgraph(
                _ReplicaBlock(layer, index),
                name=f"nemotron_h_replica_block_{index}",
                prefix=_layer_prefix(layer),
            )
            out = block(h, plant)
            h = out[0]
            traces.extend(out[1:])
        return traces
