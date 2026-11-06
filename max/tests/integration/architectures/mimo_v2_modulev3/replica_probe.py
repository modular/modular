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
"""Test-only MiMo pieces for the TP4 bitwise replica check.

:class:`MiMoReplicaTrace` runs the decoder layers and returns every device's
tensors at each allreduce boundary. It scales each device's partial sum by a
per-step power of two just before the reduction, so a step can plant an
operand spread on any subset of the allreduces without recompiling.
:func:`replica_weights` gives the trace nonzero weights on every allreduce
input, and :func:`stage_replica_trace` stages it with them.

The trace is not the model. It reproduces the layer forward's body with the
MoE layer staged as a subgraph, so the real ``MiMoV2Router.forward`` does
run; only the taps read each replica's router input through ``gate._route``
rather than from the forward's output. The trace also builds a lone TP mesh,
so it covers TP_TP and nothing about TP_EP, DP_EP or a dp x tp mesh. What it
does cover is every allreduce the residual stream and the router input
depend on.
"""

from __future__ import annotations

import re

import ml_dtypes
import numpy as np
from max import functional as F
from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import (
    MiMoV2,
    MiMoV2DecoderLayer,
    MiMoV2MoE,
)
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    FULL,
    SLIDING,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.tensor import Tensor
from max.tensor.compilation import StagedGraph, as_subgraph
from transformers.configuration_utils import PretrainedConfig

ALLREDUCES_PER_LAYER = 2
"""The attention output's reduction, then the MLP or MoE output's."""

COMPOSITE_ALLREDUCE = (
    "mo.composite.distributed.allreduce_add_rms_norm_quant_fp8"
)
"""The fused allreduce, residual add, RMSNorm and FP8 quantize."""
# Every op whose name mentions allreduce, so a reduction that took a form the
# tests do not expect still counts. Op lines only: a full-width dump also names
# the composite in its kernel declaration's registration string.
_ALLREDUCE_FAMILY_OP = re.compile(r"= (mo\.[\w.]*allreduce[\w.]*)\([^\n]*")
# The forms whose cross-rank add order is the kernels' global rank order: the
# plain op, and the composite, which carries that same reduction. A bundled
# collective adds its peers from inside a `mo.parallel` region instead.
_ALLOWED_ALLREDUCES = frozenset(
    ("mo.distributed.allreduce.sum", COMPOSITE_ALLREDUCE)
)
_BUNDLED_OP = re.compile(r"= mo\.(?:composite\.)?bundled\.")

_FP8_BLOCK_SCALE = 1.0 / 64.0
_O_PROJ_STD = 0.05
_ROUTER_STD = 0.01
_EXPERT_VARIANTS = 8
# E8M0 exponents of the routed experts' MXFP4 projections.
_EXPERT_SCALE_EXPONENT = {"gate_proj": -8, "up_proj": -8, "down_proj": -4}


def allreduce_ops(post_fusion_ir: str) -> list[str]:
    """Returns the names of a post-PatternFusion dump's allreduce-family ops.

    Plain and composite ops alike, in program order, so ``2 * layer`` is a
    layer's attention reduction and ``2 * layer + 1`` its MLP or MoE
    output's.
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
    """Asserts every reduction in the dump takes an allowed form.

    Local routing needs the kernels' global rank order, so a reduction that
    took any other form fails here, where the failure names the op, rather
    than in a numerics check that would only report a mismatch.
    """
    ops = allreduce_ops(post_fusion_ir)
    unexpected = [op for op in ops if op not in _ALLOWED_ALLREDUCES]
    assert not unexpected, (
        "an allreduce took a form whose cross-rank order is unproven:\n"
        + "\n".join(unexpected)
    )
    bundled = _BUNDLED_OP.findall(post_fusion_ir)
    assert not bundled, "the graph bundled a collective: " + ", ".join(bundled)


def first_four_layer_config() -> PretrainedConfig:
    """Returns the published MiMo-V2.6 dimensions for layers 0 through 3."""
    return PretrainedConfig(
        architectures=["MiMoV2ForCausalLM"],
        model_type="mimo_v2",
        vocab_size=1,
        hidden_size=4096,
        num_hidden_layers=4,
        hybrid_layer_pattern=[0, 1, 1, 1],
        moe_layer_freq=[0, 1, 1, 1],
        num_attention_heads=64,
        swa_num_attention_heads=64,
        num_key_value_heads=4,
        swa_num_key_value_heads=8,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        partial_rotary_factor=0.334,
        rope_theta=10000000.0,
        swa_rope_theta=10000.0,
        rope_parameters={
            "partial_rotary_factor": 0.334,
            "rope_theta": 10000000.0,
            "rope_type": "default",
            "type": "default",
        },
        max_position_embeddings=1048576,
        sliding_window=128,
        attention_value_scale=0.707,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        layernorm_epsilon=1e-6,
        intermediate_size=16384,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=None,
        norm_topk_prob=True,
        routed_scaling_factor=None,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=False,
        attention_projection_layout="fused_qkv",
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "store_dtype": "mxfp4",
            "mxfp4_block_size": 32,
        },
    )


def _as(raw: np.ndarray, dtype: DType) -> Buffer:
    return Buffer.from_numpy(raw).view(dtype, list(raw.shape))


def _fp8_codes(rng: np.random.Generator, shape: tuple[int, ...]) -> Buffer:
    values = rng.standard_normal(shape, dtype=np.float32)
    return _as(
        values.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),
        DType.float8_e4m3fn,
    )


def _bf16(values: np.ndarray) -> Buffer:
    return _as(
        values.astype(ml_dtypes.bfloat16).view(np.uint16), DType.bfloat16
    )


def _filled(value: float, shape: tuple[int, ...], dtype: DType) -> Buffer:
    if dtype == DType.bfloat16:
        return _bf16(np.full(shape, value, np.float32))
    return Buffer.from_numpy(np.full(shape, value, dtype.to_numpy()))


def replica_weights(
    trace: MiMoReplicaTrace, seed: int = 17
) -> dict[str, DLPackArray]:
    """Returns a host buffer for every trace parameter.

    The projections that feed each reduction get random values sized to give
    partial sums of order one: the attention's QKV and output projections,
    the dense MLP's FP8 projections, the router and every routed expert. The
    experts cycle through a few random MXFP4 variants and share one constant
    scale per projection, which keeps the host arrays small while each
    device still holds different channels. Norms, sinks and the routing bias
    get the model's initial values.

    No entry is one of the module's own parameters: those are unrealized,
    and loading one realizes every tensor the model built, host FP8 and E8M0
    stacks included, in one eager graph that the GPU targets must compile.
    """
    rng = np.random.default_rng(seed)
    weights: dict[str, DLPackArray] = {}
    variants: dict[tuple[str, tuple[int, ...]], list[Buffer]] = {}
    for name, parameter in trace.parameters:
        shape = tuple(int(dim) for dim in parameter.shape)
        parts = name.split(".")
        if ".experts." in name:
            projection, field = parts[-2], parts[-1]
            if field == "weight":
                key = (projection, shape)
                if key not in variants:
                    variants[key] = [
                        _as(
                            rng.integers(0, 256, shape, dtype=np.uint8),
                            DType.uint8,
                        )
                        for _ in range(_EXPERT_VARIANTS)
                    ]
                weights[name] = variants[key][int(parts[-3]) % _EXPERT_VARIANTS]
            else:
                key = (f"{projection}.scale", shape)
                if key not in variants:
                    exponent = _EXPERT_SCALE_EXPONENT[projection] + 127
                    scale = np.full(shape, exponent, np.uint8)
                    variants[key] = [_as(scale, DType.float8_e8m0fnu)]
                weights[name] = variants[key][0]
        elif parameter.dtype == DType.float8_e4m3fn:
            weights[name] = _fp8_codes(rng, shape)
        elif name.endswith("_proj.weight_scale"):
            weights[name] = _filled(_FP8_BLOCK_SCALE, shape, parameter.dtype)
        elif name.endswith("o_proj.weight"):
            weights[name] = _bf16(rng.normal(0.0, _O_PROJ_STD, shape))
        elif name.endswith("gate_score.weight"):
            weights[name] = _bf16(rng.normal(0.0, _ROUTER_STD, shape))
        elif name.endswith("layernorm.weight"):
            weights[name] = _filled(1.0, shape, parameter.dtype)
        elif name.endswith(("attention_sink_bias", "e_score_correction_bias")):
            weights[name] = _filled(0.0, shape, parameter.dtype)
        else:
            raise AssertionError(f"no replica weight rule for {name!r}")
    return weights


def stage_replica_trace(
    trace: MiMoReplicaTrace, devices: list[DeviceRef]
) -> tuple[StagedGraph[..., list[Tensor]], dict[str, Tensor]]:
    """Stages the trace with :func:`replica_weights`.

    The inputs are ``h0`` and the plant on the first device, the ragged row
    offsets, then the KV leaves.
    """
    norm = trace.layers[0].input_layernorm.weight
    symbolic = modulev3_ragged_kv_symbolic_inputs(
        kv_params=trace.kv_params, device_refs=devices
    )
    return trace.stage(
        TensorType(norm.dtype, ["rows", norm.shape[0]], device=devices[0]),
        TensorType(
            norm.dtype,
            [ALLREDUCES_PER_LAYER * len(trace.layers), len(devices)],
            device=devices[0],
        ),
        symbolic[2],
        *symbolic[3:],
        weights=replica_weights(trace),
    )


class MiMoReplicaTrace(Module[..., list[Tensor]]):
    """Runs MiMo decoder layers and returns per-rank boundary tensors.

    For each layer it returns, in order: every device's router input
    (``post_attention_layernorm(h)``), every device's expert ids on an MoE
    layer, every device's residual after the attention's reduction, and
    every device's residual after the MLP or MoE output's reduction.
    """

    def __init__(self, model: MiMoV2) -> None:
        self.layers = model.layers
        self.ropes = model.ropes
        self.kv_params = model.kv_params
        self.layer_groups = model._layer_groups

    def forward(
        self,
        h0: Tensor,
        plant: Tensor,
        input_row_offsets: Tensor,
        *kv_inputs: Tensor,
    ) -> list[Tensor]:
        """Returns each layer's boundary tensors on every device.

        Args:
            h0: The first layer's input rows, on the first device.
            plant: ``[allreduces, devices]`` power-of-two factors, on the
                first device. Row ``2 * layer`` scales each device's
                attention output and row ``2 * layer + 1`` its MLP or MoE
                output, just before their reductions.
            input_row_offsets: Ragged sequence boundaries.
            kv_inputs: The KV cache leaves.
        """
        mesh = self.layers[0].input_layernorm.weight.mesh
        h = h0.to(mesh)
        if mesh.num_devices > 1:
            input_row_offsets = F.distributed_broadcast(input_row_offsets, mesh)
            plant = F.distributed_broadcast(plant, mesh)
        kv_iter = iter(x._graph_value for x in kv_inputs)
        sliding_kv, full_kv = self.kv_params.unflatten_basic_kv_tree(kv_iter)
        kv_by_group = {SLIDING: list(sliding_kv), FULL: list(full_kv)}
        freqs_cis = {
            group: rope.freqs_cis.cast(h.dtype).to(h.mesh)
            for group, rope in self.ropes.items()
        }
        traces: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            group = self.layer_groups[i]
            h, layer_traces = self._layer_trace(
                layer,
                h,
                kv_by_group[group],
                freqs_cis[group],
                input_row_offsets,
                plant,
                ALLREDUCES_PER_LAYER * i,
            )
            traces.extend(layer_traces)
        return traces

    @staticmethod
    def _planted(partial: Tensor, plant: Tensor, row: int) -> Tensor:
        # A power-of-two factor is exact in bf16, and a factor of one leaves
        # the reduction's operands as the model computes them.
        shards = [
            shard * factors[row : row + 1, device : device + 1]
            for device, (shard, factors) in enumerate(
                zip(partial.local_shards, plant.local_shards, strict=True)
            )
        ]
        return Tensor.from_shard_values(shards, partial.mapping)

    @classmethod
    def _layer_trace(
        cls,
        layer: MiMoV2DecoderLayer,
        h: Tensor,
        kv_collection: list[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        plant: Tensor,
        first_allreduce: int,
    ) -> tuple[Tensor, list[Tensor]]:
        attn_out = layer.self_attn(
            layer.input_layernorm(h),
            kv_collection,
            freqs_cis,
            input_row_offsets,
        )
        h = h + cls._planted(attn_out, plant, first_allreduce).to(h.mapping)
        router_input = layer.post_attention_layernorm(h)
        traces = list(router_input.local_shards)
        if isinstance(layer.mlp, MiMoV2MoE):
            gate = layer.mlp.gate
            for rank in range(router_input.mesh.num_devices):
                expert_ids, _ = gate._route(
                    router_input.local_shards[rank],
                    gate.gate_score.weight.local_shards[rank],
                    gate.e_score_correction_bias.local_shards[rank],
                )
                traces.append(expert_ids)
            mlp_out = as_subgraph(layer.mlp, name="mimo_v2_moe")(router_input)
        else:
            mlp_out = layer.mlp(router_input)
        traces.extend(h.local_shards)
        h = h + cls._planted(mlp_out, plant, first_allreduce + 1).to(h.mapping)
        traces.extend(h.local_shards)
        return h, traces
