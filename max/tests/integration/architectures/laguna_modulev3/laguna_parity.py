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
"""Runs the ModuleV3 Laguna model against the graph architecture's.

Both pipeline models load the same random checkpoint, run the same prefill,
and must agree on the last-token logits. The graph architecture is the
one verified against vLLM on the released checkpoints, so agreement here is
what carries that verification over.

The checkpoint is shrunk but keeps every per-layer axis Laguna varies: full
and sliding attention, two query-head counts, a dense first layer and sparse
MoE layers, a partial-rotary YaRN table, and each gating mode. Prompts run past
the sliding window so the sliding mask is exercised.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from max import tree
from max.driver import Buffer, DeviceSpec, load_devices
from max.engine import InferenceSession
from max.graph.weights import SafetensorWeights
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.architectures.hf_config_shims import LagunaHFConfig
from max.pipelines.architectures.laguna import model as graph_model
from max.pipelines.architectures.laguna import weight_adapters as graph_adapters
from max.pipelines.architectures.laguna_modulev3 import model as v3_model
from max.pipelines.architectures.laguna_modulev3 import (
    weight_adapters as v3_adapters,
)
from max.pipelines.lib import MemoryPlan
from test_common.mocks import DummyPipelineConfig
from test_common.simple_kv_cache import paged_kv_cache_inputs

TESTDATA = Path(__file__).parents[1] / "laguna" / "testdata"

HIDDEN_SIZE = 256
HEAD_DIM = 64
NUM_KV_HEADS = 2
NUM_EXPERTS = 8
MOE_DIM = 128
DENSE_DIM = 512
VOCAB_SIZE = 512
SLIDING_WINDOW = 16
MAX_LENGTH = 1024
PROMPT_LENS = (40, 23)
YARN_FACTOR = 8.0
NVFP4_INPUT_GLOBAL_SCALE = 500.0

CASES: dict[str, dict[str, Any]] = {
    # S-2.1's precision split: NVFP4 routed experts, then bfloat16 ones in
    # the last layer.
    "s21_like_nvfp4": dict(
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_attention_heads_per_layer=[4, 8, 8, 4],
        gating="per-head",
        bias_dtype=torch.bfloat16,
        nvfp4_layers=frozenset({1, 2}),
    ),
    # M.1's shape: uniform full attention, a per-element gate, and a float32
    # correction bias.
    "m1_like": dict(
        layer_types=["full_attention"] * 3,
        num_attention_heads_per_layer=[4, 4, 4],
        gating="per-element",
        bias_dtype=torch.float32,
    ),
}


def _hf_config(case: dict[str, Any]) -> LagunaHFConfig:
    layer_types = case["layer_types"]
    num_layers = len(layer_types)
    extra: dict[str, Any] = {}
    if case.get("nvfp4_layers"):
        # The released checkpoint's own quantization block, so config parsing
        # takes the path it takes in production.
        with (TESTDATA / "s21_config.json").open() as f:
            extra["quantization_config"] = json.load(f)["quantization_config"]
    return LagunaHFConfig(
        **extra,
        architectures=["LagunaForCausalLM"],
        hidden_size=HIDDEN_SIZE,
        head_dim=HEAD_DIM,
        num_attention_heads=case["num_attention_heads_per_layer"][0],
        num_attention_heads_per_layer=case["num_attention_heads_per_layer"],
        num_key_value_heads=NUM_KV_HEADS,
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        mlp_layer_types=["dense"] + ["sparse"] * (num_layers - 1),
        mlp_only_layers=[0],
        intermediate_size=DENSE_DIM,
        moe_intermediate_size=MOE_DIM,
        shared_expert_intermediate_size=MOE_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        moe_routed_scaling_factor=2.5,
        norm_topk_prob=True,
        gating=case["gating"],
        sliding_window=SLIDING_WINDOW,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_LENGTH,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": YARN_FACTOR,
                "original_max_position_embeddings": MAX_LENGTH // 8,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "partial_rotary_factor": 0.5,
                "attention_factor": 0.1 * math.log(YARN_FACTOR) + 1.0,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
    )


def _nvfp4_projection(
    gen: torch.Generator, out_dim: int, in_dim: int, global_scale: float
) -> dict[str, torch.Tensor]:
    """A random compressed-tensors NVFP4 projection, by tensor suffix.

    Block scales are e4m3 bytes in [0.5, 1.0], and the global scales are in
    the compressed-tensors convention (the reciprocal of what MAX multiplies
    by), sized so dequantized weights sit near the bfloat16 layers' scale and
    activations fit the e4m3 block-scale range.
    """
    return {
        "weight_packed": torch.randint(
            0, 256, (out_dim, in_dim // 2), generator=gen, dtype=torch.uint8
        ),
        "weight_scale": torch.randint(
            0x30, 0x39, (out_dim, in_dim // 16), generator=gen
        ).to(torch.uint8),
        "weight_global_scale": torch.tensor([global_scale]),
        # One activation scale for every expert: the graph architecture
        # quantizes the gate/up input with the max across experts, ModuleV3
        # with each expert's own, and only a uniform scale makes them agree.
        "input_global_scale": torch.tensor([NVFP4_INPUT_GLOBAL_SCALE]),
    }


def _checkpoint(
    hf_config: LagunaHFConfig,
    bias_dtype: torch.dtype,
    nvfp4_layers: frozenset[int] = frozenset(),
) -> dict[str, torch.Tensor]:
    """Random weights under the checkpoint's own HuggingFace names.

    The routed experts of ``nvfp4_layers`` are packed NVFP4; everything else,
    shared experts included, is bfloat16, as in S-2.1.
    """
    gen = torch.Generator().manual_seed(0)

    def rand(*shape: int, scale: float = 0.05) -> torch.Tensor:
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

    def norm(dim: int) -> torch.Tensor:
        return (1.0 + torch.randn(dim, generator=gen) * 0.1).to(torch.bfloat16)

    weights = {
        "model.embed_tokens.weight": rand(VOCAB_SIZE, HIDDEN_SIZE, scale=1.0),
        "model.norm.weight": norm(HIDDEN_SIZE),
        "lm_head.weight": rand(VOCAB_SIZE, HIDDEN_SIZE),
    }
    for i, heads in enumerate(hf_config.num_attention_heads_per_layer):
        p = f"model.layers.{i}"
        q_dim = heads * HEAD_DIM
        kv_dim = NUM_KV_HEADS * HEAD_DIM
        gate_dim = heads if hf_config.gating == "per-head" else q_dim
        weights |= {
            f"{p}.input_layernorm.weight": norm(HIDDEN_SIZE),
            f"{p}.post_attention_layernorm.weight": norm(HIDDEN_SIZE),
            f"{p}.self_attn.q_proj.weight": rand(q_dim, HIDDEN_SIZE),
            f"{p}.self_attn.k_proj.weight": rand(kv_dim, HIDDEN_SIZE),
            f"{p}.self_attn.v_proj.weight": rand(kv_dim, HIDDEN_SIZE),
            f"{p}.self_attn.o_proj.weight": rand(HIDDEN_SIZE, q_dim),
            f"{p}.self_attn.g_proj.weight": rand(gate_dim, HIDDEN_SIZE),
            f"{p}.self_attn.q_norm.weight": norm(HEAD_DIM),
            f"{p}.self_attn.k_norm.weight": norm(HEAD_DIM),
        }
        if hf_config.mlp_layer_types[i] == "dense":
            mlps = {f"{p}.mlp": DENSE_DIM}
        else:
            weights[f"{p}.mlp.gate.weight"] = rand(NUM_EXPERTS, HIDDEN_SIZE)
            weights[f"{p}.mlp.experts.e_score_correction_bias"] = (
                torch.randn(NUM_EXPERTS, generator=gen) * 0.1
            ).to(bias_dtype)
            mlps = {f"{p}.mlp.shared_expert": MOE_DIM} | {
                f"{p}.mlp.experts.{e}": MOE_DIM for e in range(NUM_EXPERTS)
            }
        for prefix, dim in mlps.items():
            if i in nvfp4_layers and ".experts." in prefix:
                # Gate and up share a global scale, as the fused gate/up
                # matmul of both architectures assumes.
                expert_scale = 40.0 + 5.0 * int(prefix.rsplit(".", 1)[1])
                shapes = {
                    "gate_proj": (dim, HIDDEN_SIZE, expert_scale),
                    "up_proj": (dim, HIDDEN_SIZE, expert_scale),
                    "down_proj": (HIDDEN_SIZE, dim, 2 * expert_scale),
                }
                for proj, (out_dim, in_dim, scale) in shapes.items():
                    for suffix, tensor in _nvfp4_projection(
                        gen, out_dim, in_dim, scale
                    ).items():
                        weights[f"{prefix}.{proj}.{suffix}"] = tensor
                continue
            weights |= {
                f"{prefix}.gate_proj.weight": rand(dim, HIDDEN_SIZE),
                f"{prefix}.up_proj.weight": rand(dim, HIDDEN_SIZE),
                f"{prefix}.down_proj.weight": rand(HIDDEN_SIZE, dim),
            }
    return weights


def _weights(checkpoint: dict[str, torch.Tensor]) -> SafetensorWeights:
    return SafetensorWeights(
        [],
        tensors=set(checkpoint),
        tensors_to_file_idx={},
        _st_weight_map={
            name: Buffer.from_dlpack(tensor.contiguous())
            for name, tensor in checkpoint.items()
        },
    )


def _kv_inputs(
    kv_params: KVCacheParamInterface, prompt_lens: tuple[int, ...]
) -> list[Buffer]:
    """Fresh prefill KV buffers for every cache leaf, in graph-input order."""
    leaves = (
        list(kv_params.children.values())
        if isinstance(kv_params, MultiKVCacheParams)
        else [kv_params]
    )
    buffers: list[Buffer] = []
    for leaf in leaves:
        assert isinstance(leaf, KVCacheParams)
        buffers += tree.leaves(paged_kv_cache_inputs(leaf, prompt_lens))
    assert len(buffers) == len(kv_params.flattened_kv_inputs())
    return buffers


def _token_inputs(device: Any) -> tuple[Buffer, Buffer, Buffer]:
    gen = np.random.default_rng(1)
    tokens = gen.integers(0, VOCAB_SIZE, size=sum(PROMPT_LENS), dtype=np.int64)
    offsets = np.cumsum((0, *PROMPT_LENS)).astype(np.uint32)
    return (
        Buffer.from_numpy(tokens).to(device),
        Buffer.from_numpy(offsets).to(device),
        Buffer.from_numpy(np.array([1], dtype=np.int64)),
    )


def _build(model_cls: type, adapter: Any, case: dict[str, Any]) -> Any:
    hf_config = _hf_config(case)
    device_specs = [DeviceSpec.accelerator()]
    pipeline_config = DummyPipelineConfig(
        model_path="poolside/Laguna-S-2.1-NVFP4",
        quantization_encoding=(
            "float4_e2m1fnx2" if case.get("nvfp4_layers") else "bfloat16"
        ),
        max_batch_size=len(PROMPT_LENS),
        max_length=MAX_LENGTH,
        device_specs=device_specs,
        weight_path=[Path("fake.safetensors")],
        huggingface_config=hf_config,
    )
    devices = load_devices(device_specs)
    return model_cls(
        pipeline_config=pipeline_config,
        session=InferenceSession(devices=devices),
        devices=devices,
        kv_cache_config=pipeline_config.model.kv_cache,
        weights=_weights(
            _checkpoint(
                hf_config,
                case["bias_dtype"],
                case.get("nvfp4_layers", frozenset()),
            )
        ),
        adapter=adapter,
        memory_plan=MemoryPlan(
            planned_max_batch_size=len(PROMPT_LENS),
            footprint=0,
            planned_max_length=MAX_LENGTH,
        ),
    )


def _graph_logits(model: Any) -> torch.Tensor:
    tokens, offsets, n_logits = _token_inputs(model.devices[0])
    (logits,) = model.model.execute(
        tokens,
        offsets,
        n_logits,
        *model.signal_buffers,
        *_kv_inputs(model.kv_params, PROMPT_LENS),
    )
    return torch.from_dlpack(logits).float().cpu()


def _v3_logits(model: Any) -> torch.Tensor:
    tokens, offsets, n_logits = _token_inputs(model.devices[0])
    outputs = model.model(
        tokens, n_logits, offsets, *_kv_inputs(model.kv_params, PROMPT_LENS)
    )
    logits = outputs.next_token_logits
    return torch.from_dlpack(logits.driver_tensor).float().cpu()


def assert_logits_match(case_name: str) -> None:
    """Asserts both architectures produce the same logits for one case."""
    case = CASES[case_name]
    expected = _graph_logits(
        _build(
            graph_model.LagunaModel,
            graph_adapters.convert_safetensor_state_dict,
            case,
        )
    )
    actual = _v3_logits(
        _build(
            v3_model.LagunaModel,
            v3_adapters.convert_safetensor_state_dict,
            case,
        )
    )

    assert actual.shape == expected.shape == (len(PROMPT_LENS), VOCAB_SIZE)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
    assert (cosine > 0.999).all(), cosine
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
