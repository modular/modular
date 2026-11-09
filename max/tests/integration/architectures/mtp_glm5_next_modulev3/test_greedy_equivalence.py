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
"""GLM-5.3-Flash with MTP on ModuleV3 decodes the tokens greedy decoding
without it does.

The graph-API test's (``mtp_glm5_next/test_greedy_equivalence.py``) on the
ModuleV3 pairing: a random-weight checkpoint, four layers deep, served through
the pipeline registry twice on two GPUs, with tensor-parallel attention and an
expert-parallel MoE. First with MTP at three drafts and device graph capture,
then plainly. Each token the MTP run emits must be the plain model's greedy
choice given the MTP run's own prefix, up to a near-tie of
:data:`NEAR_TIE_ULPS`, after which the check resumes from the MTP run's token.
The draft agrees with the target often but not always, so steps both accept
and reject drafts, and a KDA state landed on the wrong prefix moves the logits
past the tolerance.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, NonCallableMock

import numpy as np
import pytest
import torch
from max import functional as F
from max.driver import CPU, DeviceSpec
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, Weights
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.glm5_next.model_config import (
    Glm5NextConfig as GraphGlm5NextConfig,
)
from max.pipelines.architectures.glm5_next.quantization import (
    Glm5NextQuantScheme,
)
from max.pipelines.architectures.glm5_next_modulev3.glm5_next import Glm5Next
from max.pipelines.architectures.glm5_next_modulev3.model_config import (
    Glm5NextConfig,
)
from max.pipelines.architectures.glm5_next_modulev3.mtp import Glm5NextMTP
from max.pipelines.architectures.glm5_next_modulev3.weight_adapters import (
    glm5_next_mtp_weight_rules,
    glm5_next_weight_rules,
)
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    KVCacheConfig,
    ModelOutputs,
    OverlapTextGenerationPipeline,
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
    WeightTransform,
    _hf_config,
)
from max.pipelines.lib.generate import GenerateIterator
from max.pipelines.speculative import SpeculativeConfig
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from safetensors import safe_open
from safetensors.torch import save_file

Pipeline = OverlapTextGenerationPipeline[TextContext]

NUM_DRAFTS = 3
NUM_DEVICES = 2
NUM_LAYERS = 4
HIDDEN = 4096
VOCAB = 1024
SPECIAL_TOKENS = {
    "<think>": VOCAB - 3,
    "</think>": VOCAB - 2,
    "<eos>": VOCAB - 1,
}
"""The reasoning delimiters GLM's tokenizer looks up, and the end token. The
model never emits them: their rows of the head are zero."""
EOS = SPECIAL_TOKENS["<eos>"]
PROMPT_LENGTHS = (5, 38, 91)
MAX_NEW_TOKENS = 64
NEAR_TIE_ULPS = 4
"""Largest gap, in bfloat16 ulps of the top logit, between the plain model's
top logit and the MTP run's token. The runs' rounding differences measured at
most two."""
RESIDUAL_GAIN = 0.1
"""Scale of each sublayer's output against the unit-RMS token embedding."""
KDA_GAIN = 0.5
"""The KDA layers' output scale, larger so their state shows in the logits."""
KDA_OUTPUTS = {
    f"language_model.layers.{i}.self_attn.o_proj.weight"
    for i in range(NUM_LAYERS - 1)
}

TEXT_CONFIG: dict[str, object] = {
    "model_type": "glm5_next_text",
    "dtype": "bfloat16",
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "num_hidden_layers": NUM_LAYERS,
    "layer_types": ["linear_attention"] * (NUM_LAYERS - 1)
    + ["deepseek_sparse_attention"],
    "indexer_types": ["full"] * NUM_LAYERS,
    "first_k_dense_replace": 1,
    "intermediate_size": 1024,
    "moe_intermediate_size": 256,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "moe_router_dtype": "float32",
    "swiglu_limit": 10.0,
    "hidden_act": "silu",
    # The recipe's per-device attention shapes at tensor parallel 8.
    "num_attention_heads": 8 * NUM_DEVICES,
    "num_key_value_heads": 8 * NUM_DEVICES,
    "head_dim": 0,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "mla_use_nope": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 2048,
    "index_kpool": 4,
    "index_kpool_compress": True,
    "index_kpool_always_select_tail": True,
    "index_share_for_mtp_iteration": True,
    "linear_attn_config": {
        "num_heads": 8 * NUM_DEVICES,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": list(range(NUM_LAYERS - 1)),
        "full_attn_layers": [NUM_LAYERS - 1],
    },
    "mhc": True,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "num_nextn_predict_layers": 1,
    "tie_word_embeddings": False,
    "eos_token_id": [EOS],
    "pad_token_id": EOS,
}

VISION_CONFIG: dict[str, object] = {
    "depth": 1,
    "hidden_size": 128,
    "num_heads": 2,
    "intermediate_size": 256,
    "projection_intermediate_size": 256,
    "out_hidden_size": HIDDEN,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
}
"""The checkpoint's tower config. The ModuleV3 target is text-only, so the
checkpoint ships no tower weights."""

MEDIA_TOKENS = {
    name: VOCAB + i
    for i, name in enumerate(
        (
            "image_token_id",
            "image_start_token_id",
            "image_end_token_id",
            "video_token_id",
            "video_start_token_id",
            "video_end_token_id",
        )
    )
}
"""The media token ids GLM's tokenizer reads off the config, past the
vocabulary, since no prompt holds media."""

QUANTIZATION_CONFIG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    # As in the published checkpoint, the bfloat16 `kv_b_proj`s are listed:
    # the rules read which MLA projections stay unquantized off this list.
    "modules_to_not_convert": [
        "lm_head",
        "model.embed_tokens",
        *(
            f"model.layers.{i}.self_attn.kv_b_proj"
            for i in (NUM_LAYERS - 1, NUM_LAYERS)
        ),
    ],
}
"""The checkpoint's FP8 config."""

_MLA_RAW = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj")


def _quant_scheme() -> Glm5NextQuantScheme:
    """The checkpoint's FP8 map: every MLP, and the sparse and MTP layers'
    attention projections."""
    mlp_layers = frozenset(range(NUM_LAYERS + 1))
    attn_layers = frozenset({NUM_LAYERS - 1, NUM_LAYERS})
    quant = QuantConfig.blockscaled_fp8(
        mlp_quantized_layers=mlp_layers,
        attn_quantized_layers=attn_layers,
        embedding_output_dtype=DType.bfloat16,
    )
    return Glm5NextQuantScheme(
        config=quant,
        mlp_layers=mlp_layers,
        attn_layers=attn_layers,
        compute_dtype=DType.bfloat16,
    )


def _shapes() -> tuple[
    dict[str, tuple[DType, list[int]]], dict[str, tuple[DType, list[int]]]
]:
    """Returns the target's and the draft's parameters, by module name."""
    text_config = SimpleNamespace(**TEXT_CONFIG)
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = SpeculativeConfig(
        speculative_method="mtp", num_speculative_tokens=NUM_DRAFTS
    )
    kv_params = GraphGlm5NextConfig.construct_kv_params(
        huggingface_config=SimpleNamespace(text_config=text_config),
        pipeline_config=pipeline_config,
        devices=[DeviceRef.GPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
    )
    assert isinstance(kv_params, MultiKVCacheParams)
    scheme = _quant_scheme()
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    config = Glm5NextConfig(
        dtype=DType.float8_e4m3fn,
        kv_params=kv_params,
        devices=[DeviceRef.GPU()],
        max_seq_len=1024,
        max_position_embeddings=4096,
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=1024,
        moe_intermediate_size=256,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=8 * NUM_DEVICES,
        num_key_value_heads=8 * NUM_DEVICES,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=0,
        qk_nope_head_dim=256,
        v_head_dim=256,
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        correction_bias_dtype=DType.float32,
        graph_mode="decode",
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        indexer_types=["full"] * NUM_LAYERS,
        layer_types=list(text_config.layer_types),
        linear_num_heads=8 * NUM_DEVICES,
        linear_head_dim=128,
        linear_conv_kernel_dim=4,
        mla_latent_pad_to=576,
        declared_dtype=DType.bfloat16,
        quant_config=scheme.config,
        quant_scheme=scheme,
        mesh=mesh,
    )
    with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
        target = Glm5Next(config, kv_params)
        draft = Glm5NextMTP(config)

    def shapes(module: Any) -> dict[str, tuple[DType, list[int]]]:
        return {
            name: (t.dtype, [int(d) for d in t.shape])
            for name, t in module.parameters
        }

    return shapes(target), shapes(draft)


def _checkpoint_name(name: str) -> str:
    """Returns a target parameter's checkpoint name, the one the ModuleV3
    adapter renames to ``name``."""
    if name == "language_model.lm_head.weight":
        return "lm_head.weight"
    assert name.startswith("language_model."), name
    rest = name.removeprefix("language_model.")
    for site in ("hc_attn", "hc_ffn"):
        for part in ("fn", "base", "scale"):
            rest = rest.replace(f".{site}.hc_{part}", f".{site}_{part}")
    rest = rest.replace("mlp.gate.gate_score.weight", "mlp.gate.weight")
    rest = rest.replace(
        "self_attn.kv_a_proj_layernorm", "self_attn.kv_a_layernorm.weight"
    )
    rest = rest.replace(
        "indexer.index_kpool_compress_gate.weight",
        "indexer.index_kpool_compress_gate",
    )
    last = rest.rsplit(".", 1)[-1]
    if last.endswith("_conv1d") or last in _MLA_RAW:
        rest += ".weight"
    return "model.language_model." + rest


def _draft_checkpoint_name(name: str) -> str:
    """Places a draft parameter under the checkpoint's MTP layer."""
    prefix = f"layers.{NUM_LAYERS}."
    if name.startswith("decoder_layer."):
        return _checkpoint_name(
            "language_model." + prefix + name.removeprefix("decoder_layer.")
        )
    if name.startswith("shared_head_norm."):
        name = "shared_head.norm." + name.removeprefix("shared_head_norm.")
    return "model.language_model." + prefix + name


def _random_weights(
    shapes: dict[str, tuple[DType, list[int]]], rng: np.random.Generator
) -> dict[str, torch.Tensor]:
    """Returns weights that keep each sublayer's output near its gain
    against a unit-RMS residual."""
    out: dict[str, torch.Tensor] = {}
    for name, (dtype, shape) in shapes.items():
        if name.endswith(".weight_scale_inv"):
            continue
        last = name.rsplit(".", 1)[-1]
        fan_in = shape[-1] if len(shape) > 1 else 1
        if name in KDA_OUTPUTS:
            gain = KDA_GAIN
        elif name.endswith(
            ("o_proj.weight", "down_proj.weight", "f_b_proj.weight")
        ):
            gain = RESIDUAL_GAIN
        else:
            gain = 1.0
        if "norm" in name and len(shape) == 1:
            value = np.ones(shape)
        elif last == "e_score_correction_bias":
            # Every token reaches one expert on each device.
            value = np.zeros(shape)
            value[[0, shape[0] // 2]] = 1.0
        elif last == "bias" or name.endswith("_base"):
            value = np.zeros(shape)
        elif name.endswith("_scale"):
            value = np.ones(shape)
        elif last == "A_log":
            value = np.zeros(shape)
        elif last == "dt_bias":
            # With A = 1 and a small gate input, every channel decays by
            # about half per token, so the state holds a few recent tokens
            # and one wrong record changes it visibly.
            value = np.full(shape, -1.8)
        elif name.endswith("_fn"):
            value = rng.normal(0.0, 0.01, shape)
        elif name.endswith("embed_tokens.weight"):
            value = rng.normal(0.0, 1.0, shape)
        elif dtype == DType.float8_e4m3fn:
            value = np.clip(rng.normal(0.0, 1.0, shape), -4.0, 4.0)
            scale_shape = shapes[name + "_scale_inv"][1]
            out[name + "_scale_inv"] = torch.full(
                scale_shape, gain / np.sqrt(fan_in), dtype=torch.float32
            )
            out[name] = torch.from_numpy(value.astype(np.float32)).to(
                torch.float8_e4m3fn
            )
            continue
        else:
            value = rng.normal(0.0, gain / np.sqrt(fan_in), shape)
        out[name] = torch.from_numpy(np.asarray(value, np.float32)).to(
            torch.float32 if dtype == DType.float32 else torch.bfloat16
        )
    return out


def _write_tokenizer(path: Path) -> None:
    """A word-level vocabulary of ``t0``..., which the test never encodes
    with: it feeds token ids directly."""
    vocab = {f"t{i}": i for i in range(VOCAB - len(SPECIAL_TOKENS))}
    vocab.update(SPECIAL_TOKENS)
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [
                    {
                        "id": token_id,
                        "content": token,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                    for token, token_id in SPECIAL_TOKENS.items()
                ],
                "normalizer": None,
                "pre_tokenizer": {"type": "WhitespaceSplit"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": vocab,
                    "unk_token": "<eos>",
                },
            }
        )
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<eos>",
                "pad_token": "<eos>",
                "model_max_length": 4096,
            }
        )
    )


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("glm53_flash_tiny_modulev3")
    rng = np.random.default_rng(0)
    target_shapes, draft_shapes = _shapes()
    target_weights = _random_weights(target_shapes, rng)
    # No random logit row loses to a zero row, so every row runs to its
    # token budget without a reasoning delimiter.
    for token_id in SPECIAL_TOKENS.values():
        target_weights["language_model.lm_head.weight"][token_id] = 0
    draft_weights = _random_weights(draft_shapes, rng)
    # The draft reads only its token: `eh_proj` passes the normalized
    # embedding through and drops the target's hidden state.
    draft_weights["eh_proj.weight"] = torch.cat(
        [torch.eye(HIDDEN), torch.zeros(HIDDEN, HIDDEN)], dim=1
    ).to(torch.bfloat16)
    weights = {
        _checkpoint_name(name): value for name, value in target_weights.items()
    }
    weights.update(
        {
            _draft_checkpoint_name(name): value
            for name, value in draft_weights.items()
        }
    )
    save_file(weights, str(path / "model.safetensors"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm5NextForConditionalGeneration"],
                "model_type": "glm5_next",
                "text_config": TEXT_CONFIG,
                "vision_config": VISION_CONFIG,
                **MEDIA_TOKENS,
                "tie_word_embeddings": False,
                "quantization_config": QUANTIZATION_CONFIG,
            }
        )
    )
    _write_tokenizer(path)
    return path


def test_the_checkpoint_round_trips_through_the_rules(
    checkpoint: Path,
) -> None:
    """Checks the generated names are the ones the rules read, so every
    parameter loads from the weight written for it."""
    target_shapes, draft_shapes = _shapes()
    with safe_open(str(checkpoint / "model.safetensors"), "np") as f:
        names = set(f.keys())
    stub = {name: _name_only(name) for name in names}
    hf_config = SimpleNamespace(
        num_hidden_layers=NUM_LAYERS, quantization_config=QUANTIZATION_CONFIG
    )
    target = WeightTransform(glm5_next_weight_rules(hf_config))(stub)
    assert set(target) == set(target_shapes)
    draft = WeightTransform(glm5_next_mtp_weight_rules(hf_config))(stub)
    assert set(draft) == set(draft_shapes)


def _name_only(name: str) -> Weights:
    """A checkpoint weight the rules only rename."""
    weight = NonCallableMock()
    weight.data.return_value = WeightData.from_numpy(
        np.zeros((2, 2), np.float32), name
    )
    return weight


def _pipeline(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    speculative: bool,
) -> Pipeline:
    # A speculative config rewrites the cached HF config in place.
    monkeypatch.setattr(_hf_config, "_config_cache", {})
    register_all_models()
    config = PipelineConfig.from_args(
        PipelineArgs(
            model_path=str(checkpoint),
            device_specs=[
                DeviceSpec.accelerator(i) for i in range(NUM_DEVICES)
            ],
            quantization_encoding="float8_e4m3fn",
            max_length=1024,
            kv_cache=KVCacheConfig(
                enable_prefix_caching=False, device_memory_utilization=0.3
            ),
            runtime=PipelineRuntimeConfig(
                ep_size=NUM_DEVICES,
                prefer_module_v3=True,
                max_batch_size=len(PROMPT_LENGTHS),
                device_graph_capture=speculative,
                # The plain run reports the logits behind each token.
                disable_overlap=not speculative,
            ),
            speculative=(
                SpeculativeConfig(
                    speculative_method="mtp",
                    num_speculative_tokens=NUM_DRAFTS,
                )
                if speculative
                else None
            ),
        )
    )
    _, pipeline = PIPELINE_REGISTRY.retrieve(config)
    assert isinstance(pipeline, OverlapTextGenerationPipeline)
    return pipeline


def _prompts() -> list[list[int]]:
    rng = np.random.default_rng(1)
    return [
        [int(t) for t in rng.integers(0, VOCAB - len(SPECIAL_TOKENS), length)]
        for length in PROMPT_LENGTHS
    ]


class _CountingReplay:
    """Counts the steps that replay a captured graph."""

    def __init__(self, replay: Callable[..., ModelOutputs]) -> None:
        self.replay = replay
        self.calls = 0

    def __call__(self, **kwargs: object) -> ModelOutputs:
        self.calls += 1
        return self.replay(**kwargs)


def _generate_with_mtp(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[list[int]], list[list[int]], int]:
    """Returns each row's new tokens, how many each step added per row, and
    how many steps replayed a captured graph."""
    pipeline = _pipeline(checkpoint, monkeypatch, speculative=True)
    pipeline.warmup_graph_capture()
    runner = pipeline._graph_capture_runner
    assert runner is not None
    replays = _CountingReplay(runner.replay)
    monkeypatch.setattr(runner, "replay", replays)
    prompts = _prompts()
    per_step: list[list[int]] = [[] for _ in prompts]
    with GenerateIterator(
        pipeline, prompts, max_new_tokens=MAX_NEW_TOKENS, top_k=1
    ) as batch:
        while not batch.done:
            for row, tokens in batch.step().items():
                if tokens:
                    per_step[row].append(len(tokens))
        generated = [
            sequence[len(prompt) :]
            for sequence, prompt in zip(batch.sequences, prompts, strict=True)
        ]
    # Nothing may hold the pipeline once the plain run needs its memory.
    monkeypatch.undo()
    del pipeline, runner
    gc.collect()
    return generated, per_step, replays.calls


def _greedy(
    pipeline: Pipeline, prompt: list[int], max_new_tokens: int
) -> tuple[list[int], np.ndarray]:
    with GenerateIterator(
        pipeline,
        [prompt],
        max_new_tokens=max_new_tokens,
        top_k=1,
        ignore_eos=True,
        output_logits=True,
    ) as batch:
        while not batch.done:
            batch.step()
        tokens = batch.sequences[0][len(prompt) :]
        return tokens, batch.logits[0]


def _divergences(
    greedy: Callable[[list[int], int], tuple[list[int], np.ndarray]],
    prompt: list[int],
    mtp_tokens: list[int],
) -> list[tuple[int, float]]:
    """Returns where ``mtp_tokens`` leaves the plain model's greedy choice,
    with the plain model's gap there in bfloat16 ulps of its top logit.

    After each divergence the plain model resumes from the MTP run's token,
    so every token is checked against its own prefix.
    """
    found: list[tuple[int, float]] = []
    start = 0
    while start < len(mtp_tokens):
        tokens, logits = greedy(
            prompt + mtp_tokens[:start], len(mtp_tokens) - start
        )
        offset = next(
            (
                i
                for i, (got, want) in enumerate(
                    zip(mtp_tokens[start:], tokens, strict=True)
                )
                if got != want
            ),
            None,
        )
        if offset is None:
            break
        row = logits[offset]
        top = float(row.max())
        ulp = 2.0 ** (np.floor(np.log2(abs(top))) - 7)
        gap = top - float(row[mtp_tokens[start + offset]])
        found.append((start + offset, gap / ulp))
        start += offset + 1
    return found


def test_mtp_decodes_the_plain_models_greedy_tokens(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated, per_step, replays = _generate_with_mtp(checkpoint, monkeypatch)
    assert replays > 0, "no step replayed a captured graph"
    assert all(len(tokens) == MAX_NEW_TOKENS for tokens in generated)

    # The first step is the prefill's one token; every later step but the
    # last adds one plus the drafts it accepted.
    decode_steps = [n for steps in per_step for n in steps[1:-1]]
    accepted = sum(n - 1 for n in decode_steps)
    rejected = sum(NUM_DRAFTS + 1 - n for n in decode_steps)
    print(
        f"{len(decode_steps)} decode steps, {replays} replayed, accepted"
        f" {accepted} drafts and rejected {rejected}"
    )
    assert accepted > 0 and rejected > 0, (accepted, rejected)

    pipeline = _pipeline(checkpoint, monkeypatch, speculative=False)

    def greedy(prompt: list[int], n: int) -> tuple[list[int], np.ndarray]:
        return _greedy(pipeline, prompt, n)

    for prompt, tokens in zip(_prompts(), generated, strict=True):
        divergences = _divergences(greedy, prompt, tokens)
        print(f"prompt of {len(prompt)}: diverged at {divergences}")
        assert all(gap <= NEAR_TIE_ULPS for _, gap in divergences), divergences
