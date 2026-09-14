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
"""Tests for the ModuleV3 Gemma4TextModel.

Port of max/tests/integration/architectures/gemma4/test_text_model.py to the
eager (ModuleV3) architecture. Structural checks run against the module's
``parameters`` qualified names instead of the graph arch's
``raw_state_dict()``; execution checks run the module eagerly with real
tensors instead of building a Graph.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from conftest import (
    TEXT_ATTENTION_K_EQ_V,
    TEXT_FINAL_LOGIT_SOFTCAPPING,
    TEXT_GLOBAL_HEAD_DIM,
    TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR,
    TEXT_GLOBAL_ROPE_THETA,
    TEXT_HEAD_DIM,
    TEXT_HIDDEN_ACTIVATION,
    TEXT_HIDDEN_SIZE,
    TEXT_INTERMEDIATE_SIZE,
    TEXT_LAYER_TYPES,
    TEXT_NUM_ATTENTION_HEADS,
    TEXT_NUM_GLOBAL_KEY_VALUE_HEADS,
    TEXT_NUM_HIDDEN_LAYERS,
    TEXT_NUM_KEY_VALUE_HEADS,
    TEXT_RMS_NORM_EPS,
    TEXT_SLIDING_WINDOW,
    TEXT_SLIDING_WINDOW_ROPE_THETA,
    TEXT_TIE_WORD_EMBEDDINGS,
    TEXT_VOCAB_SIZE,
    TorchGemma4TextModel,
)
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import Tensor, default_dtype
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    create_empty_embeddings,
    create_empty_indices,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from max.pipelines.architectures.gemma4_modulev3.gemma4 import Gemma4TextModel
from max.pipelines.architectures.gemma4_modulev3.layers.attention import (
    Gemma4Attention,
)
from max.pipelines.architectures.gemma4_modulev3.layers.transformer_block import (
    Gemma4TransformerBlock,
)
from max.pipelines.architectures.gemma4_modulev3.weight_adapters import (
    convert_language_state_dict_for_module,
)

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv_params(
    devices: list[DeviceRef],
) -> MultiKVCacheParams:
    """Build MultiKVCacheParams matching the gemma-4-31B-it config."""
    sliding_layers = sum(
        1 for t in TEXT_LAYER_TYPES if t == "sliding_attention"
    )
    global_layers = sum(1 for t in TEXT_LAYER_TYPES if t == "full_attention")

    sliding_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=TEXT_NUM_KEY_VALUE_HEADS,
        head_dim=TEXT_HEAD_DIM,
        num_layers=sliding_layers,
        devices=devices,
        page_size=128,
    )
    global_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=TEXT_NUM_GLOBAL_KEY_VALUE_HEADS,
        head_dim=TEXT_GLOBAL_HEAD_DIM,
        num_layers=global_layers,
        devices=devices,
        page_size=128,
    )
    return MultiKVCacheParams.from_params(
        {"sliding_attention": sliding_kv, "full_attention": global_kv}
    )


def _make_text_config(
    devices: list[DeviceRef],
    num_hidden_layers: int = TEXT_NUM_HIDDEN_LAYERS,
    layer_types: list[str] | None = None,
) -> Gemma4TextConfig:
    """Build a Gemma4TextConfig matching the canonical config."""
    if layer_types is None:
        layer_types = TEXT_LAYER_TYPES[:num_hidden_layers]

    # Text config uses the sliding-window KVCacheParams (inherited field).
    text_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=TEXT_NUM_KEY_VALUE_HEADS,
        head_dim=TEXT_HEAD_DIM,
        num_layers=num_hidden_layers,
        devices=devices,
        page_size=128,
    )

    return Gemma4TextConfig(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
        num_key_value_heads=TEXT_NUM_KEY_VALUE_HEADS,
        head_dim=TEXT_HEAD_DIM,
        hidden_activation="gelu_tanh",
        max_position_embeddings=131072,
        max_seq_len=131072,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        query_pre_attn_scalar=TEXT_HEAD_DIM,
        sliding_window=TEXT_SLIDING_WINDOW,
        final_logit_softcapping=TEXT_FINAL_LOGIT_SOFTCAPPING,
        attn_logit_softcapping=None,
        rope_local_base_freq=TEXT_SLIDING_WINDOW_ROPE_THETA,
        sliding_window_pattern=-1,
        dtype=MAX_DTYPE,
        devices=devices,
        interleaved_rope_weights=False,
        kv_params=text_kv,
        num_global_key_value_heads=TEXT_NUM_GLOBAL_KEY_VALUE_HEADS,
        global_head_dim=TEXT_GLOBAL_HEAD_DIM,
        attention_k_eq_v=TEXT_ATTENTION_K_EQ_V,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR,
        ),
        global_rope_theta=TEXT_GLOBAL_ROPE_THETA,
        sliding_window_rope_theta=TEXT_SLIDING_WINDOW_ROPE_THETA,
        layer_types=layer_types,
    )


def _make_vision_config() -> Gemma4VisionConfig:
    """Build a minimal Gemma4VisionConfig for constructing the top-level config."""
    return Gemma4VisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=72,
        hidden_activation="gelu_tanh",
        rms_norm_eps=1e-6,
        max_position_embeddings=131072,
        patch_size=16,
        position_embedding_size=10240,
        pooling_kernel_size=3,
    )


def _make_model_config(
    devices: list[DeviceRef],
    num_hidden_layers: int = TEXT_NUM_HIDDEN_LAYERS,
    layer_types: list[str] | None = None,
) -> Gemma4ForConditionalGenerationConfig:
    """Build a Gemma4ForConditionalGenerationConfig for model construction."""
    kv_params = _make_kv_params(devices)
    text_config = _make_text_config(devices, num_hidden_layers, layer_types)
    text_config.return_logits = ReturnLogits.LAST_TOKEN
    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        dtype=MAX_DTYPE,
        kv_params=kv_params,
        text_config=text_config,
        vision_config=_make_vision_config(),
        image_token_index=258880,
        tie_word_embeddings=TEXT_TIE_WORD_EMBEDDINGS,
    )


def _build_lazy_model(
    config: Gemma4ForConditionalGenerationConfig,
) -> Gemma4TextModel:
    """Build a Gemma4TextModel lazily on CPU (structure only, no realization)."""
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    with F.lazy(), default_dtype(MAX_DTYPE):
        return Gemma4TextModel(config, mesh)


def _param_names(model: Gemma4TextModel) -> set[str]:
    return {name for name, _ in model.parameters}


@pytest.fixture(scope="module")
def full_model() -> Gemma4TextModel:
    """The canonical 60-layer tied-embedding model, built lazily on CPU."""
    return _build_lazy_model(_make_model_config([DeviceRef.CPU()]))


@pytest.fixture(scope="module")
def small_model() -> Gemma4TextModel:
    """A 6-layer (5 sliding + 1 full) tied-embedding model, built lazily."""
    num_layers = 6
    config = _make_model_config(
        [DeviceRef.CPU()],
        num_hidden_layers=num_layers,
        layer_types=TEXT_LAYER_TYPES[:num_layers],
    )
    return _build_lazy_model(config)


# ---------------------------------------------------------------------------
# Tests: Module parameter structure
# ---------------------------------------------------------------------------


def test_parameters_have_embed_tokens(full_model: Gemma4TextModel) -> None:
    """Verify the module tree contains embedding weights."""
    embed_keys = [
        k for k in _param_names(full_model) if k.startswith("embed_tokens.")
    ]
    assert len(embed_keys) > 0, "Expected embed_tokens weights in parameters"


def test_parameters_have_norm(full_model: Gemma4TextModel) -> None:
    """Verify the module tree contains the final norm weight."""
    assert "norm.weight" in _param_names(full_model)


def test_parameters_have_lm_head_when_untied() -> None:
    """Verify the module tree contains lm_head weights when not tied."""
    num_layers = 2
    config = _make_model_config(
        [DeviceRef.CPU()],
        num_hidden_layers=num_layers,
        layer_types=TEXT_LAYER_TYPES[:num_layers],
    )
    config.tie_word_embeddings = False
    model = _build_lazy_model(config)
    lm_head_keys = [k for k in _param_names(model) if k.startswith("lm_head.")]
    assert len(lm_head_keys) > 0, "Expected lm_head weights in parameters"


def test_tied_embedding_omits_lm_head(full_model: Gemma4TextModel) -> None:
    """When tie_word_embeddings=True, no lm_head parameter should exist.

    Intentional V3 difference: the graph arch registered a
    ColumnParallelLinear whose tied_weight aliased the embedding; the V3
    module sets ``lm_head = None`` outright and matmuls against
    ``embed_tokens.weight`` in ``_compute_logits``.
    """
    names = _param_names(full_model)
    assert "embed_tokens.weight" in names
    assert not any(k.startswith("lm_head.") for k in names)
    assert full_model.lm_head is None


def test_parameters_layer_count(small_model: Gemma4TextModel) -> None:
    """Verify the correct number of layers are present in the module tree."""
    layer_indices = {
        int(k.split(".")[1])
        for k in _param_names(small_model)
        if k.startswith("layers.")
    }
    assert layer_indices == set(range(6))


def test_all_layers_have_layer_scalar(small_model: Gemma4TextModel) -> None:
    """Verify all decoder layers have a layer_scalar parameter."""
    names = _param_names(small_model)
    for i in range(6):
        key = f"layers.{i}.layer_scalar"
        assert key in names, f"Missing {key}"


def test_four_norms_per_layer(small_model: Gemma4TextModel) -> None:
    """Verify each decoder layer has all four norm weights."""
    names = _param_names(small_model)
    norm_names = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ]
    for i in range(6):
        for norm_name in norm_names:
            key = f"layers.{i}.{norm_name}.weight"
            assert key in names, f"Missing {key}"


def test_v_proj_only_on_sliding_layers(small_model: Gemma4TextModel) -> None:
    """Sliding layers register v_proj; global k==v layers do not.

    Intentional V3 difference: attention fuses QKV at graph-build time via
    the ``wqkv`` property, but still registers separate q/k/v projection
    parameters (no fused ``wqkv`` parameter exists). With
    ``attention_k_eq_v=True`` the checkpoint has no v_proj on full-attention
    layers, so the module omits the parameter there and reuses K for V.
    """
    names = _param_names(small_model)
    assert not any("wqkv" in k for k in names)
    for i, layer_type in enumerate(TEXT_LAYER_TYPES[:6]):
        for proj in ("q_proj", "k_proj", "o_proj"):
            key = f"layers.{i}.self_attn.{proj}.weight"
            assert key in names, f"Missing {key}"
        v_key = f"layers.{i}.self_attn.v_proj.weight"
        if layer_type == "sliding_attention":
            assert v_key in names, f"Missing {v_key}"
        else:
            assert v_key not in names, f"Unexpected {v_key} on k==v layer"


# ---------------------------------------------------------------------------
# Tests: Layer-to-KV-cache routing
# ---------------------------------------------------------------------------


def test_layer_kv_key_length(full_model: Gemma4TextModel) -> None:
    """_layer_kv_key should have one entry per layer."""
    assert len(full_model._layer_kv_key) == TEXT_NUM_HIDDEN_LAYERS


def _attention(model: Gemma4TextModel, layer_idx: int) -> Gemma4Attention:
    layer = model.layers[layer_idx]
    assert isinstance(layer, Gemma4TransformerBlock)
    attn = layer.self_attn
    assert isinstance(attn, Gemma4Attention)
    return attn


@pytest.mark.parametrize(
    "layer_idx",
    [0, 1, 4],
    ids=["layer_0_sliding", "layer_1_sliding", "layer_4_sliding"],
)
def test_sliding_layers_map_to_sliding_cache(
    full_model: Gemma4TextModel, layer_idx: int
) -> None:
    """Sliding attention layers use the sliding cache key and geometry."""
    assert TEXT_LAYER_TYPES[layer_idx] == "sliding_attention"
    assert full_model._layer_kv_key[layer_idx] == "sliding_attention"
    attn = _attention(full_model, layer_idx)
    assert attn.use_local
    assert isinstance(attn.kv_params, MHAKVCacheParams)
    assert attn.kv_params.n_kv_heads == TEXT_NUM_KEY_VALUE_HEADS
    assert attn.kv_params.head_dim == TEXT_HEAD_DIM
    assert attn.num_key_value_heads == TEXT_NUM_KEY_VALUE_HEADS
    assert attn.head_dim == TEXT_HEAD_DIM


@pytest.mark.parametrize(
    "layer_idx",
    [5, 11, 59],
    ids=["layer_5_full", "layer_11_full", "layer_59_full"],
)
def test_full_attention_layers_map_to_global_cache(
    full_model: Gemma4TextModel, layer_idx: int
) -> None:
    """Full attention layers use the full-attention cache key and geometry."""
    assert TEXT_LAYER_TYPES[layer_idx] == "full_attention"
    assert full_model._layer_kv_key[layer_idx] == "full_attention"
    attn = _attention(full_model, layer_idx)
    assert not attn.use_local
    assert isinstance(attn.kv_params, MHAKVCacheParams)
    assert attn.kv_params.n_kv_heads == TEXT_NUM_GLOBAL_KEY_VALUE_HEADS
    assert attn.kv_params.head_dim == TEXT_GLOBAL_HEAD_DIM
    assert attn.num_key_value_heads == TEXT_NUM_GLOBAL_KEY_VALUE_HEADS
    assert attn.head_dim == TEXT_GLOBAL_HEAD_DIM


def test_layer_kv_key_matches_layer_types(
    full_model: Gemma4TextModel,
) -> None:
    """Every _layer_kv_key entry should match its layer type."""
    for i, (kv_key, layer_type) in enumerate(
        zip(full_model._layer_kv_key, TEXT_LAYER_TYPES, strict=True)
    ):
        assert kv_key == layer_type, (
            f"Layer {i}: expected KV key {layer_type}, got {kv_key}"
        )


def test_kv_key_counts_match_layer_type_counts(
    full_model: Gemma4TextModel,
) -> None:
    """The number of sliding/global KV keys should match the layer type counts."""
    sliding_count = sum(
        1 for k in full_model._layer_kv_key if k == "sliding_attention"
    )
    global_count = sum(
        1 for k in full_model._layer_kv_key if k == "full_attention"
    )

    expected_sliding = sum(
        1 for t in TEXT_LAYER_TYPES if t == "sliding_attention"
    )
    expected_global = sum(1 for t in TEXT_LAYER_TYPES if t == "full_attention")

    assert sliding_count == expected_sliding
    assert global_count == expected_global


def test_layer_idx_in_cache_increments_per_type(
    full_model: Gemma4TextModel,
) -> None:
    """Each attention layer indexes its own cache by per-type position.

    V3 equivalent of the graph arch's layer->KV-index mapping: instead of a
    per-layer cache index into a MultiKVCacheParams list, each attention
    holds ``layer_idx_in_cache``, its position among layers of the same
    type. It must count 0..N-1 within each type, in layer order.
    """
    counts = {"sliding_attention": 0, "full_attention": 0}
    for i, layer_type in enumerate(TEXT_LAYER_TYPES):
        attn = _attention(full_model, i)
        assert attn.layer_idx_in_cache == counts[layer_type], (
            f"Layer {i} ({layer_type}): expected cache index "
            f"{counts[layer_type]}, got {attn.layer_idx_in_cache}"
        )
        counts[layer_type] += 1


# ---------------------------------------------------------------------------
# Tests: Model structure mirrors torch reference
# ---------------------------------------------------------------------------


def _torch_identity_attn_factory(**kwargs: Any) -> torch.nn.Module:
    """Factory that creates identity attention stubs."""

    class _Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor, **kw: Any) -> torch.Tensor:
            return x

    return _Identity()


def _make_torch_model(num_layers: int) -> TorchGemma4TextModel:
    return TorchGemma4TextModel(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        num_hidden_layers=num_layers,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        hidden_activation=TEXT_HIDDEN_ACTIVATION,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        layer_types=TEXT_LAYER_TYPES[:num_layers],
        attn_factory=_torch_identity_attn_factory,
    )


def test_torch_reference_layer_count_matches(
    small_model: Gemma4TextModel,
) -> None:
    """Verify the torch and MAX models have the same number of layers."""
    torch_model = _make_torch_model(num_layers=6)
    assert len(small_model.layers) == len(torch_model.layers) == 6


def test_embed_scale_matches_torch(full_model: Gemma4TextModel) -> None:
    """Verify the embedding scale matches the torch reference."""
    expected_scale = TEXT_HIDDEN_SIZE**0.5
    assert full_model.embed_tokens.embed_scale == pytest.approx(expected_scale)


def test_parameter_prefixes_structurally_match_torch() -> None:
    """Verify the MAX module has the same top-level parameter prefixes as torch.

    Intentional V3 difference from the graph-arch test: with tied embeddings
    the V3 module registers no lm_head at all, so the prefix sets are equal
    rather than torch being a strict subset.
    """
    num_layers = 2
    torch_model = _make_torch_model(num_layers)
    torch_prefixes = {k.split(".")[0] for k in torch_model.state_dict()}

    config = _make_model_config(
        [DeviceRef.CPU()],
        num_hidden_layers=num_layers,
        layer_types=TEXT_LAYER_TYPES[:num_layers],
    )
    assert config.tie_word_embeddings
    max_model = _build_lazy_model(config)
    max_prefixes = {k.split(".")[0] for k in _param_names(max_model)}

    assert torch_prefixes == max_prefixes, (
        f"Torch prefixes {torch_prefixes} != MAX prefixes {max_prefixes}"
    )


def test_per_layer_norm_keys_match_torch() -> None:
    """Verify each layer's norm weight keys match the torch reference."""
    num_layers = 2
    torch_model = _make_torch_model(num_layers)

    config = _make_model_config(
        [DeviceRef.CPU()],
        num_hidden_layers=num_layers,
        layer_types=TEXT_LAYER_TYPES[:num_layers],
    )
    max_model = _build_lazy_model(config)

    for i in range(num_layers):
        torch_layer_keys = {
            k.replace(f"layers.{i}.", "")
            for k in torch_model.state_dict()
            if k.startswith(f"layers.{i}.")
        }
        max_layer_keys = {
            k.replace(f"layers.{i}.", "")
            for k in _param_names(max_model)
            if k.startswith(f"layers.{i}.")
        }

        norm_keys = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        }
        assert norm_keys.issubset(torch_layer_keys), (
            f"Torch layer {i} missing norm keys: {norm_keys - torch_layer_keys}"
        )
        assert norm_keys.issubset(max_layer_keys), (
            f"MAX layer {i} missing norm keys: {norm_keys - max_layer_keys}"
        )


# ---------------------------------------------------------------------------
# Tests: Execution (embed → layers → norm → logits)
# ---------------------------------------------------------------------------

# Reduced dimensions for execution tests.  These are intentionally smaller
# than the 31B reference to keep memory low.  All other config values (eps,
# activation, rope thetas, etc.) reuse the 31B constants from conftest so
# the test mirrors the real model structure.
_EXEC_HIDDEN = 64
_EXEC_INTERMEDIATE = 128
_EXEC_VOCAB = 256
_EXEC_NUM_LAYERS = 6  # 5 sliding + 1 full (preserves the 5:1 ratio)
_EXEC_LAYER_TYPES = [
    "sliding_attention" if (i + 1) % 6 else "full_attention"
    for i in range(_EXEC_NUM_LAYERS)
]
_EXEC_HEAD_DIM = 32
_EXEC_GLOBAL_HEAD_DIM = 64
_EXEC_N_HEADS = 2
_EXEC_N_KV_HEADS = 2
_EXEC_N_GLOBAL_KV_HEADS = 1


def _make_small_model_config(
    devices: list[DeviceRef],
) -> Gemma4ForConditionalGenerationConfig:
    """Build a small Gemma4 config suitable for execution tests."""
    sliding_layers = sum(
        1 for t in _EXEC_LAYER_TYPES if t == "sliding_attention"
    )
    global_layers = sum(1 for t in _EXEC_LAYER_TYPES if t == "full_attention")

    sliding_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_EXEC_N_KV_HEADS,
        head_dim=_EXEC_HEAD_DIM,
        num_layers=sliding_layers,
        devices=devices,
        page_size=128,
    )
    global_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_EXEC_N_GLOBAL_KV_HEADS,
        head_dim=_EXEC_GLOBAL_HEAD_DIM,
        num_layers=global_layers,
        devices=devices,
        page_size=128,
    )
    kv_params = MultiKVCacheParams.from_params(
        {"sliding_attention": sliding_kv, "full_attention": global_kv}
    )

    text_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_EXEC_N_KV_HEADS,
        head_dim=_EXEC_HEAD_DIM,
        num_layers=_EXEC_NUM_LAYERS,
        devices=devices,
        page_size=128,
    )

    text_config = Gemma4TextConfig(
        vocab_size=_EXEC_VOCAB,
        hidden_size=_EXEC_HIDDEN,
        intermediate_size=_EXEC_INTERMEDIATE,
        num_hidden_layers=_EXEC_NUM_LAYERS,
        num_attention_heads=_EXEC_N_HEADS,
        num_key_value_heads=_EXEC_N_KV_HEADS,
        head_dim=_EXEC_HEAD_DIM,
        hidden_activation="gelu_tanh",
        max_position_embeddings=1024,
        max_seq_len=1024,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        query_pre_attn_scalar=_EXEC_HEAD_DIM,
        sliding_window=TEXT_SLIDING_WINDOW,
        final_logit_softcapping=TEXT_FINAL_LOGIT_SOFTCAPPING,
        attn_logit_softcapping=None,
        rope_local_base_freq=TEXT_SLIDING_WINDOW_ROPE_THETA,
        sliding_window_pattern=-1,
        dtype=MAX_DTYPE,
        devices=devices,
        interleaved_rope_weights=False,
        kv_params=text_kv,
        num_global_key_value_heads=_EXEC_N_GLOBAL_KV_HEADS,
        global_head_dim=_EXEC_GLOBAL_HEAD_DIM,
        attention_k_eq_v=TEXT_ATTENTION_K_EQ_V,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR,
        ),
        global_rope_theta=TEXT_GLOBAL_ROPE_THETA,
        sliding_window_rope_theta=TEXT_SLIDING_WINDOW_ROPE_THETA,
        layer_types=_EXEC_LAYER_TYPES,
    )
    text_config.return_logits = ReturnLogits.LAST_TOKEN

    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        dtype=MAX_DTYPE,
        kv_params=kv_params,
        text_config=text_config,
        vision_config=_make_vision_config(),
        image_token_index=0,
        tie_word_embeddings=TEXT_TIE_WORD_EMBEDDINGS,
    )


class _IdentityAttention(Module[..., Tensor]):
    """Identity stand-in for Gemma4Attention; ignores the KV collection.

    This avoids the need for real KV caches or flash-attention kernels
    while keeping the rest of the forward pass (norms, MLP, residuals,
    layer_scalar, embedding, logits) intact.
    """

    def forward(
        self,
        x: Tensor,
        kv_collection: Any,
        *,
        input_row_offsets: Tensor,
    ) -> Tensor:
        return x


def _stub_attention(model: Gemma4TextModel) -> None:
    """Replace every decoder layer's attention with an identity module."""
    for layer in model.layers:
        assert isinstance(layer, Gemma4TransformerBlock)
        layer.self_attn = _IdentityAttention()


def _build_shared_weights(
    max_model: Gemma4TextModel,
) -> dict[str, torch.Tensor]:
    """Generate random weights shared between the MAX and torch models.

    The module's parameter names (with attention stubbed out) coincide with
    the torch reference's state-dict names, so one dict serves both.
    """
    torch.manual_seed(42)
    w: dict[str, torch.Tensor] = {}

    for key, param in max_model.parameters:
        shape = tuple(int(d) for d in param.shape)
        w[key] = torch.randn(shape, dtype=TORCH_DTYPE)

    # Set layer scalars to a non-trivial value.
    for key in w:
        if key.endswith(".layer_scalar"):
            w[key] = torch.tensor([2.0], dtype=TORCH_DTYPE)

    return w


def _build_eager_stubbed_model(
    device: Accelerator,
) -> tuple[Gemma4TextModel, dict[str, torch.Tensor]]:
    """Build the small model eagerly on GPU, stub attention, load weights."""
    mesh = DeviceMesh((device,), (1,), ("tp",))
    config = _make_small_model_config([DeviceRef.GPU()])
    config.text_config.return_logits = ReturnLogits.ALL
    with default_dtype(MAX_DTYPE):
        model = Gemma4TextModel(config, mesh)
    _stub_attention(model)

    shared_weights = _build_shared_weights(model)
    # Load first, then transfer: for single-device (non-distributed)
    # parameters load_state_dict adopts the source tensor's device, so the
    # CPU torch weights would otherwise stay on host.
    model.load_state_dict(shared_weights)
    model.to(mesh)
    return model, shared_weights


def _model_inputs(
    device: Accelerator, tokens: torch.Tensor, seq_len: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build (tokens, row_offsets, return_n_logits, image_emb, image_idx)."""
    tokens_t = Tensor.from_dlpack(tokens).to(device)
    row_offsets = torch.tensor([0, seq_len], dtype=torch.uint32)
    row_offsets_t = Tensor.from_dlpack(row_offsets).to(device)
    # Unused by ReturnLogits.ALL / LAST_TOKEN but required as an input.
    return_n_logits_t = Tensor.from_dlpack(
        torch.tensor([1], dtype=torch.uint32)
    )
    image_embeddings = Tensor.from_dlpack(
        create_empty_embeddings([device], _EXEC_HIDDEN, DType.bfloat16)[0]
    )
    image_token_indices = Tensor.from_dlpack(create_empty_indices([device])[0])
    return (
        tokens_t,
        row_offsets_t,
        return_n_logits_t,
        image_embeddings,
        image_token_indices,
    )


def _to_torch(t: Tensor) -> torch.Tensor:
    return torch.from_dlpack(t).cpu()


@pytest.mark.parametrize(
    "seq_len",
    [1],
    ids=["single_token"],
)
def test_text_model_execution_matches_torch(seq_len: int) -> None:
    """Execute the text model eagerly and compare against the torch reference.

    Uses identity attention stubs so the test exercises: scaled embedding,
    multimodal merge (no-op for text), 4-norm decoder layers, gelu_tanh MLP,
    layer_scalar, final norm, tied logits, and final logit softcapping.
    Both sliding and full attention layer types are present.

    Uses ``ReturnLogits.ALL`` so logits for every token position are
    compared, plus the LAST_TOKEN gather path via the first output.
    """
    device = Accelerator(0)
    max_model, shared_weights = _build_eager_stubbed_model(device)

    torch_model = TorchGemma4TextModel(
        vocab_size=_EXEC_VOCAB,
        hidden_size=_EXEC_HIDDEN,
        num_hidden_layers=_EXEC_NUM_LAYERS,
        intermediate_size=_EXEC_INTERMEDIATE,
        hidden_activation=TEXT_HIDDEN_ACTIVATION,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        layer_types=_EXEC_LAYER_TYPES,
        attn_factory=_torch_identity_attn_factory,
    )
    torch_model.load_state_dict(shared_weights, strict=False)

    # Input tokens (valid indices into the small vocab).
    torch.manual_seed(99)
    tokens = torch.randint(0, _EXEC_VOCAB, (seq_len,), dtype=torch.int64)

    # -- Run torch reference (embed → layers → norm → tied head) --
    torch_model = torch_model.to(TORCH_DTYPE)
    with torch.no_grad():
        torch_hidden = torch_model(tokens)
    embed_w = shared_weights["embed_tokens.weight"].to(TORCH_DTYPE)
    # Match the module's logit path: cast to float32, then softcap
    # (cap * tanh(logits / cap)).
    torch_logits = (torch_hidden @ embed_w.T).float()
    cap = TEXT_FINAL_LOGIT_SOFTCAPPING
    torch_logits = torch.tanh(torch_logits / cap) * cap

    # -- Run the MAX module eagerly, with None for the unused KV caches --
    inputs = _model_inputs(device, tokens, seq_len)
    tokens_t, row_offsets_t, return_n_logits_t, image_emb, image_idx = inputs
    results = max_model(
        tokens_t,
        None,
        None,
        return_n_logits_t,
        row_offsets_t,
        image_emb,
        image_idx,
    )
    # ReturnLogits.ALL returns (last_logits, all_logits, offsets).
    assert len(results) == 3
    max_last_logits = _to_torch(results[0]).float()
    max_logits = _to_torch(results[1]).float()

    # Compare all token logits, and the LAST_TOKEN gather path.
    torch.testing.assert_close(torch_logits, max_logits, rtol=0.02, atol=0.07)
    torch.testing.assert_close(
        torch_logits[-1:], max_last_logits, rtol=0.02, atol=0.07
    )


def test_text_model_per_layer_hidden_states_match_torch() -> None:
    """Per-layer post-block outputs match the torch reference at tapped layers.

    Port of the graph arch's SELECTED_LAYERS test. Intentional V3
    difference: the eager Gemma4TextModel has no
    ``ReturnHiddenStates.SELECTED_LAYERS`` return path, so the taps are
    collected eagerly by running the model's own embed/layer modules and
    recording the post-block hidden state at each target layer.
    """
    device = Accelerator(0)
    seq_len = 4
    target_layer_ids = [1, 3, 5]

    max_model, shared_weights = _build_eager_stubbed_model(device)

    torch_model = TorchGemma4TextModel(
        vocab_size=_EXEC_VOCAB,
        hidden_size=_EXEC_HIDDEN,
        num_hidden_layers=_EXEC_NUM_LAYERS,
        intermediate_size=_EXEC_INTERMEDIATE,
        hidden_activation=TEXT_HIDDEN_ACTIVATION,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        layer_types=_EXEC_LAYER_TYPES,
        attn_factory=_torch_identity_attn_factory,
    )
    torch_model.load_state_dict(shared_weights, strict=False)

    torch.manual_seed(99)
    tokens = torch.randint(0, _EXEC_VOCAB, (seq_len,), dtype=torch.int64)

    # Capture each tapped layer's post-block output (torch layers return a
    # tuple whose first element is the hidden state).
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(layer_id: int) -> Any:
        def _hook(module: torch.nn.Module, args: Any, output: Any) -> None:
            captured[layer_id] = output[0].detach()

        return _hook

    for layer_id in target_layer_ids:
        torch_model.layers[layer_id].register_forward_hook(_make_hook(layer_id))
    torch_model = torch_model.to(TORCH_DTYPE)
    with torch.no_grad():
        torch_model(tokens)

    # -- Run the MAX modules eagerly, tapping each target layer --
    inputs = _model_inputs(device, tokens, seq_len)
    tokens_t, row_offsets_t, _, _, _ = inputs
    tokens_mesh = tokens_t.to(max_model.mesh)
    row_offsets_mesh = row_offsets_t.to(max_model.mesh)

    h = max_model.embed_tokens(tokens_mesh)
    max_taps: dict[int, torch.Tensor] = {}
    for idx, layer in enumerate(max_model.layers):
        h = layer(h, None, input_row_offsets=row_offsets_mesh)
        if idx in target_layer_ids:
            max_taps[idx] = _to_torch(h).float()

    for layer_id in target_layer_ids:
        tap = max_taps[layer_id]
        assert tap.shape == (seq_len, _EXEC_HIDDEN), (
            f"Unexpected tap shape {tuple(tap.shape)}"
        )
        torch.testing.assert_close(
            captured[layer_id].float(), tap, rtol=0.02, atol=0.5
        )


# ---------------------------------------------------------------------------
# Tests: Checkpoint key conversion matches the module tree
# ---------------------------------------------------------------------------


def _small_checkpoint() -> dict[str, torch.Tensor]:
    """Checkpoint-shaped language keys for the small execution config.

    Mirrors the HF gemma4 layout: everything under ``model.language_model.``,
    no v_proj on k==v full-attention layers, no lm_head when tied.
    """

    def z(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape, dtype=torch.bfloat16)

    wm: dict[str, torch.Tensor] = {}
    prefix = "model.language_model."
    wm[prefix + "embed_tokens.weight"] = z(_EXEC_VOCAB, _EXEC_HIDDEN)
    wm[prefix + "norm.weight"] = z(_EXEC_HIDDEN)
    for i, layer_type in enumerate(_EXEC_LAYER_TYPES):
        lp = f"{prefix}layers.{i}."
        sliding = layer_type == "sliding_attention"
        hd = _EXEC_HEAD_DIM if sliding else _EXEC_GLOBAL_HEAD_DIM
        n_kv = _EXEC_N_KV_HEADS if sliding else _EXEC_N_GLOBAL_KV_HEADS
        wm[lp + "self_attn.q_proj.weight"] = z(_EXEC_N_HEADS * hd, _EXEC_HIDDEN)
        wm[lp + "self_attn.k_proj.weight"] = z(n_kv * hd, _EXEC_HIDDEN)
        if sliding:
            wm[lp + "self_attn.v_proj.weight"] = z(n_kv * hd, _EXEC_HIDDEN)
        wm[lp + "self_attn.o_proj.weight"] = z(_EXEC_HIDDEN, _EXEC_N_HEADS * hd)
        wm[lp + "self_attn.q_norm.weight"] = z(hd)
        wm[lp + "self_attn.k_norm.weight"] = z(hd)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            wm[lp + norm + ".weight"] = z(_EXEC_HIDDEN)
        wm[lp + "mlp.gate_proj.weight"] = z(_EXEC_INTERMEDIATE, _EXEC_HIDDEN)
        wm[lp + "mlp.up_proj.weight"] = z(_EXEC_INTERMEDIATE, _EXEC_HIDDEN)
        wm[lp + "mlp.down_proj.weight"] = z(_EXEC_HIDDEN, _EXEC_INTERMEDIATE)
        wm[lp + "layer_scalar"] = z(1)
    return wm


def test_checkpoint_key_conversion_matches_module_tree() -> None:
    """The language converter's output keys must exactly cover the module.

    ``Module.compile(weights=...)`` silently ignores unmatched entries, so
    without this a misnamed checkpoint key would leave a parameter serving
    its default initialization.
    """
    wm = _small_checkpoint()
    weights = SafetensorWeights(
        [],
        tensors=set(wm.keys()),
        tensors_to_file_idx={},
        _st_weight_map={
            name: Buffer.from_dlpack(tensor) for name, tensor in wm.items()
        },
    )
    converted = set(
        convert_language_state_dict_for_module(dict(weights.items()))
    )

    config = _make_small_model_config([DeviceRef.CPU()])
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    with F.lazy(), default_dtype(MAX_DTYPE):
        model = Gemma4TextModel(config, mesh)

    # The eager module tree nests the text model under ``language_model.``
    # (see Gemma4 in gemma4.py); the converter re-adds that prefix.
    expected = {f"language_model.{name}" for name in _param_names(model)}
    assert converted == expected
