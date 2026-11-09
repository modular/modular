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
"""Pins the ``k3_dspark`` config parse against the published checkpoint."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import max.pipelines.architectures.hf_config_shims  # noqa: F401
import pytest
from _dspark_test_data import DRAFT_CONFIG
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import Weights
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.kimi_k3_modulev3.dspark.config import (
    K3_BLOCK_SIZE,
    k3_dspark_draft_width,
    parse_k3_dspark_config,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.model_config import (
    K3DSparkDraftArchConfig,
    _mla_buffer_size,
    _supports_mla_latent_cache,
)
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.speculative.config import SpeculativeConfig
from max.support.math import ceildiv
from transformers import AutoConfig


def test_parses_the_published_checkpoint() -> None:
    config = parse_k3_dspark_config(DRAFT_CONFIG)
    assert config.hidden_size == 7168
    assert config.num_hidden_layers == 5
    assert config.num_attention_heads == 64
    assert config.q_head_dim == 192
    assert config.vocab_size == 163840
    assert config.mask_token_id == 163837
    # Post-layer taps, unshifted (the same convention MAX captures with).
    assert config.target_layer_ids == (2, 23, 47, 71, 89)
    assert config.num_target_layers == 93
    assert config.num_context_features == 5
    assert config.markov_rank == 256
    assert config.rms_norm_eps == 1e-5
    assert config.block_size == K3_BLOCK_SIZE


def test_attention_scale_matches_the_reference() -> None:
    # TorchSpec's K3DSparkMLAAttention: mscale**2 / sqrt(qk_head_dim),
    # with mscale = 0.1 * mscale_all_dim * ln(factor) + 1.
    mscale = 0.1 * 1.0 * math.log(32.0) + 1.0
    expected = (mscale * mscale) / math.sqrt(192)
    config = parse_k3_dspark_config(DRAFT_CONFIG)
    assert config.attention_scale == pytest.approx(expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("markov_head_type", "lstm", "vanilla"),
        ("mla_use_output_gate", True, "output gate"),
        ("mla_use_nope", True, "yarn RoPE"),
        # context_proj consumes the taps at the draft's own width, as many
        # taps as target_layer_ids names; a checkpoint declaring another
        # width or trained tap count would only fail at weight load.
        ("target_hidden_size", 8192, "target_hidden_size"),
        ("num_target_layers", 4, "num_target_layers"),
    ],
)
def test_refuses_unsupported_variants(
    field: str, value: object, message: str
) -> None:
    config = dict(DRAFT_CONFIG)
    config[field] = value
    with pytest.raises(ValueError, match=message):
        parse_k3_dspark_config(config)


def test_declares_no_trained_tap_count() -> None:
    # num_target_layers sizes a weight the module loads, so a checkpoint
    # that does not declare it cannot be checked against the module.
    config = dict(DRAFT_CONFIG)
    del config["num_target_layers"]
    with pytest.raises(ValueError, match="num_target_layers"):
        parse_k3_dspark_config(config)


def test_accepts_an_undeclared_target_hidden_size() -> None:
    # The reference reads the taps' width from target_hidden_size; the
    # module reads it from the draft's own hidden_size, so an undeclared
    # target width defaults to the draft's and parses.
    config = dict(DRAFT_CONFIG)
    del config["target_hidden_size"]
    assert parse_k3_dspark_config(config).hidden_size == 7168


def test_refuses_other_model_types() -> None:
    config = dict(DRAFT_CONFIG, model_type="qwen3_dspark")
    with pytest.raises(ValueError, match="model_type"):
        parse_k3_dspark_config(config)


def test_refuses_other_rope() -> None:
    config = dict(DRAFT_CONFIG)
    config["rope_parameters"] = {"rope_type": "linear", "factor": 4.0}
    with pytest.raises(ValueError, match="yarn"):
        parse_k3_dspark_config(config)


@pytest.mark.parametrize(
    "key",
    ["hidden_size", "target_layer_ids", "target_num_hidden_layers"],
)
def test_a_missing_required_key_raises_value_error(key: str) -> None:
    # The parser promises ValueError for every checkpoint it refuses, so a
    # missing key must not surface as a KeyError from the dict lookup.
    config = dict(DRAFT_CONFIG)
    del config[key]
    with pytest.raises(ValueError, match=key):
        parse_k3_dspark_config(config)


def test_refuses_taps_outside_the_declared_target_depth() -> None:
    # The taps must index a target as deep as the one the checkpoint
    # declares it was trained against; a deeper tap id can only read a
    # different model's residual stream.
    config = dict(DRAFT_CONFIG, target_num_hidden_layers=89)
    with pytest.raises(ValueError, match="target_num_hidden_layers"):
        parse_k3_dspark_config(config)


def _speculative(num_speculative_tokens: int | None) -> SpeculativeConfig:
    return SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=num_speculative_tokens,
    )


def test_draft_width_is_the_block_size() -> None:
    assert (
        k3_dspark_draft_width(_speculative(7), None, DRAFT_CONFIG)
        == K3_BLOCK_SIZE
    )


def test_draft_width_accepts_an_unset_token_count() -> None:
    assert (
        k3_dspark_draft_width(_speculative(None), None, DRAFT_CONFIG)
        == K3_BLOCK_SIZE
    )


def test_draft_width_refuses_a_requested_mismatch() -> None:
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        k3_dspark_draft_width(_speculative(3), None, DRAFT_CONFIG)


def test_draft_width_refuses_a_foreign_block_size() -> None:
    config = dict(DRAFT_CONFIG, block_size=16)
    with pytest.raises(ValueError, match="block_size"):
        k3_dspark_draft_width(_speculative(7), None, config)


def test_parses_the_production_huggingface_config_path() -> None:
    # Production loads the draft checkpoint through transformers' AutoConfig
    # (a PretrainedConfig), so the parser sees to_dict()'s view of the
    # config, not the raw config.json; pin that the round trip keeps every
    # field the drafter reads.
    raw = dict(DRAFT_CONFIG)
    hf_config = AutoConfig.for_model(raw.pop("model_type"), **raw)
    config = parse_k3_dspark_config(hf_config)
    assert config.rope_theta == 50000.0
    assert config.rope_factor == 32.0
    assert config.rope_original_max_position_embeddings == 32768
    assert config.target_layer_ids == (2, 23, 47, 71, 89)
    assert config.num_target_layers == 93
    assert config.attention_scale == pytest.approx(
        parse_k3_dspark_config(DRAFT_CONFIG).attention_scale
    )


def test_rope_theta_prefers_the_top_level_key() -> None:
    # The reference resolves rope_theta top-level first, then the rope
    # dict; a config whose keys disagree must follow the reference, not
    # silently read the dict's.
    config = dict(DRAFT_CONFIG, rope_theta=40000.0)
    assert parse_k3_dspark_config(config).rope_theta == 40000.0


def test_rope_theta_falls_back_to_the_rope_dict() -> None:
    config = dict(DRAFT_CONFIG)
    del config["rope_theta"]
    assert parse_k3_dspark_config(config).rope_theta == 50000.0


def test_rope_theta_defaults_to_the_references_base() -> None:
    # A yarn config that declares no rope_theta anywhere resolves to the
    # reference's 10000.0 base, not to the published checkpoint's value.
    config = dict(DRAFT_CONFIG)
    del config["rope_theta"]
    published = config["rope_parameters"]
    assert isinstance(published, dict)
    rope = dict(published)
    del rope["rope_theta"]
    config["rope_parameters"] = rope
    assert parse_k3_dspark_config(config).rope_theta == 10000.0


def test_falls_back_to_rope_scaling_when_rope_parameters_is_none() -> None:
    # A normalized config can carry rope_parameters: null; the reference
    # falls back to rope_scaling, and so must the parser.
    config = dict(DRAFT_CONFIG)
    rope = config.pop("rope_parameters")
    config["rope_parameters"] = None
    config["rope_scaling"] = rope
    assert parse_k3_dspark_config(config).rope_factor == 32.0


@pytest.mark.parametrize(
    ("num_devices", "cache_dtype"),
    [
        pytest.param(1, DType.bfloat16, id="tp1_bf16"),
        pytest.param(2, DType.bfloat16, id="tp2_bf16"),
        pytest.param(1, DType.float8_e4m3fn, id="tp1_fp8"),
        pytest.param(8, DType.float8_e4m3fn, id="tp8_fp8"),
    ],
)
def test_the_latent_supports_gate_passes_the_published_checkpoint(
    num_devices: int, cache_dtype: DType
) -> None:
    """The published geometry serves through the fused MLA kernels."""
    draft = parse_k3_dspark_config(DRAFT_CONFIG)
    assert _supports_mla_latent_cache(draft, cache_dtype, num_devices)


@pytest.mark.parametrize(
    ("override", "num_devices", "cache_dtype"),
    [
        # A latent the fused kernels do not cache: they assert 576 (or
        # 512 NoPE), and this drafter always ropes.
        pytest.param(
            {"kv_lora_rank": 24}, 1, DType.bfloat16, id="narrow_latent"
        ),
        # A head count the mesh cannot split into whole heads per rank.
        pytest.param(
            {"num_attention_heads": 5}, 2, DType.bfloat16, id="odd_heads_tp2"
        ),
        # A cache dtype the fused MLA kernels do not take.
        pytest.param(
            {"num_attention_heads": 64},
            1,
            DType.float32,
            id="fp32_cache",
        ),
    ],
)
def test_the_latent_supports_gate_refuses_an_unservable_draft(
    override: dict[str, object], num_devices: int, cache_dtype: DType
) -> None:
    """Every clause of the gate is a fused-kernel constraint, so none may
    be relaxed to keep a draft the kernels would refuse at runtime."""
    config = dict(DRAFT_CONFIG)
    config.update(override)
    draft = parse_k3_dspark_config(config)
    assert not _supports_mla_latent_cache(draft, cache_dtype, num_devices)


def _pipeline_config() -> PipelineConfig:
    """A pipeline config as ``construct_kv_params`` reads it."""
    return cast(
        PipelineConfig,
        SimpleNamespace(
            speculative=SpeculativeConfig(
                speculative_method="dspark", num_speculative_tokens=7
            ),
            model=SimpleNamespace(data_parallel_degree=1),
        ),
    )


def test_construct_kv_params_refuses_an_unservable_draft() -> None:
    """The drafter has no decompressed path to fall back to, so a draft the
    fused MLA kernels cannot serve is a config-time error, never a slower
    route or a runtime crash."""
    config = dict(DRAFT_CONFIG, kv_lora_rank=24)
    with pytest.raises(ValueError, match="576"):
        K3DSparkDraftArchConfig.construct_kv_params(
            config,
            _pipeline_config(),
            [DeviceRef.CPU()],
            KVCacheConfig(),
            DType.bfloat16,
        )


def test_construct_kv_params_builds_the_latent_leaf() -> None:
    """The happy path: the drafter's leaf is the compressed latent, at the
    target's own cache dtype, one KV head behind the full query-head count
    -- never a quantized leaf, which would make both fused arms
    unreachable."""
    params = K3DSparkDraftArchConfig.construct_kv_params(
        DRAFT_CONFIG,
        _pipeline_config(),
        [DeviceRef.CPU()],
        KVCacheConfig(),
        DType.float8_e4m3fn,
    )
    assert isinstance(params, MLAKVCacheParams)
    draft = parse_k3_dspark_config(DRAFT_CONFIG)
    assert params.head_dim == draft.kv_lora_rank + draft.qk_rope_head_dim
    assert params.num_q_heads == draft.num_attention_heads
    assert params.dtype == DType.float8_e4m3fn
    assert params.page_size == 128
    # One latent per token behind one KV head, whatever the query-head
    # count: the compressed leaf, not a per-head one.
    assert params.kv_dim == 1
    assert params.n_kv_heads_per_device == 1


# The load context's plan: one request's length bound and a batch-slot
# count the user never set but planning resolved.
_PLAN_MAX_SEQ_LEN = 512
_PLANNED_BATCH_SIZE = 64


def _draft_kv_params() -> MLAKVCacheParams:
    """The drafter's latent leaf, as the pairing hands it to the draft."""
    params = K3DSparkDraftArchConfig.construct_kv_params(
        DRAFT_CONFIG,
        _pipeline_config(),
        [DeviceRef.CPU()],
        KVCacheConfig(),
        DType.bfloat16,
    )
    assert isinstance(params, MLAKVCacheParams)
    return params


def _load_context(
    *, max_batch_size: int | None, planned_max_batch_size: int | None
) -> ModelLoadContext:
    """The load context the pairing hands the drafter: the runtime the
    user configured, over the memory plan that sized the pool."""
    return ModelLoadContext(
        pipeline_config=cast(
            PipelineConfig,
            SimpleNamespace(
                runtime=PipelineRuntimeConfig(max_batch_size=max_batch_size)
            ),
        ),
        # `weights` is only read by `checkpoint_weights`, which these tests
        # never call; the context field just has to exist.
        weights=cast(Weights, None),
        adapter=None,
        memory_plan=MemoryPlan(
            planned_max_batch_size=cast(int, planned_max_batch_size),
            footprint=0,
            planned_max_length=_PLAN_MAX_SEQ_LEN,
            planned_max_batch_total_tokens=None,
        ),
        return_logits=ReturnLogits.LAST_TOKEN,
        return_hidden_states=ReturnHiddenStates.NONE,
    )


def test_the_drafter_plan_cap_sizes_every_batch_slot() -> None:
    """The cap is priced per batch slot, off the same resolution the
    target's own cap uses: the user's ``--max-batch-size``, or what memory
    planning inferred when it was left unset. Sizing it for one request
    would silently truncate every step past the first.
    """
    kv_params = _draft_kv_params()
    assert kv_params.page_size is not None
    context = _load_context(
        max_batch_size=None, planned_max_batch_size=_PLANNED_BATCH_SIZE
    )

    cap = _mla_buffer_size(
        context, parse_k3_dspark_config(DRAFT_CONFIG), kv_params
    )

    # The block call's window is ``2 * block_size + 1`` rows past the
    # length bound (the plan clamps to the cap and drops what spills at
    # max_chunks=1), and every planned slot holds one.
    table_need = (
        ceildiv(_PLAN_MAX_SEQ_LEN + 2 * K3_BLOCK_SIZE + 1, kv_params.page_size)
        * kv_params.page_size
    )
    assert cap == _PLANNED_BATCH_SIZE * table_need


def test_the_drafter_plan_cap_requires_a_resolved_batch_size() -> None:
    """The TG worst case is priced per batch slot, so a plan that resolved
    none fails fast instead of sizing the cap for one request and
    silently truncating every step past the first.
    """
    context = _load_context(max_batch_size=None, planned_max_batch_size=None)

    with pytest.raises(ValueError, match="max_batch_size is unresolved"):
        _mla_buffer_size(
            context, parse_k3_dspark_config(DRAFT_CONFIG), _draft_kv_params()
        )


def test_mscale_all_dim_defaults_to_torchspecs() -> None:
    # TorchSpec's yarn_defaults leave mscale_all_dim at 0.0 when a config
    # omits it: the correction then rides the cos/sin tables and the
    # attention scale falls to 1/sqrt(q_head_dim), not mscale^2/sqrt.
    config = dict(DRAFT_CONFIG)
    published = config["rope_parameters"]
    assert isinstance(published, dict)
    rope = dict(published)
    del rope["mscale_all_dim"]
    config["rope_parameters"] = rope
    parsed = parse_k3_dspark_config(config)
    assert parsed.attention_scale == pytest.approx(
        1.0 / math.sqrt(parsed.q_head_dim)
    )
