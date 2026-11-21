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
"""``KimiK3Config`` (ModuleV3): the cache tree and the derived model shape.

CPU-only: everything here is config resolution, not graph construction.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.driver import CPU, DeviceSpec
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
)
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.kimi_k3_modulev3.layers.vision.transformer import (
    KimiK3VisionEncoder,
)
from max.pipelines.architectures.kimi_k3_modulev3.model_config import (
    KimiK3Config,
)
from max.pipelines.architectures.kimi_k3_modulev3.state_cache import (
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    STATE_CACHE_KEY,
    attn_cache,
    kda_state_regions,
)
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.lib.module_registration import _vision_enabled
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.pipelines.speculative import SpeculativeConfig
from max.tensor.sharding import DeviceMesh

_HIDDEN_SIZE = 1024
_NUM_ATTENTION_HEADS = 8
_NUM_HIDDEN_LAYERS = 8
_QK_NOPE_HEAD_DIM = 64
_QK_ROPE_HEAD_DIM = 32
_V_HEAD_DIM = 64
# 1-INDEXED in the checkpoint's `linear_attn_config`; layers 4 and 8 are MLA.
_FULL_ATTN_LAYERS = [4, 8]
_KDA_LAYERS = [1, 2, 3, 5, 6, 7]
_KDA_NUM_HEADS = 8
_KDA_HEAD_DIM = 32
_KDA_CONV_KERNEL_SIZE = 4
_NUM_EXPERTS = 8
_TOP_K = 2
_RUD_EXPERT_LATENT = 512

_EXPECTED_HEAD_DIM = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM
_EXPECTED_NUM_FULL_ATTN_LAYERS = len(_FULL_ATTN_LAYERS)

_MAX_BATCH_SIZE = 4
_MAX_BATCH_INPUT_TOKENS = 8192
_MAX_LENGTH = 512

# Per KDA layer at this fixture's 8 heads x 32 head_dim x 4-tap conv: a
# `3 * heads * head_dim`-wide conv window over `kernel - 1` taps in bfloat16,
# plus FP32 `heads x head_dim x head_dim` recurrent state.
_CONV_BYTES_PER_LAYER = (
    (3 * _KDA_NUM_HEADS * _KDA_HEAD_DIM) * (_KDA_CONV_KERNEL_SIZE - 1) * 2
)
_REC_BYTES_PER_LAYER = _KDA_NUM_HEADS * _KDA_HEAD_DIM * _KDA_HEAD_DIM * 4
_STATE_BYTES_PER_REQUEST = len(_KDA_LAYERS) * (
    _CONV_BYTES_PER_LAYER + _REC_BYTES_PER_LAYER
)

# Real K3's latent width is the only one the fused MLA kernels take; this
# fixture's 128 + 32 misses it on purpose so the default tests exercise the
# decompressed fallback, matching `kimi_k3`'s own `test_kv_params.py` fixture.
_REAL_KV_LORA_RANK = 512
_REAL_QK_ROPE_HEAD_DIM = 64
_REAL_NUM_ATTENTION_HEADS = 96
_TP_DEGREE = 8


class _FakeTextConfig(SimpleNamespace):
    """Minimal stand-in for the HuggingFace text sub-config.

    Needs `to_dict()` (unlike a plain `SimpleNamespace`) because
    `KimiK3Config.construct_kv_params`/`initialize` build a
    `KimiK3ModelConfig` from the parsed `config.json` dict shape, not from
    attribute access.
    """

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _text_config(
    *,
    kv_lora_rank: int = 128,
    qk_rope_head_dim: int = _QK_ROPE_HEAD_DIM,
    num_attention_heads: int = _NUM_ATTENTION_HEADS,
) -> _FakeTextConfig:
    return _FakeTextConfig(
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=_NUM_HIDDEN_LAYERS,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=_V_HEAD_DIM,
        q_lora_rank=256,
        kv_lora_rank=kv_lora_rank,
        linear_attn_config={
            "full_attn_layers": _FULL_ATTN_LAYERS,
            "kda_layers": _KDA_LAYERS,
            "num_heads": _KDA_NUM_HEADS,
            "head_dim": _KDA_HEAD_DIM,
            "short_conv_kernel_size": _KDA_CONV_KERNEL_SIZE,
        },
        intermediate_size=2048,
        vocab_size=163840,
        rms_norm_eps=1e-5,
        max_position_embeddings=4096,
        attn_res_block_size=4,
        first_k_dense_replace=1,
        routed_expert_hidden_size=512,
        moe_intermediate_size=256,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )


@pytest.fixture
def configs() -> tuple[Any, Any, Any]:
    """Returns ``(huggingface_config, pipeline_config, model_config)``."""
    huggingface_config = SimpleNamespace(text_config=_text_config())
    model_config = SimpleNamespace(
        kv_cache=KVCacheConfig(),
        quantization_encoding="bfloat16",
        graph_quantization_encoding=None,
        # The name carries the encoding on purpose; encoding resolution reads
        # the file name and this namespace has no repo path to fall back to.
        weight_path=[Path("model-bfloat16.safetensors")],
        device_specs=[DeviceSpec(id=0, device_type="cpu")],
        default_device_spec=DeviceSpec(id=0, device_type="cpu"),
        data_parallel_degree=1,
        huggingface_config=huggingface_config,
        max_length=_MAX_LENGTH,
        model_path="kimi-k3-modulev3-tiny",
    )
    pipeline_config = SimpleNamespace(
        model=model_config,
        speculative=None,
        runtime=PipelineRuntimeConfig(
            max_batch_size=_MAX_BATCH_SIZE,
            max_batch_input_tokens=_MAX_BATCH_INPUT_TOKENS,
            max_batch_total_tokens=None,
            ep_size=1,
            ep_use_allreduce=False,
        ),
    )
    return huggingface_config, pipeline_config, model_config


def test_layer_split_matches_the_checkpoint_partition(
    configs: tuple[Any, Any, Any],
) -> None:
    _huggingface_config, pipeline_config, _ = configs

    config = KimiK3Config.initialize(
        pipeline_config, pipeline_config.model, max_seq_len=_MAX_LENGTH
    )

    assert config.model.kda_layer_indices == sorted(i - 1 for i in _KDA_LAYERS)
    assert config.model.full_attn_layer_indices == sorted(
        i - 1 for i in _FULL_ATTN_LAYERS
    )
    assert config.model.num_hidden_layers == _NUM_HIDDEN_LAYERS


def test_routed_experts_sized_for_the_batch_token_budget(
    configs: tuple[Any, Any, Any],
) -> None:
    """A budget past the layer's 8192-row fallback reaches the routed
    experts, which size their GEMM grid and MegaFFN workspace from it."""
    _huggingface_config, pipeline_config, _ = configs
    pipeline_config.runtime = pipeline_config.runtime.model_copy(
        update={"max_batch_input_tokens": 16384, "enable_chunked_prefill": True}
    )

    config = KimiK3Config.initialize(
        pipeline_config, pipeline_config.model, max_seq_len=_MAX_LENGTH
    )

    assert config.max_tokens_per_expert == config.routed_expert_row_bound(
        pipeline_config
    )
    assert config.max_tokens_per_expert >= 16384


def test_cache_is_the_attention_leaf_beside_the_kda_state(
    configs: tuple[Any, Any, Any],
) -> None:
    _, pipeline_config, _ = configs

    config = KimiK3Config.initialize(
        pipeline_config, pipeline_config.model, max_seq_len=_MAX_LENGTH
    )

    assert isinstance(config.kv_params, MultiKVCacheParams)
    assert list(config.kv_params.children) == ["attn", "state"]
    assert isinstance(config.attn_kv_params, MHAKVCacheParams)
    assert config.attn_kv_params.head_dim == _EXPECTED_HEAD_DIM
    assert config.attn_kv_params.num_layers == _EXPECTED_NUM_FULL_ATTN_LAYERS
    assert not config.mla_latent_cache


def test_stored_kv_params_match_construct_kv_params(
    configs: tuple[Any, Any, Any],
) -> None:
    """The field the graph body would read must equal what sizes the cache."""
    huggingface_config, pipeline_config, model_config = configs

    config = KimiK3Config.initialize(
        pipeline_config, model_config, max_seq_len=_MAX_LENGTH
    )
    constructed = KimiK3Config.construct_kv_params(
        huggingface_config=huggingface_config,
        pipeline_config=pipeline_config,
        devices=[DeviceRef.CPU()],
        kv_cache_config=model_config.kv_cache,
        cache_dtype=DType.bfloat16,
    )

    assert (
        config.kv_params.flattened_kv_inputs()[0]
        == constructed.flattened_kv_inputs()[0]
    )


def _state_leaf(config: KimiK3Config) -> RecurrentStateParams:
    """The KDA state child of the ``{attn, state}`` cache tree."""
    assert isinstance(config.kv_params, MultiKVCacheParams)
    state = config.kv_params.children[STATE_CACHE_KEY]
    assert isinstance(state, RecurrentStateParams)
    return state


def test_state_bytes_match_the_declared_pool_shapes(
    configs: tuple[Any, Any, Any],
) -> None:
    """The declared leaf and the priced figure are the same number.

    ``kda_state_regions`` sizes the pool the cache manager allocates while
    ``per_request_state_bytes`` is what memory planning charges, and nothing
    else reads both -- a dtype or head-split change applied to one of them
    alone would otherwise pass every test here.
    """
    _, pipeline_config, _ = configs

    config = KimiK3Config.initialize(
        pipeline_config, pipeline_config.model, max_seq_len=_MAX_LENGTH
    )

    assert config.ssm_dtype == DType.bfloat16
    assert config.recurrent_state_dtype == DType.float32
    assert config.per_request_state_bytes() == _STATE_BYTES_PER_REQUEST
    assert config.per_request_state_bytes() > 0
    assert _state_leaf(config).bytes_per_state == _STATE_BYTES_PER_REQUEST
    assert _state_leaf(config).bytes_per_state == (
        config.per_request_state_bytes()
    )


def test_state_bytes_follow_the_head_split_across_devices(
    configs: tuple[Any, Any, Any],
) -> None:
    """Each device's share of the state is ``1 / n_devices`` of the whole.

    ``per_request_state_bytes`` is per device, while the cache's declared
    state sums over the devices, so the total is unchanged by the split -- a
    figure that did not follow the head split would under-report the pool by
    the device count.
    """
    _huggingface_config, pipeline_config, model_config = configs
    model_config.device_specs = [
        DeviceSpec(id=i, device_type="gpu") for i in range(2)
    ]

    config = KimiK3Config.initialize(
        pipeline_config, model_config, max_seq_len=_MAX_LENGTH
    )

    assert config.tensor_parallel_degree == 2
    assert config.per_request_state_bytes() == _STATE_BYTES_PER_REQUEST // 2
    assert (
        len(config.devices) * config.per_request_state_bytes()
        == _STATE_BYTES_PER_REQUEST
    )
    # The declared leaf is the per-device shard too, not the whole state.
    assert _state_leaf(config).bytes_per_state == (
        config.per_request_state_bytes()
    )
    assert (
        config.tensor_parallel_degree * _state_leaf(config).bytes_per_state
        == _STATE_BYTES_PER_REQUEST
    )


def test_declared_state_regions_are_one_device_shard(
    configs: tuple[Any, Any, Any],
) -> None:
    """Each region's row is a device's slice of the channels and heads.

    The conv window splits on its fused q/k/v width and the recurrent state
    on its head count, both by the TP degree -- a region declared at full
    width would allocate the whole pool on every device.
    """
    _, pipeline_config, model_config = configs
    model_config.device_specs = [
        DeviceSpec(id=i, device_type="gpu") for i in range(2)
    ]

    config = KimiK3Config.initialize(
        pipeline_config, model_config, max_seq_len=_MAX_LENGTH
    )
    regions = _state_leaf(config).regions

    assert [region.leaf_id for region in regions] == [
        CONV_LEAF_ID,
        RECURRENT_LEAF_ID,
    ]
    conv, recurrent = regions
    assert conv.num_layers == len(_KDA_LAYERS)
    assert conv.row_shape == (
        3 * _KDA_NUM_HEADS * _KDA_HEAD_DIM // 2,
        _KDA_CONV_KERNEL_SIZE - 1,
    )
    assert conv.dtype == DType.bfloat16
    assert recurrent.num_layers == len(_KDA_LAYERS)
    assert recurrent.row_shape == (
        _KDA_NUM_HEADS // 2,
        _KDA_HEAD_DIM,
        _KDA_HEAD_DIM,
    )
    assert recurrent.dtype == DType.float32


def test_a_tp_degree_the_state_cannot_split_by_is_refused() -> None:
    """A degree that leaves a rank a fraction of a head is refused here.

    ``initialize`` reaches it only through the attention leaf's own split
    check, so the state's refusal is pinned against ``kda_state_regions``
    directly -- otherwise nothing observes it.
    """
    with pytest.raises(ValueError, match="divide by the tensor-parallel"):
        kda_state_regions(
            num_kda_layers=len(_KDA_LAYERS),
            conv_dim=3 * _KDA_NUM_HEADS * _KDA_HEAD_DIM,
            conv_kernel_size=_KDA_CONV_KERNEL_SIZE,
            num_heads=_KDA_NUM_HEADS,
            head_dim=_KDA_HEAD_DIM,
            conv_dtype=DType.bfloat16,
            recurrent_dtype=DType.float32,
            num_devices=3,
        )


def _state_dtypes(
    configs: tuple[Any, Any, Any],
    *,
    encoding: str = "bfloat16",
    state_pool_dtype: str | None = None,
) -> tuple[DType, DType]:
    """Resolves ``(conv, recurrent)`` for one checkpoint and pool setting."""
    _, pipeline_config, model_config = configs
    model_config.quantization_encoding = encoding
    model_config.weight_path = [Path(f"model-{encoding}.safetensors")]
    return KimiK3Config.declared_state_dtypes(
        pipeline_config,
        KVCacheConfig(state_pool_dtype=state_pool_dtype),
    )


def test_the_conv_leaf_follows_the_resolved_encoding(
    configs: tuple[Any, Any, Any],
) -> None:
    """The conv history is kept at the model's own precision.

    The leaf is declared before a config exists and the budget is priced from
    the resolved dtype, so reading the two off different encodings makes
    ``check_state_budget`` fail at load on a float32 checkpoint.
    """
    assert _state_dtypes(configs)[0] == DType.bfloat16
    assert _state_dtypes(configs, encoding="float32")[0] == DType.float32


def test_a_pool_override_moves_the_recurrent_dtype_alone(
    configs: tuple[Any, Any, Any],
) -> None:
    """``--state-pool-dtype`` is the recurrent pool's knob only.

    The conv leaf follows the model dtype either way; moving it with the
    override would round the conv history differently from the graph the
    checkpoint was trained against.
    """
    assert _state_dtypes(configs, state_pool_dtype="bfloat16") == (
        DType.bfloat16,
        DType.bfloat16,
    )
    assert _state_dtypes(
        configs, encoding="float32", state_pool_dtype="bfloat16"
    ) == (DType.float32, DType.bfloat16)
    assert _state_dtypes(configs, state_pool_dtype="float32") == (
        DType.bfloat16,
        DType.float32,
    )


def test_an_unsupported_pool_dtype_is_rejected(
    configs: tuple[Any, Any, Any],
) -> None:
    """A pool dtype the kernels cannot bind is refused here rather than at
    the first launch."""
    with pytest.raises(ValueError, match="state_pool_dtype"):
        _state_dtypes(configs, state_pool_dtype="float8_e4m3fn")


def test_latent_cache_reachable_at_real_k3_shape_and_tp(
    configs: tuple[Any, Any, Any],
) -> None:
    """At real widths and TP8, the fused MLA kernels take the latent cache."""
    huggingface_config, pipeline_config, model_config = configs
    text_config = huggingface_config.text_config
    text_config.kv_lora_rank = _REAL_KV_LORA_RANK
    text_config.qk_rope_head_dim = _REAL_QK_ROPE_HEAD_DIM
    text_config.num_attention_heads = _REAL_NUM_ATTENTION_HEADS
    model_config.device_specs = [
        DeviceSpec(id=i, device_type="gpu") for i in range(_TP_DEGREE)
    ]

    config = KimiK3Config.initialize(
        pipeline_config, model_config, max_seq_len=_MAX_LENGTH
    )

    assert config.mla_latent_cache
    assert isinstance(config.attn_kv_params, MLAKVCacheParams)
    assert (
        config.attn_kv_params.head_dim
        == _REAL_KV_LORA_RANK + _REAL_QK_ROPE_HEAD_DIM
    )


def test_fp8_kv_cache_format_keeps_the_latent_path(
    configs: tuple[Any, Any, Any],
) -> None:
    """``--kv-cache-format float8_e4m3fn`` must halve the latent, not drop it.

    ``supports_mla_latent_cache`` admits exactly two cache dtypes, so an
    unrecognised one silently falls back to the decompressed cache at 64x the
    bytes -- a memory regression wearing the costume of a memory saving. The
    fp8 latent is also SCALE-FREE: ``construct_kv_params`` passes no
    ``kvcache_quant_config``, and the MLA kernels' fp8 arms are gated on
    ``not CacheType.quantization_enabled``, so a scale config here would route
    to a quantized cache those arms do not serve.
    """
    huggingface_config, pipeline_config, model_config = configs
    text_config = huggingface_config.text_config
    text_config.kv_lora_rank = _REAL_KV_LORA_RANK
    text_config.qk_rope_head_dim = _REAL_QK_ROPE_HEAD_DIM
    text_config.num_attention_heads = _REAL_NUM_ATTENTION_HEADS
    devices = [DeviceRef.GPU(i) for i in range(_TP_DEGREE)]

    def _params(cache_dtype: DType) -> KVCacheParams:
        return attn_cache(
            KimiK3Config.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=devices,
                kv_cache_config=model_config.kv_cache,
                cache_dtype=cache_dtype,
            )
        )

    fp8 = _params(DType.float8_e4m3fn)
    assert isinstance(fp8, MLAKVCacheParams)
    assert fp8.head_dim == _REAL_KV_LORA_RANK + _REAL_QK_ROPE_HEAD_DIM
    assert fp8.is_fp8_kv_dtype
    assert not fp8.quantized_kv_cache
    assert fp8.row_bytes * 2 == _params(DType.bfloat16).row_bytes


def test_a_non_dspark_speculative_config_sizes_the_plain_leaf(
    configs: tuple[Any, Any, Any],
) -> None:
    """An eagle config keeps initializing, over a decode-sized leaf.

    Only DSpark block-drafts through this tower; any other speculative
    method runs the plain cache sizing rather than failing at KV-cache
    time with a message about sizing. The leaf's draft width stays zero
    and its per-step accounting stays one draft row per step.
    """
    _, pipeline_config, model_config = configs
    pipeline_config.speculative = SpeculativeConfig(speculative_method="eagle")

    config = KimiK3Config.initialize(
        pipeline_config, model_config, max_seq_len=_MAX_LENGTH
    )

    assert config.attn_kv_params.num_draft_tokens == 0
    assert config.attn_kv_params.num_draft_tokens_per_step == 1
    assert config.attn_kv_params.speculative_method == "eagle"


def test_max_seq_len_uses_the_requested_length_within_bound(
    configs: tuple[Any, Any, Any],
) -> None:
    huggingface_config, _, model_config = configs

    resolved = KimiK3Config.calculate_max_seq_len(
        huggingface_config, model_config
    )

    assert resolved == _MAX_LENGTH


def test_max_seq_len_rejects_a_request_above_the_model_limit(
    configs: tuple[Any, Any, Any],
) -> None:
    huggingface_config, _, model_config = configs
    model_config.max_length = 8192

    with pytest.raises(ValueError, match="exceeds"):
        KimiK3Config.calculate_max_seq_len(huggingface_config, model_config)


def test_validate_tensor_parallel_split_rejects_indivisible_degree(
    configs: tuple[Any, Any, Any],
) -> None:
    """Unit-tests the split check directly against a mismatched field.

    Not exercised through `initialize()`: the KV cache's own
    `MHAKVCacheParams.__post_init__` already rejects a `num_attention_heads`
    split indivisible by the TP degree before this method would even run, so
    routing through a real `num_attention_heads` mismatch would only ever
    observe the cache's error, never this one.
    """
    _, pipeline_config, model_config = configs

    config = KimiK3Config.initialize(
        pipeline_config, model_config, max_seq_len=_MAX_LENGTH
    )
    # `vocab_size` divides every TP degree this fixture's other fields do
    # (all powers of two); make it the one field that does not.
    config.model.vocab_size += 1

    with pytest.raises(ValueError, match=r"vocab_size.*does not divide by it"):
        KimiK3Config(
            **{**vars(config), "devices": config.devices * 2}
        ).validate_tensor_parallel_split()


def test_data_parallelism_is_rejected(configs: tuple[Any, Any, Any]) -> None:
    _, pipeline_config, model_config = configs
    model_config.data_parallel_degree = 2

    with pytest.raises(ValueError, match="Data parallelism"):
        KimiK3Config.initialize(
            pipeline_config, model_config, max_seq_len=_MAX_LENGTH
        )


def _row_bound(
    config: KimiK3Config,
    *,
    speculative: Any = None,
    max_batch_size: int | None = _MAX_BATCH_SIZE,
    enable_chunked_prefill: bool = True,
    max_batch_total_tokens: int | None = None,
) -> int:
    """Runs the bound over a stand-in carrying only the fields it reads."""
    pipeline_config = SimpleNamespace(
        speculative=speculative,
        runtime=PipelineRuntimeConfig(
            max_batch_size=max_batch_size,
            max_batch_input_tokens=_MAX_BATCH_INPUT_TOKENS,
            max_batch_total_tokens=max_batch_total_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
        ),
    )
    return config.routed_expert_row_bound(cast(PipelineConfig, pipeline_config))


def _bound_config(configs: tuple[Any, Any, Any]) -> KimiK3Config:
    _, pipeline_config, _ = configs
    return KimiK3Config.initialize(
        pipeline_config, pipeline_config.model, max_seq_len=_MAX_LENGTH
    )


def test_the_default_row_bound_is_the_input_budget(
    configs: tuple[Any, Any, Any],
) -> None:
    """Chunked prefill makes the input budget a hard cap."""
    assert _row_bound(_bound_config(configs)) == _MAX_BATCH_INPUT_TOKENS


def test_a_verify_step_adds_its_draft_rows(
    configs: tuple[Any, Any, Any],
) -> None:
    """Each request can carry ``num_speculative_tokens`` rows the budget never
    charged, on top of a full prefill chunk in a mixed batch."""
    bound = _row_bound(
        _bound_config(configs),
        speculative=SimpleNamespace(num_speculative_tokens=8),
    )

    assert bound == _MAX_BATCH_INPUT_TOKENS + 8 * _MAX_BATCH_SIZE


def test_an_unset_batch_size_falls_back_to_the_token_budget(
    configs: tuple[Any, Any, Any],
) -> None:
    """The scheduler holds ``max_batch_size`` at or below the input budget."""
    bound = _row_bound(
        _bound_config(configs),
        speculative=SimpleNamespace(num_speculative_tokens=2),
        max_batch_size=None,
    )

    assert bound == 3 * _MAX_BATCH_INPUT_TOKENS


def test_speculation_disabled_or_unresolved_adds_nothing(
    configs: tuple[Any, Any, Any],
) -> None:
    """Both read as zero, which is how the scheduler reads them too."""
    config = _bound_config(configs)

    assert (
        _row_bound(
            config, speculative=SimpleNamespace(num_speculative_tokens=0)
        )
        == _MAX_BATCH_INPUT_TOKENS
    )
    assert (
        _row_bound(
            config, speculative=SimpleNamespace(num_speculative_tokens=None)
        )
        == _MAX_BATCH_INPUT_TOKENS
    )


def test_without_chunking_the_total_context_budget_bounds(
    configs: tuple[Any, Any, Any],
) -> None:
    """An unchunked batch is admitted whole, so the context budget is what it
    can hold."""
    config = _bound_config(configs)
    total = 4 * _MAX_BATCH_INPUT_TOKENS

    assert (
        _row_bound(
            config,
            enable_chunked_prefill=False,
            max_batch_total_tokens=total,
        )
        == total
    )
    assert (
        _row_bound(
            config,
            speculative=SimpleNamespace(num_speculative_tokens=8),
            enable_chunked_prefill=False,
            max_batch_total_tokens=total,
        )
        == total + 8 * _MAX_BATCH_SIZE
    )


def test_without_chunking_or_a_total_budget_the_batch_size_bounds(
    configs: tuple[Any, Any, Any],
) -> None:
    """No context budget leaves a full batch's worth of sequences: here
    ``max_seq_len * max_batch_size`` is past the input budget, so it is what
    binds."""
    batch_size = 32
    assert batch_size * _MAX_LENGTH > _MAX_BATCH_INPUT_TOKENS

    assert _row_bound(
        _bound_config(configs),
        enable_chunked_prefill=False,
        max_batch_size=batch_size,
    ) == (_MAX_LENGTH * batch_size)


def test_nothing_bounding_a_prefill_batch_is_refused(
    configs: tuple[Any, Any, Any],
) -> None:
    """A bound the routed matmul sizes itself from must never be a guess."""
    with pytest.raises(ValueError, match="nothing bounds a context-encoding"):
        _row_bound(
            _bound_config(configs),
            enable_chunked_prefill=False,
            max_batch_size=None,
        )


def _ep_runtime(
    *,
    ep_size: int,
    n_devices: int,
    use_allreduce: bool = False,
    max_batch_input_tokens: int = _MAX_BATCH_INPUT_TOKENS,
) -> ModuleRuntime:
    """A deployment over a simulated mesh.

    The exchange's sizing reads the mesh's device count alone, so the mesh is
    host devices: this target touches no accelerator.
    """
    return ModuleRuntime(
        mesh=DeviceMesh(
            tuple(CPU() for _ in range(n_devices)),
            (n_devices,),
            ("ep",),
        ),
        data_parallel_degree=1,
        max_batch_input_tokens=max_batch_input_tokens,
        max_batch_total_tokens=None,
        ep_size=ep_size,
        ep_use_allreduce=use_allreduce,
        max_batch_size=_MAX_BATCH_SIZE,
    )


def test_ep_buffers_hold_one_device_shard(
    configs: tuple[Any, Any, Any],
) -> None:
    """The dispatch sends one device's shard of the batch.

    ``grouped_matmul_bounds`` multiplies this back out by the rank count, so
    the grouped matmul's M grid still covers the whole batch: the exchange
    shards the tokens, and what one device receives is bounded by the batch.
    """
    config = _bound_config(configs)

    for ep_size in (2, 4, 8):
        ep = config.expert_parallel_config(
            _ep_runtime(ep_size=ep_size, n_devices=ep_size)
        )
        assert ep.max_tokens_per_rank == _MAX_BATCH_INPUT_TOKENS // ep_size
        assert ep.n_gpus_per_node == ep_size
        assert ep.n_nodes == 1


def test_ep_buffers_follow_a_raised_token_budget(
    configs: tuple[Any, Any, Any],
) -> None:
    """A configured budget above the default has to reach the buffers."""
    config = _bound_config(configs)
    raised = 4 * _MAX_BATCH_INPUT_TOKENS

    ep = config.expert_parallel_config(
        _ep_runtime(
            ep_size=2,
            n_devices=2,
            max_batch_input_tokens=raised,
        )
    )

    assert ep.max_tokens_per_rank == raised // 2


def test_the_allreduce_variant_holds_the_whole_batch(
    configs: tuple[Any, Any, Any],
) -> None:
    """An allreduce combine crosses no reduce-scatter, so no rank holds a
    shard and the buffer has to take the whole batch."""
    ep = _bound_config(configs).expert_parallel_config(
        _ep_runtime(ep_size=2, n_devices=2, use_allreduce=True)
    )

    assert ep.use_allreduce
    assert ep.max_tokens_per_rank == _MAX_BATCH_INPUT_TOKENS


def test_dispatch_width_is_the_latent(configs: tuple[Any, Any, Any]) -> None:
    """K3's experts read the latent, so dispatching at model width would hand
    the grouped matmuls the wrong width."""
    ep = _bound_config(configs).expert_parallel_config(
        _ep_runtime(ep_size=2, n_devices=2)
    )

    assert ep.hidden_size == _RUD_EXPERT_LATENT
    assert ep.n_experts == _NUM_EXPERTS
    assert ep.top_k == _TOP_K
    assert ep.dispatch_dtype == DType.bfloat16
    assert ep.combine_dtype == DType.bfloat16


def test_a_multi_node_deployment_is_refused_without_an_mpi_environment(
    configs: tuple[Any, Any, Any],
) -> None:
    """``ep_size`` is the whole deployment, so a mesh smaller than it implies
    several nodes -- which the exchange cannot start without NVSHMEM."""
    with pytest.raises(ValueError, match="multi-node EP"):
        _bound_config(configs).expert_parallel_config(
            _ep_runtime(ep_size=8, n_devices=4)
        )


def test_mxfp4_experts_are_refused_under_expert_parallelism(
    configs: tuple[Any, Any, Any],
) -> None:
    """The quantized dispatch has no expert-parallel transport yet, and a
    silently bf16-dispatching MXFP4 expert set would hand the grouped matmul
    packed weights."""
    config = replace(
        _bound_config(configs),
        quant_config=QuantConfig.mxfp4(mlp_quantized_layers=[0]),
    )

    with pytest.raises(NotImplementedError, match="MXFP4"):
        config.expert_parallel_config(_ep_runtime(ep_size=2, n_devices=2))


def test_a_speculative_pairing_is_refused_under_expert_parallelism(
    configs: tuple[Any, Any, Any],
) -> None:
    """The DSpark draft pairs with the TP trunk, and the fused graph takes no
    exchange buffers."""
    config = replace(_bound_config(configs), spec_num_draft_tokens=8)

    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        config.expert_parallel_config(_ep_runtime(ep_size=2, n_devices=2))


def test_text_only_checkpoint_disables_vision_role(
    configs: tuple[Any, Any, Any],
) -> None:
    _, pipeline_config, _ = configs
    config = KimiK3Config.initialize(
        pipeline_config, pipeline_config.model, max_seq_len=_MAX_LENGTH
    )
    assert config.vision_config is None
    assert not _vision_enabled(KimiK3VisionEncoder, config)


@pytest.mark.parametrize("dspark", [False, True])
def test_vision_role_is_disabled_only_for_the_dspark_pairing(
    configs: tuple[Any, Any, Any], dspark: bool
) -> None:
    huggingface_config, pipeline_config, model_config = configs
    huggingface_config.vision_config = SimpleNamespace(
        patch_size=14,
        vt_hidden_size=64,
        qkv_hidden_size=64,
        vt_intermediate_size=128,
        vt_num_attention_heads=1,
        vt_num_hidden_layers=1,
        init_pos_emb_height=4,
        init_pos_emb_width=4,
        init_pos_emb_time=4,
        merge_kernel_size=[2, 2],
        mm_hidden_size=64,
        text_hidden_size=_HIDDEN_SIZE,
        projector_ln_eps=1e-5,
    )
    if dspark:
        pipeline_config.speculative = SpeculativeConfig(
            speculative_method="dspark", num_speculative_tokens=7
        )
    config = KimiK3Config.initialize(
        pipeline_config, model_config, max_seq_len=_MAX_LENGTH
    )
    assert (config.vision_config is None) == dspark
    assert _vision_enabled(KimiK3VisionEncoder, config) == (not dspark)
    assert config.attn_kv_params.num_draft_tokens == (7 if dspark else 0)


def test_a_vision_width_unlike_the_text_width_is_refused(
    configs: tuple[Any, Any, Any],
) -> None:
    huggingface_config, pipeline_config, _ = configs
    huggingface_config.vision_config = SimpleNamespace(
        patch_size=14,
        vt_hidden_size=64,
        qkv_hidden_size=64,
        vt_intermediate_size=128,
        vt_num_attention_heads=1,
        vt_num_hidden_layers=1,
        init_pos_emb_height=4,
        init_pos_emb_width=4,
        init_pos_emb_time=4,
        merge_kernel_size=[2, 2],
        mm_hidden_size=64,
        text_hidden_size=_HIDDEN_SIZE * 2,
        projector_ln_eps=1e-5,
    )
    with pytest.raises(ValueError, match="text_hidden_size"):
        KimiK3Config.initialize(
            pipeline_config, pipeline_config.model, max_seq_len=_MAX_LENGTH
        )
