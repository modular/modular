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

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from max.driver import DeviceSpec
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.architectures.kimi_k3_modulev3.model_config import (
    KimiK3Config,
)
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig

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


def test_state_bytes_match_the_declared_pool_shapes(
    configs: tuple[Any, Any, Any],
) -> None:
    _, pipeline_config, _ = configs

    config = KimiK3Config.initialize(
        pipeline_config, pipeline_config.model, max_seq_len=_MAX_LENGTH
    )

    assert config.ssm_dtype == DType.bfloat16
    assert config.recurrent_state_dtype == DType.float32
    assert config.per_request_state_bytes() == _STATE_BYTES_PER_REQUEST
    assert config.per_request_state_bytes() > 0


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
    assert config.mla_prefill_scratch_bytes(_MAX_LENGTH) > 0


def test_decompressed_cache_has_no_mla_prefill_window(
    configs: tuple[Any, Any, Any],
) -> None:
    _, pipeline_config, _ = configs

    config = KimiK3Config.initialize(
        pipeline_config, pipeline_config.model, max_seq_len=_MAX_LENGTH
    )

    assert not config.mla_latent_cache
    assert config.mla_prefill_scratch_bytes(_MAX_LENGTH) == 0


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


def test_ep_comm_buffer_is_zero_without_expert_parallelism(
    configs: tuple[Any, Any, Any],
) -> None:
    _, pipeline_config, model_config = configs

    config = KimiK3Config.initialize(
        pipeline_config, model_config, max_seq_len=_MAX_LENGTH
    )

    assert not config.expert_parallel
    assert (
        config.ep_comm_buffer_bytes(pipeline_config, use_allreduce=False) == 0
    )


def test_ep_comm_buffer_is_positive_under_expert_parallelism(
    configs: tuple[Any, Any, Any],
) -> None:
    _, pipeline_config, model_config = configs
    pipeline_config.runtime = PipelineRuntimeConfig(
        **{**pipeline_config.runtime.model_dump(), "ep_size": 2}
    )

    config = KimiK3Config.initialize(
        pipeline_config, model_config, max_seq_len=_MAX_LENGTH
    )

    assert config.expert_parallel
    assert config.ep_comm_buffer_bytes(pipeline_config, use_allreduce=False) > 0


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
