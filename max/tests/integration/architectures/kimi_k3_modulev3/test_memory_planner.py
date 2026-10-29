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
"""Component tests for :class:`KimiK3ModuleV3MemoryPlanner`.

No network, GPU or weights: the HF config is a synthetic stand-in and the
pipeline config is built from the real ``PipelineRuntimeConfig``/
``KVCacheConfig`` classes so a new required field cannot silently go unread
here. Mirrors ``kimi_k3/tests/test_memory_planner.py``'s behaviours, adjusted
for this planner's narrower scope (no DSpark, no KDA live-state term).
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, cast

import pytest
from max.driver import Device, DeviceSpec
from max.pipelines.architectures.kimi_k3_modulev3.memory_planner import (
    KimiK3ModuleV3MemoryPlanner,
    _layer_counts,
    _per_request_state_bytes,
    vision_tower_bytes,
)
from max.pipelines.architectures.kimi_k3_modulev3.memory_planner import (
    _text_config as _resolve_text_config,
)
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from max.pipelines.lib.config import PipelineConfig

_GiB = 1024**3

_FULL_ATTN_LAYERS = [4, 8]
_KDA_LAYERS = [1, 2, 3, 5, 6, 7]

_KDA_NUM_HEADS = 8
_KDA_HEAD_DIM = 32
_KDA_CONV_KERNEL = 4

_PER_REQUEST_BF16 = 6 * ((768 * 3 + 8 * 32 * 32) * 2)
_PER_REQUEST_MIXED = 6 * (768 * 3 * 2 + 8 * 32 * 32 * 4)
_PER_REQUEST_FP32 = 2 * _PER_REQUEST_BF16

_MAX_BATCH_INPUT_TOKENS = 8192
_MAX_LENGTH = 512

_HIDDEN = 1024
_LATENT = 512
_MOE_FFN = 256
_NUM_EXPERTS = 8
_TOP_K = 2
_MOE_LAYERS = 7
_REPLICATED_ELEMS = (
    len(_KDA_LAYERS) * _KDA_HEAD_DIM * _HIDDEN
    + len(_FULL_ATTN_LAYERS) * _HIDDEN * (256 + 128 + 32)
    + _MOE_LAYERS * (_LATENT * _HIDDEN + _NUM_EXPERTS * (_HIDDEN + 1))
)
_EP_EXTRA_REPLICATED_ELEMS = _MOE_LAYERS * (
    _HIDDEN * _LATENT + 3 * _HIDDEN * _MOE_FFN
)


def _text_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        hidden_size=_HIDDEN,
        num_attention_heads=8,
        num_hidden_layers=8,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        q_lora_rank=256,
        kv_lora_rank=128,
        linear_attn_config={
            "full_attn_layers": _FULL_ATTN_LAYERS,
            "kda_layers": _KDA_LAYERS,
            "num_heads": _KDA_NUM_HEADS,
            "head_dim": _KDA_HEAD_DIM,
            "short_conv_kernel_size": _KDA_CONV_KERNEL,
        },
        vocab_size=163840,
        rms_norm_eps=1e-5,
        first_k_dense_replace=1,
        routed_expert_hidden_size=_LATENT,
        moe_intermediate_size=_MOE_FFN,
        num_experts=_NUM_EXPERTS,
        num_experts_per_token=_TOP_K,
        num_shared_experts=1,
    )


_VT_HIDDEN = 256
_VT_QKV_HIDDEN = 1536
_VT_INTERMEDIATE = 1024
_VT_LAYERS = 24
_VT_PATCH = 14
_VT_POS_EMB = 64


def _vision_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        vt_hidden_size=_VT_HIDDEN,
        vt_intermediate_size=_VT_INTERMEDIATE,
        vt_num_hidden_layers=_VT_LAYERS,
        qkv_hidden_size=_VT_QKV_HIDDEN,
        patch_size=_VT_PATCH,
        num_channels=3,
        init_pos_emb_height=_VT_POS_EMB,
        init_pos_emb_width=_VT_POS_EMB,
    )


_VT_ELEMS = _VT_LAYERS * (
    4 * _VT_QKV_HIDDEN * _VT_HIDDEN
    + 2 * _VT_HIDDEN * _VT_INTERMEDIATE
    + 2 * _VT_HIDDEN
) + (
    _VT_HIDDEN * 3 * _VT_PATCH**2
    + _VT_POS_EMB * _VT_POS_EMB * _VT_HIDDEN
    + _VT_HIDDEN
)


def _huggingface_config(*, vision: bool, stored_dtype: str = "bfloat16") -> Any:
    text_only = types.SimpleNamespace(
        text_config=_text_config(), torch_dtype=stored_dtype
    )
    if not vision:
        return text_only
    text_only.vision_config = _vision_config()
    return text_only


def _model_config(
    huggingface_config: Any,
    *,
    encoding: str,
    n_devices: int = 1,
    pool_dtype: str | None = None,
) -> Any:
    return types.SimpleNamespace(
        kv_cache=KVCacheConfig(state_pool_dtype=pool_dtype),
        quantization_encoding=encoding,
        graph_quantization_encoding=None,
        weight_path=[Path(f"model-{encoding}.safetensors")],
        rope_type="normal",
        device_specs=[
            DeviceSpec(id=i, device_type="gpu") for i in range(n_devices)
        ],
        default_device_spec=DeviceSpec(id=0, device_type="gpu"),
        data_parallel_degree=1,
        huggingface_config=huggingface_config,
        max_length=_MAX_LENGTH,
        model_path="kimi-k3-tiny",
        weights_size=lambda: 0,
    )


def _pipeline_config(
    *,
    encoding: str = "bfloat16",
    ep_size: int = 1,
    n_devices: int = 1,
    pool_dtype: str | None = None,
    vision: bool = False,
    weights_total: int = 0,
    max_batch_size: int | None = None,
) -> PipelineConfig:
    huggingface_config = _huggingface_config(vision=vision)
    model_config = _model_config(
        huggingface_config,
        encoding=encoding,
        n_devices=n_devices,
        pool_dtype=pool_dtype,
    )
    model_config.weights_size = lambda: weights_total
    stub = types.SimpleNamespace(
        model=model_config,
        speculative=None,
        runtime=PipelineRuntimeConfig(
            max_batch_input_tokens=_MAX_BATCH_INPUT_TOKENS,
            max_batch_size=max_batch_size,
            ep_size=ep_size,
        ),
    )
    return cast(PipelineConfig, stub)


def _devices(free_memory: int) -> list[Device]:
    return cast(
        "list[Device]",
        [types.SimpleNamespace(stats={"free_memory": free_memory})],
    )


def _planner() -> KimiK3ModuleV3MemoryPlanner:
    # The planner reads straight off pipeline_config/huggingface_config, so
    # the arch-config instance it is constructed with is never consulted --
    # only its shape (`ModelConfigWithKVCache`) matters to the base class.
    fake_config = types.SimpleNamespace(devices=[], get_kv_params=lambda: None)
    return KimiK3ModuleV3MemoryPlanner(fake_config)


def test_weights_size_discounts_the_unloaded_vision_tower() -> None:
    discount = _VT_ELEMS * 2
    assert vision_tower_bytes(_huggingface_config(vision=True)) == discount

    estimate = _planner().estimate_weights_size(
        _pipeline_config(vision=True, weights_total=4 * _GiB)
    )

    assert estimate == 4 * _GiB - discount


def test_weights_size_is_unchanged_for_a_text_only_checkpoint() -> None:
    estimate = _planner().estimate_weights_size(
        _pipeline_config(vision=False, weights_total=4 * _GiB)
    )

    assert estimate == 4 * _GiB


def test_weights_size_never_goes_negative() -> None:
    estimate = _planner().estimate_weights_size(
        _pipeline_config(vision=True, weights_total=1024)
    )

    assert estimate == 0


def test_replicated_weights_grow_under_expert_parallelism() -> None:
    """EP replicates the exit projection and the shared experts; TP splits both."""
    tp_estimate = _planner().estimate_weights_size(
        _pipeline_config(n_devices=2, ep_size=1, weights_total=8 * _GiB)
    )
    ep_estimate = _planner().estimate_weights_size(
        _pipeline_config(n_devices=2, ep_size=2, weights_total=8 * _GiB)
    )

    # Both are cluster totals; EP's extra replicated bytes ride on every
    # device while the split part is unchanged (it is still divided across
    # the same 2 devices), so only (n_devices - 1) extra copies show up.
    assert ep_estimate > tp_estimate
    elem = 2  # bfloat16
    n_devices = 2
    expected_extra = (n_devices - 1) * _EP_EXTRA_REPLICATED_ELEMS * elem
    assert ep_estimate - tp_estimate == expected_extra


def test_activation_memory_does_not_reserve_the_kda_state() -> None:
    """The state is a cache leaf, so this planner reserves none of it."""
    pipeline_config = _pipeline_config()
    activation = _planner().estimate_activation_memory(
        pipeline_config, pipeline_config.model.huggingface_config
    )
    assert activation == 0


def test_activation_memory_includes_ep_buffers_on_every_device() -> None:
    n_devices = 2
    pipeline_config = _pipeline_config(n_devices=n_devices, ep_size=n_devices)

    activation = _planner().estimate_activation_memory(
        pipeline_config, pipeline_config.model.huggingface_config
    )

    assert activation > 0
    recv_rows = _MAX_BATCH_INPUT_TOKENS * min(_NUM_EXPERTS, n_devices * _TOP_K)
    per_device = (
        _MAX_BATCH_INPUT_TOKENS + recv_rows
    ) * _LATENT * 2 + _TOP_K * _MAX_BATCH_INPUT_TOKENS * _LATENT * 2
    assert activation == n_devices * per_device


def test_activation_memory_reserves_the_mxfp4_dequant_buffer() -> None:
    pipeline_config = _pipeline_config(encoding="float4_e2m1fnx2")

    activation = _planner().estimate_activation_memory(
        pipeline_config, pipeline_config.model.huggingface_config
    )

    # 8 experts x 512 latent x 3 x 256 FFN x 2 bytes, two layers' worth.
    assert activation == 2 * (8 * 512 * 3 * 256 * 2)


def test_dense_checkpoint_reserves_no_dequant_buffer() -> None:
    pipeline_config = _pipeline_config(encoding="bfloat16")

    activation = _planner().estimate_activation_memory(
        pipeline_config, pipeline_config.model.huggingface_config
    )

    assert activation == 0


def test_per_request_state_bytes_covers_only_the_kda_layers() -> None:
    huggingface_config = _huggingface_config(vision=False)
    text_config = _resolve_text_config(huggingface_config)
    num_kda, num_full_attn = _layer_counts(text_config)
    assert num_kda == len(_KDA_LAYERS)
    assert num_full_attn == len(_FULL_ATTN_LAYERS)

    from max.dtype import DType

    per_request = _per_request_state_bytes(
        text_config,
        n_devices=1,
        ssm_dtype=DType.bfloat16,
        recurrent_state_dtype=DType.float32,
        num_kda=num_kda,
    )
    assert per_request == _PER_REQUEST_MIXED


def test_infer_max_batch_size_accounts_for_the_kda_state() -> None:
    pipeline_config = _pipeline_config(weights_total=900_000_000)
    weights_size = _planner().estimate_weights_size(pipeline_config)

    inferred = _planner().infer_max_batch_size(
        pipeline_config, _devices(free_memory=1 * _GiB), weights_size
    )

    budget = int(1 * _GiB * 0.9) - weights_size
    assert inferred == (budget // 2) // _PER_REQUEST_MIXED


def test_inferred_batch_size_is_clamped() -> None:
    pipeline_config = _pipeline_config(weights_total=1 * _GiB)
    weights_size = _planner().estimate_weights_size(pipeline_config)
    assert (
        _planner().infer_max_batch_size(
            pipeline_config, _devices(free_memory=200 * _GiB), weights_size
        )
        == 512
    )

    pipeline_config = _pipeline_config(weights_total=2 * _GiB)
    weights_size = _planner().estimate_weights_size(pipeline_config)
    assert (
        _planner().infer_max_batch_size(
            pipeline_config, _devices(free_memory=1 * _GiB), weights_size
        )
        == 1
    )


@pytest.mark.parametrize("pool_dtype", [None, "bfloat16", "float32"])
def test_recurrent_override_updates_the_state_cost(
    pool_dtype: str | None,
) -> None:
    pipeline_config = _pipeline_config(pool_dtype=pool_dtype, n_devices=1)
    weights_size = _planner().estimate_weights_size(pipeline_config)
    per_request = (
        _PER_REQUEST_BF16 if pool_dtype == "bfloat16" else _PER_REQUEST_MIXED
    )

    free_memory = (weights_size + 66_367_641) * 10 // 9
    budget = int(free_memory * 0.9) - weights_size
    expected = min(512, max(1, (budget // 2) // per_request))

    inferred = _planner().infer_max_batch_size(
        pipeline_config, _devices(free_memory), weights_size
    )
    assert inferred == expected


def test_weights_size_without_a_huggingface_config_uses_the_file_total() -> (
    None
):
    model_config = _model_config(None, encoding="bfloat16")
    model_config.weights_size = lambda: 4 * _GiB
    stub = types.SimpleNamespace(
        model=model_config,
        speculative=None,
        runtime=PipelineRuntimeConfig(
            max_batch_input_tokens=_MAX_BATCH_INPUT_TOKENS
        ),
    )
    pipeline_config = cast(PipelineConfig, stub)

    estimate = _planner().estimate_weights_size(pipeline_config)

    assert estimate == 4 * _GiB
