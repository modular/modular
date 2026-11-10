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

The gfx950 chunk-workspace tests patch the driver queries inside
``max.nn._graph_ops.state_space.kimi_delta``, the module whose helper decides the
reservation, so they run from any host.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, cast

import pytest
from max.driver import Device, DeviceSpec
from max.dtype import DType
from max.nn._graph_ops.state_space import kimi_delta
from max.nn._graph_ops.state_space.kimi_delta import kda_chunk_workspace_bytes
from max.pipelines.architectures.kimi_k3_modulev3 import memory_planner
from max.pipelines.architectures.kimi_k3_modulev3.memory_planner import (
    KimiK3ModuleV3MemoryPlanner,
    _layer_counts,
    _per_request_state_bytes,
    vision_tower_bytes,
)
from max.pipelines.architectures.kimi_k3_modulev3.memory_planner import (
    _text_config as _resolve_text_config,
)
from max.pipelines.architectures.kimi_k3_modulev3.state_cache import (
    kda_state_regions,
)
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.modeling.config_enums import PipelineRole
from max.pipelines.speculative.config import SpeculativeConfig
from max.support.math import ceildiv

_GiB = 1024**3

_FULL_ATTN_LAYERS = [4, 8]
_KDA_LAYERS = [1, 2, 3, 5, 6, 7]

_KDA_NUM_HEADS = 8
_KDA_HEAD_DIM = 32
_KDA_CONV_KERNEL = 4

# Published K3's KDA geometry, which the gfx950 chunk path is built for.
_K3_KDA_HEADS = 96
_K3_KDA_HEAD_DIM = 128

_PER_REQUEST_BF16 = 6 * ((768 * 3 + 8 * 32 * 32) * 2)
_PER_REQUEST_MIXED = 6 * (768 * 3 * 2 + 8 * 32 * 32 * 4)
_PER_REQUEST_FP32 = 2 * _PER_REQUEST_BF16

_DRAFT_TOKENS = 7
"""K3's DSpark draft width, as a real speculative config declares it."""

_RING_LEN = 8
"""The compiled ring length the ``1 + _DRAFT_TOKENS`` window needs."""

# One request's verify ring: num_layers * heads * ring_len * stride * f32
# bytes. The stride is the record's own 2 * key_dim + value_dim = 96
# elements -- already line-aligned at this geometry, and the page it forms
# divides the live leaves' pages.
_PER_REQUEST_RING = 6 * 8 * _RING_LEN * 96 * 4

_MAX_BATCH_INPUT_TOKENS = 8192
_MAX_LENGTH = 512

# Published K3's MLA geometry, whose 576-wide latent is what the fused MLA
# kernels (and so their prefill window) serve.
_K3_KV_LORA = 512
_K3_QK_ROPE = 64

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


def _text_config(
    linear_attn_config: dict[str, Any] | None = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        hidden_size=_HIDDEN,
        num_attention_heads=8,
        num_hidden_layers=8,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        q_lora_rank=256,
        kv_lora_rank=128,
        linear_attn_config=(
            linear_attn_config
            or {
                "full_attn_layers": _FULL_ATTN_LAYERS,
                "kda_layers": _KDA_LAYERS,
                "num_heads": _KDA_NUM_HEADS,
                "head_dim": _KDA_HEAD_DIM,
                "short_conv_kernel_size": _KDA_CONV_KERNEL,
            }
        ),
        vocab_size=163840,
        rms_norm_eps=1e-5,
        first_k_dense_replace=1,
        routed_expert_hidden_size=_LATENT,
        moe_intermediate_size=_MOE_FFN,
        num_experts=_NUM_EXPERTS,
        num_experts_per_token=_TOP_K,
        num_shared_experts=1,
    )


def _k3_mla_text_config() -> types.SimpleNamespace:
    """The text config at published K3's MLA geometry, which alone opens
    the fused-kernel latent-cache path the prefill window serves."""
    config = _text_config()
    config.kv_lora_rank = _K3_KV_LORA
    config.qk_rope_head_dim = _K3_QK_ROPE
    return config


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


def _huggingface_config(
    *,
    vision: bool,
    stored_dtype: str = "bfloat16",
    linear_attn_config: dict[str, Any] | None = None,
    text_config: types.SimpleNamespace | None = None,
) -> Any:
    text_only = types.SimpleNamespace(
        text_config=text_config or _text_config(linear_attn_config),
        torch_dtype=stored_dtype,
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
    kv_cache_format: str | None = None,
) -> Any:
    return types.SimpleNamespace(
        kv_cache=KVCacheConfig(
            state_pool_dtype=pool_dtype, kv_cache_format=kv_cache_format
        ),
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
    pipeline_role: PipelineRole = "prefill_and_decode",
    enable_chunked_prefill: bool = True,
    linear_attn_config: dict[str, Any] | None = None,
    draft_model: types.SimpleNamespace | None = None,
    kv_cache_format: str | None = None,
    speculative: SpeculativeConfig | types.SimpleNamespace | None = None,
    text_config: types.SimpleNamespace | None = None,
) -> PipelineConfig:
    huggingface_config = _huggingface_config(
        vision=vision,
        linear_attn_config=linear_attn_config,
        text_config=text_config,
    )
    model_config = _model_config(
        huggingface_config,
        encoding=encoding,
        n_devices=n_devices,
        pool_dtype=pool_dtype,
        kv_cache_format=kv_cache_format,
    )
    model_config.weights_size = lambda: weights_total
    stub = types.SimpleNamespace(
        model=model_config,
        speculative=speculative,
        draft_model=draft_model,
        runtime=PipelineRuntimeConfig(
            max_batch_input_tokens=_MAX_BATCH_INPUT_TOKENS,
            max_batch_size=max_batch_size,
            ep_size=ep_size,
            pipeline_role=pipeline_role,
            enable_chunked_prefill=enable_chunked_prefill,
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


def test_a_speculative_build_prices_the_verify_ring() -> None:
    """A speculative config draws one scratch ring page per request on top
    of the live pools, sized off the same region declaration the cache
    builds from."""
    text_config = _text_config()
    ring_len = 8
    live = _per_request_state_bytes(
        text_config,
        n_devices=2,
        ssm_dtype=DType.bfloat16,
        recurrent_state_dtype=DType.float32,
        num_kda=len(_KDA_LAYERS),
    )
    ringed = _per_request_state_bytes(
        text_config,
        n_devices=2,
        ssm_dtype=DType.bfloat16,
        recurrent_state_dtype=DType.float32,
        num_kda=len(_KDA_LAYERS),
        ring_len=ring_len,
    )

    regions = kda_state_regions(
        num_kda_layers=len(_KDA_LAYERS),
        conv_dim=3 * _KDA_NUM_HEADS * _KDA_HEAD_DIM,
        conv_kernel_size=_KDA_CONV_KERNEL,
        num_heads=_KDA_NUM_HEADS,
        head_dim=_KDA_HEAD_DIM,
        conv_dtype=DType.bfloat16,
        recurrent_dtype=DType.float32,
        num_devices=2,
        ring_len=ring_len,
    )

    assert ringed - live == regions[-1].bytes_per_page


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


def _dspark_draft_model(heads: int = 64) -> types.SimpleNamespace:
    """A latent-capable DSpark draft checkpoint, as the planner reads it.

    The published head geometry -- the drafter's latent is 576 wide, which
    is what lets it serve through the fused MLA kernels at all. The planner
    reads the geometry straight off the config's attributes, so the stub
    carries them as attributes.
    """
    return types.SimpleNamespace(
        huggingface_config=types.SimpleNamespace(
            model_type="k3_dspark",
            num_attention_heads=heads,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
    )


# A speculative config as the planner reads it: the method and token count
# the target's own speculative term reads, and the draft width the drafter's
# cache leaf is stamped with.
_SPECULATIVE = types.SimpleNamespace(
    speculative_method="dspark", draft_width=7, num_speculative_tokens=7
)


def _drafter_window(
    planner: KimiK3ModuleV3MemoryPlanner,
    *,
    n_devices: int = 1,
    pipeline_role: PipelineRole = "prefill_and_decode",
    kv_cache_format: str | None = None,
    max_batch_size: int | None = None,
    speculative: SpeculativeConfig | types.SimpleNamespace | None = None,
) -> int:
    """The drafter's share of the activation estimate.

    A speculative build also prices the target's plan window at every
    role, so the drafter's term is the estimate with the draft less the
    same config's estimate without it.
    """

    def estimate(draft_model: types.SimpleNamespace | None) -> int:
        pipeline_config = _pipeline_config(
            draft_model=draft_model,
            n_devices=n_devices,
            pipeline_role=pipeline_role,
            kv_cache_format=kv_cache_format,
            max_batch_size=max_batch_size,
            speculative=speculative,
        )
        return planner.estimate_activation_memory(
            pipeline_config, pipeline_config.model.huggingface_config
        )

    return estimate(_dspark_draft_model()) - estimate(None)


def test_a_dspark_draft_reserves_its_prefill_window_at_every_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drafter's window is a decode-step cost the planner must price.

    A BF16 latent never folds, so the block call takes the one-shot
    decompression prefill arm on every decode step -- including a
    decode-only pipeline, whose own target term is role-gated away. The
    window is the plan cap times the full head count's per-row key and
    value bytes, the same shape the target's term prices.
    """
    monkeypatch.setattr(memory_planner, "accelerator_api", lambda: "hip")
    window = _drafter_window(
        _planner(),
        pipeline_role="decode_only",
        max_batch_size=4,
        speculative=_SPECULATIVE,
    )

    # The block call's window is its cache length -- the committed
    # context plus up to ``draft_width + 1`` accepted rows -- plus the
    # block's own ``draft_width``: the same ``2 * 7 + 1`` rows of slack
    # the pool reserves for a block draft, which the plan cap must cover
    # or the plan silently truncates the trailing windows. The per-row
    # cost is every head's ``qk_nope + v_head`` at bf16, and the cap
    # holds one such window per batch slot.
    cap = ceildiv(_MAX_LENGTH + 15, 128) * 128
    assert window == 4 * cap * 64 * (128 + 128) * 2


def _k3_linear_attn_config(
    gate_lower_bound: float | None = -5.0,
) -> dict[str, Any]:
    """K3's KDA geometry: the widths the gfx950 chunk path is built for."""
    return {
        "full_attn_layers": _FULL_ATTN_LAYERS,
        "kda_layers": _KDA_LAYERS,
        "num_heads": _K3_KDA_HEADS,
        "head_dim": _K3_KDA_HEAD_DIM,
        "short_conv_kernel_size": _KDA_CONV_KERNEL,
        "gate_lower_bound": gate_lower_bound,
    }


def _patch_driver(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    monkeypatch.setattr(kimi_delta, "accelerator_api", lambda: api)
    monkeypatch.setattr(
        kimi_delta, "accelerator_architecture_name", lambda: architecture
    )


def _activation_memory(
    *,
    linear_attn_config: dict[str, Any] | None = None,
    n_devices: int = 1,
    ep_size: int = 1,
    max_batch_size: int | None = None,
    pool_dtype: str | None = None,
    pipeline_role: PipelineRole = "prefill_and_decode",
    enable_chunked_prefill: bool = True,
) -> int:
    pipeline_config = _pipeline_config(
        linear_attn_config=linear_attn_config,
        n_devices=n_devices,
        ep_size=ep_size,
        max_batch_size=max_batch_size,
        pool_dtype=pool_dtype,
        pipeline_role=pipeline_role,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    return _planner().estimate_activation_memory(
        pipeline_config, pipeline_config.model.huggingface_config
    )


def test_tp8_reserves_the_chunk_workspaces_on_gfx950(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gfx950 ``kda_chunk`` launch allocates its workspaces per call.

    TP8 splits the 96 KDA heads 12 per device, and the estimate is compared
    against free memory summed over devices, so the reservation is the
    per-device bytes times the device count. The token bound caps at
    ``max_batch_size * max_length``; every other activation term is zero for
    this fixture.
    """
    _patch_driver(monkeypatch, "hip", "gfx950")

    n_devices = 8
    max_batch_size = 8
    activation = _activation_memory(
        n_devices=n_devices,
        max_batch_size=max_batch_size,
        linear_attn_config=_k3_linear_attn_config(),
    )

    tokens = min(_MAX_BATCH_INPUT_TOKENS, max_batch_size * _MAX_LENGTH)
    expected = n_devices * kda_chunk_workspace_bytes(
        _K3_KDA_HEADS // n_devices, _K3_KDA_HEAD_DIM, tokens, max_batch_size
    )
    assert activation == expected


def test_ep_mesh_reserves_replicated_chunk_workspaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the pure-EP mesh the KDA layers run replicated, 96 heads each."""
    _patch_driver(monkeypatch, "hip", "gfx950")

    n_devices = 8
    max_batch_size = 8
    routed = _activation_memory(
        linear_attn_config=_k3_linear_attn_config(),
        n_devices=n_devices,
        ep_size=n_devices,
        max_batch_size=max_batch_size,
    )
    sequential = _activation_memory(
        linear_attn_config=_k3_linear_attn_config(gate_lower_bound=None),
        n_devices=n_devices,
        ep_size=n_devices,
        max_batch_size=max_batch_size,
    )

    # The EP dispatch buffers are the only other term; differencing the two
    # gate arms removes it without re-deriving it here.
    tokens = min(_MAX_BATCH_INPUT_TOKENS, max_batch_size * _MAX_LENGTH)
    expected = n_devices * kda_chunk_workspace_bytes(
        _K3_KDA_HEADS, _K3_KDA_HEAD_DIM, tokens, max_batch_size
    )
    assert routed - sequential == expected


@pytest.mark.parametrize(
    ("api", "architecture"), [("cuda", "sm_100a"), ("hip", "gfx942")]
)
def test_off_gfx950_reserves_no_chunk_workspace(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    _patch_driver(monkeypatch, api, architecture)
    assert (
        _activation_memory(
            n_devices=8, linear_attn_config=_k3_linear_attn_config()
        )
        == 0
    )


def test_original_gate_reserves_no_chunk_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AMD chunk kernels implement only the bounded ``"safe"`` gate."""
    _patch_driver(monkeypatch, "hip", "gfx950")
    assert (
        _activation_memory(
            n_devices=8,
            linear_attn_config=_k3_linear_attn_config(gate_lower_bound=None),
        )
        == 0
    )


def test_bf16_state_pool_reserves_no_chunk_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AMD chunk kernels take only a float32 state pool."""
    _patch_driver(monkeypatch, "hip", "gfx950")
    assert (
        _activation_memory(
            n_devices=8,
            pool_dtype="bfloat16",
            linear_attn_config=_k3_linear_attn_config(),
        )
        == 0
    )


def test_decode_only_reserves_no_chunk_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decode-only pipeline never prefills, so the chunk path never runs."""
    _patch_driver(monkeypatch, "hip", "gfx950")

    routed = _activation_memory(
        n_devices=8,
        max_batch_size=8,
        linear_attn_config=_k3_linear_attn_config(),
    )
    decode_only = _activation_memory(
        n_devices=8,
        max_batch_size=8,
        pipeline_role="decode_only",
        linear_attn_config=_k3_linear_attn_config(),
    )

    assert routed > 0
    assert decode_only == 0


def test_disabled_chunked_prefill_reserves_the_soft_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without chunked prefill the token budget is soft.

    One oversized prompt rides on top of a full batch, so the reservation
    covers ``max_batch_input_tokens + max_length`` tokens, not the hard
    budget alone.
    """
    _patch_driver(monkeypatch, "hip", "gfx950")

    n_devices = 8
    max_batch_size = 32
    routed = _activation_memory(
        n_devices=n_devices,
        max_batch_size=max_batch_size,
        enable_chunked_prefill=False,
        linear_attn_config=_k3_linear_attn_config(),
    )
    chunked = _activation_memory(
        n_devices=n_devices,
        max_batch_size=max_batch_size,
        linear_attn_config=_k3_linear_attn_config(),
    )

    tokens = min(
        _MAX_BATCH_INPUT_TOKENS + _MAX_LENGTH, max_batch_size * _MAX_LENGTH
    )
    expected = n_devices * kda_chunk_workspace_bytes(
        _K3_KDA_HEADS // n_devices, _K3_KDA_HEAD_DIM, tokens, max_batch_size
    )
    assert routed == expected
    assert routed > chunked


def test_unset_max_batch_size_uses_the_inferred_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``max_batch_size`` unset the reservation uses the inferred size.

    Planning never writes the batch size :meth:`infer_max_batch_size` chose
    back to the runtime config, so the plumbing must carry it separately;
    dropping it would size the workspace for one giant batch of
    ``max_batch_input_tokens`` single-token sequences instead.
    """
    _patch_driver(monkeypatch, "hip", "gfx950")

    n_devices = 8
    planner = _planner()
    pipeline_config = _pipeline_config(
        n_devices=n_devices,
        max_batch_size=None,
        linear_attn_config=_k3_linear_attn_config(),
    )
    weights_size = planner.estimate_weights_size(pipeline_config)
    inferred = planner.infer_max_batch_size(
        pipeline_config, _devices(free_memory=38_000_000), weights_size
    )

    assert inferred is not None
    # The inferred size must bite: at or above 8192 // 512 = 16 sequences
    # the cap max_batch_size * max_length stops discriminating.
    assert inferred < _MAX_BATCH_INPUT_TOKENS // _MAX_LENGTH

    activation = planner.estimate_activation_memory(
        pipeline_config, pipeline_config.model.huggingface_config
    )
    tokens = min(_MAX_BATCH_INPUT_TOKENS, inferred * _MAX_LENGTH)
    expected = n_devices * kda_chunk_workspace_bytes(
        _K3_KDA_HEADS // n_devices, _K3_KDA_HEAD_DIM, tokens, inferred
    )
    assert activation == expected


def test_a_decode_only_speculative_build_reserves_the_mla_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decode-only node never prefills, but its verifies still read the
    MLA plan window wherever the decode fold cannot serve them (a bf16
    latent cache on AMD), and the pairing sizes that window's cap whatever
    the role -- so the reservation must follow the speculative config,
    not the role."""
    monkeypatch.setattr(memory_planner, "accelerator_api", lambda: "hip")
    spec = types.SimpleNamespace(
        speculative_method="dspark", num_speculative_tokens=7
    )

    def estimate(
        pipeline_role: PipelineRole, speculative: types.SimpleNamespace | None
    ) -> int:
        pipeline_config = _pipeline_config(
            pipeline_role=pipeline_role,
            max_batch_size=4,
            speculative=speculative,
            text_config=_k3_mla_text_config(),
        )
        return _planner().estimate_activation_memory(
            pipeline_config, pipeline_config.model.huggingface_config
        )

    # Plain decode never plans, so the role alone still reserves nothing.
    assert estimate("decode_only", None) == 0
    # The verify's TG worst case: all 4 batch slots at the rope table's row
    # count -- ``max_seq_len`` plus the 8 verify rows, page-aligned to 640
    # -- times 8 heads of one 128-wide nope+v window in bfloat16.
    assert estimate("decode_only", spec) == 4 * 640 * 8 * (64 + 64) * 2


def test_per_request_state_bytes_covers_only_the_kda_layers() -> None:
    huggingface_config = _huggingface_config(vision=False)
    text_config = _resolve_text_config(huggingface_config)
    num_kda, num_full_attn = _layer_counts(text_config)
    assert num_kda == len(_KDA_LAYERS)
    assert num_full_attn == len(_FULL_ATTN_LAYERS)

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
