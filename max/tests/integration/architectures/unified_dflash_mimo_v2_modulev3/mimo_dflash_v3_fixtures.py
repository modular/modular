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
"""Fixtures for the ModuleV3 fused MiMo-V2 DFlash tests, without a GPU.

Builds the real fused module (:class:`UnifiedDflashMiMoV2ModuleV3`) on a CPU
mesh from synthetic checkpoints: a tiny MiMo-V2.6-Flash target config at the
real checkpoint's shapes, a tiny DFlash drafter config, and their KV trees.
The stage never executes -- the W4A8 expert kernels have no CPU dispatch --
so a build needs no accelerator and runs anywhere bazel does.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import cast

from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.graph import DeviceRef
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.nn.comm.ep import EPBatchManager, EPConfig
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.dflash_mimo_v2 import DFlashMiMoV2Config
from max.pipelines.architectures.dflash_mimo_v2_modulev3.weight_adapters import (
    MASK_EMBEDDING,
    drafter_tensor_shapes,
)
from max.pipelines.architectures.dflash_mimo_v2_modulev3.weight_adapters import (
    convert_safetensor_state_dict as draft_convert,
)
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    FULL,
    SLIDING,
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.weight_adapters import (
    convert_safetensor_state_dict as target_convert,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.model_config import (
    DRAFT,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.unified_dflash_mimo_v2 import (
    UnifiedDflashMiMoV2ModuleV3,
    UnifiedDflashMiMoV2Spec,
)
from max.pipelines.speculative import SpeculativeConfig

BLOCK = 8
DRAFTS = 7
TAPS = [1, 3]
SAMPLEABLE = 448
DRAFT_HIDDEN = 4096
"""The drafter config's ``hidden_size``, which the mask embedding shares."""


class _FakeAttrConfig:
    """The target checkpoint's config.json, with the Flash values (tiny vocab)."""

    vocab_size = 512
    hidden_size = 4096
    num_hidden_layers = 4  # small: 4 layers, layer 0 dense
    layernorm_epsilon = 1e-6
    hybrid_layer_pattern = [0, 1, 1, 1]
    num_attention_heads = 64
    head_dim = 192
    v_head_dim = 128
    swa_num_attention_heads = 64
    swa_head_dim = 192
    swa_v_head_dim = 128
    num_key_value_heads = 4
    swa_num_key_value_heads = 8
    partial_rotary_factor = 0.334
    rope_theta = 10000000.0
    swa_rope_theta = 10000.0
    sliding_window = 128
    attention_value_scale = 0.707
    add_swa_attention_sink_bias = True
    add_full_attention_sink_bias = False
    intermediate_size = 16384
    moe_layer_freq = [0, 1, 1, 1]
    n_routed_experts = 16
    num_experts_per_tok = 8
    moe_intermediate_size = 2048
    norm_topk_prob = True
    scoring_func = "sigmoid"
    topk_method = "noaux_tc"
    hidden_act = "silu"
    attention_bias = False
    tie_word_embeddings = False
    attention_projection_layout = "fused_qkv"
    n_group = 1
    topk_group = 1
    n_shared_experts = None
    routed_scaling_factor = None
    rope_parameters = {"rope_type": "default", "type": "default"}
    rope_scaling = None
    quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "store_dtype": "mxfp4",
        "mxfp4_block_size": 32,
        "ignored_layers": [],
    }


class _FakeWeight:
    """A checkpoint entry, as the ``Weights`` protocol hands adapters one."""

    def __init__(self, name: str, dtype: DType, shape: list[int]) -> None:
        self._name = name
        self._dtype = dtype
        self._shape = shape

    def data(self) -> WeightData:
        return WeightData(
            name=self._name,
            dtype=self._dtype,
            shape=Shape(self._shape),
            data=Buffer.zeros(tuple(self._shape), dtype=self._dtype),
        )


def _as_weights(
    state: OrderedDict[str, _FakeWeight],
) -> dict[str, Weights]:
    return cast(dict[str, Weights], dict(state))


DFLASH_CONFIG: dict[str, object] = {
    "architectures": ["DFlashDraftModel"],
    "is_causal": False,
    "use_sliding_window": True,
    "hidden_act": "silu",
    "attention_bias": False,
    "tie_word_embeddings": False,
    "add_swa_attention_sink_bias": True,
    "num_hidden_layers": 2,
    "layer_types": ["sliding_attention"] * 2,
    "head_dim": 128,
    "v_head_dim": 128,
    "block_size": BLOCK,
    "hidden_size": 4096,
    "target_hidden_size": 4096,
    "num_target_layers": 4,
    "partial_rotary_factor": 0.5,
    "sliding_window": 1024,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "intermediate_size": 1024,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "rope_parameters": None,
    "dflash_config": {
        "attention_sink_bias": True,
        "block_size": BLOCK,
        "target_layer_ids": TAPS,
        "mask_token_id": 151666,
        "attention_value_scale": 0.612,
    },
}


def target_kv_params(
    devices: list[DeviceRef],
    *,
    num_draft_tokens: int,
    data_parallel_degree: int = 1,
) -> MultiKVCacheParams:
    """The target's ``{sliding, full}`` tree at the speculative build's widths."""
    hf = _FakeAttrConfig
    n_sliding = sum(hf.hybrid_layer_pattern)

    def group(n_kv_heads: int, num_layers: int) -> MHAKVCacheParams:
        return MHAKVCacheParams(
            num_layers=num_layers,
            n_kv_heads=n_kv_heads,
            dtype=DType.bfloat16,
            head_dim=256,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            enable_prefix_caching=False,
            per_layer_buffers=False,
            speculative_method="dflash",
            num_draft_tokens=num_draft_tokens,
        )

    return MultiKVCacheParams.from_params(
        {
            SLIDING: group(hf.swa_num_key_value_heads, n_sliding),
            FULL: group(
                hf.num_key_value_heads, hf.num_hidden_layers - n_sliding
            ),
        }
    )


def draft_kv_params(
    devices: list[DeviceRef], *, data_parallel_degree: int = 1
) -> MHAKVCacheParams:
    """The drafter's context cache: windowed, at the block's draft width."""
    return MHAKVCacheParams(
        num_layers=2,
        n_kv_heads=4,
        dtype=DType.bfloat16,
        head_dim=128,
        devices=devices,
        data_parallel_degree=data_parallel_degree,
        enable_prefix_caching=False,
        per_layer_buffers=False,
        speculative_method="dflash",
        num_draft_tokens=BLOCK,
        window_size=1024,
    )


def target_checkpoint(config: MiMoV2Config) -> OrderedDict[str, _FakeWeight]:
    """Every text-model tensor the real repo's index lists, at true shapes."""
    hf = _FakeAttrConfig
    out: OrderedDict[str, _FakeWeight] = OrderedDict()
    fp8 = DType.float8_e4m3fn
    f32 = DType.float32
    bf16 = DType.bfloat16

    def add(name: str, dtype: DType, shape: list[int]) -> None:
        out[f"model.{name}"] = _FakeWeight(name, dtype, shape)

    n_layers = hf.num_hidden_layers
    add("embed_tokens.weight", bf16, [hf.vocab_size, hf.hidden_size])
    add("norm.weight", bf16, [hf.hidden_size])
    for i in range(n_layers):
        group = SLIDING if hf.hybrid_layer_pattern[i] == 1 else FULL
        layout = config.qkv_layouts[group]
        add(
            f"layers.{i}.self_attn.qkv_proj.weight",
            fp8,
            [layout.chunks * layout.rows, hf.hidden_size],
        )
        add(
            f"layers.{i}.self_attn.qkv_proj.weight_scale_inv",
            f32,
            [layout.chunks * layout.padded_rows // 128, hf.hidden_size // 128],
        )
        add(
            f"layers.{i}.self_attn.o_proj.weight",
            bf16,
            [hf.hidden_size, config.num_attention_heads * hf.v_head_dim],
        )
        if config.sinks[group]:
            add(
                f"layers.{i}.self_attn.attention_sink_bias",
                bf16,
                [config.num_attention_heads],
            )
        add(f"layers.{i}.input_layernorm.weight", bf16, [hf.hidden_size])
        add(
            f"layers.{i}.post_attention_layernorm.weight",
            bf16,
            [hf.hidden_size],
        )
        if i in config.moe_layers:
            add(
                f"layers.{i}.mlp.gate.weight",
                bf16,
                [hf.n_routed_experts, hf.hidden_size],
            )
            add(
                f"layers.{i}.mlp.gate.e_score_correction_bias",
                f32,
                [hf.n_routed_experts],
            )
            for e in range(hf.n_routed_experts):
                for proj, out_dim, in_dim in (
                    ("gate_proj", hf.moe_intermediate_size, hf.hidden_size),
                    ("up_proj", hf.moe_intermediate_size, hf.hidden_size),
                    ("down_proj", hf.hidden_size, hf.moe_intermediate_size),
                ):
                    add(
                        f"layers.{i}.mlp.experts.{e}.{proj}.weight",
                        DType.uint8,
                        [out_dim, in_dim // 2],
                    )
                    add(
                        f"layers.{i}.mlp.experts.{e}.{proj}.weight_scale",
                        DType.uint8,
                        [out_dim, in_dim // 32],
                    )
        else:
            for proj, out_dim, in_dim in (
                ("gate_proj", hf.intermediate_size, hf.hidden_size),
                ("up_proj", hf.intermediate_size, hf.hidden_size),
                ("down_proj", hf.hidden_size, hf.intermediate_size),
            ):
                add(
                    f"layers.{i}.mlp.{proj}.weight",
                    fp8,
                    [out_dim, in_dim],
                )
                add(
                    f"layers.{i}.mlp.{proj}.weight_scale_inv",
                    f32,
                    [out_dim // 128, in_dim // 128],
                )
    add("lm_head.weight", bf16, [hf.vocab_size, hf.hidden_size])
    return out


def adapted_target(config: MiMoV2Config) -> dict[str, WeightData]:
    """The target checkpoint through the ModuleV3 adapter, by MAX name."""
    return target_convert(
        _as_weights(target_checkpoint(config)),
        huggingface_config=_FakeAttrConfig(),
    )


def draft_checkpoint(
    config: DFlashMiMoV2Config,
) -> OrderedDict[str, _FakeWeight]:
    """Every drafter tensor, at the shapes the checkpoint contract names."""
    return OrderedDict(
        (name, _FakeWeight(name, config.dtype, list(shape)))
        for name, shape in drafter_tensor_shapes(config).items()
    )


def adapted_draft(
    config: DFlashMiMoV2Config, mask_embedding: WeightData
) -> dict[str, WeightData]:
    """The drafter checkpoint through the adapter, without the mask embedding."""
    state = dict(
        draft_convert(
            _as_weights(draft_checkpoint(config)), config, mask_embedding
        )
    )
    del state[MASK_EMBEDDING]
    return state


def mask_embedding() -> WeightData:
    """The drafter's trained mask slot embedding, as its file reads it."""
    return WeightData(
        name=MASK_EMBEDDING,
        dtype=DType.bfloat16,
        shape=Shape([DRAFT_HIDDEN]),
        data=Buffer.zeros((DRAFT_HIDDEN,), dtype=DType.bfloat16),
    )


def tiny_draft_config(
    devices: list[DeviceRef],
    kv_params: MHAKVCacheParams,
    *,
    data_parallel_degree: int = 1,
) -> DFlashMiMoV2Config:
    # The degree is threaded the way the pipeline model does, beside the
    # shared reader: the checkpoint's config.json does not carry it.
    return replace(
        DFlashMiMoV2Config.from_dflash_config(
            DFLASH_CONFIG,
            devices=devices,
            kv_params=kv_params,
            max_seq_len=512 + 2 * BLOCK,
        ),
        data_parallel_degree=data_parallel_degree,
    )


class _IdCPU(CPU):
    """A CPU device carrying an arbitrary id, to stand in for a GPU.

    The traced graph inserts its signal buffers only when the module's
    devices carry at least two distinct accelerator ids, and every CPU is id
    0; a mesh of these carries the mesh's signal buffers at the engine's
    slot, as the GPU build does.
    """

    def __init__(self, device_id: int) -> None:
        super().__init__()
        self._device_id = device_id

    @property
    def id(self) -> int:
        return self._device_id


def build_spec(
    devices: list[DeviceRef],
    *,
    data_parallel_degree: int = 1,
    ep_config: EPConfig | None = None,
) -> tuple[
    UnifiedDflashMiMoV2Spec,
    MiMoV2Config,
    DFlashMiMoV2Config,
    MultiKVCacheParams,
]:
    """The fused graph's spec and its target/drafter configs, on ``devices``.

    ``data_parallel_degree`` above 1 builds the DP form: one-device
    replicas, the batch axis named ``dp``, dp-aware KV trees, and -- with
    ``ep_config`` -- the P2P expert backend the replicas dispatch through.
    """
    target_kv = target_kv_params(
        devices,
        num_draft_tokens=BLOCK,
        data_parallel_degree=data_parallel_degree,
    )
    draft_kv = draft_kv_params(
        devices, data_parallel_degree=data_parallel_degree
    )
    # The production call's shape: the config builder takes the whole mesh
    # and the degree, and runs its tensor-parallel divisibility checks
    # (whole qkv chunks per rank, whole 128-row scale granules per rank)
    # over one replica's devices, where the weights shard -- every
    # one-device replica holds them unsharded.
    config = MiMoV2Config.from_huggingface_config(
        _FakeAttrConfig(),
        devices=devices,
        kv_params=target_kv,
        max_seq_len=512,
        data_parallel_degree=data_parallel_degree,
    )
    config.data_parallel_degree = data_parallel_degree
    config.ep_config = ep_config
    draft = tiny_draft_config(
        devices, draft_kv, data_parallel_degree=data_parallel_degree
    )
    # Widen as the pipeline model does: the spec target builds wide.
    config.target_layer_ids = list(draft.target_layer_ids)
    config.return_logits = ReturnLogits.VARIABLE
    config.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
    spec = UnifiedDflashMiMoV2Spec(
        target=config,
        draft=draft,
        speculative_config=SpeculativeConfig(
            speculative_method="dflash",
            num_speculative_tokens=DRAFTS,
        ),
        num_speculative_tokens=DRAFTS,
        sampleable_vocab_size=SAMPLEABLE,
    )
    kv_tree = MultiKVCacheParams.from_params(
        {"target": target_kv, DRAFT: draft_kv}
    )
    return spec, config, draft, kv_tree


def build_module(
    spec: UnifiedDflashMiMoV2Spec,
    kv_tree: MultiKVCacheParams,
    *,
    structured_output: bool,
    ep_batch_manager: EPBatchManager | None = None,
) -> UnifiedDflashMiMoV2ModuleV3:
    """The fused module traced on the spec's mesh.

    The mesh follows the pipeline model's: one axis over the devices, named
    ``dp`` under data parallelism (the split placement keys on it) and
    ``tp`` otherwise. CPU refs build a CPU mesh, as the no-GPU staging does;
    GPU refs build over the (possibly virtual) accelerators.
    """
    n_devices = len(spec.target.devices)
    axis_name = "dp" if spec.target.data_parallel_degree > 1 else "tp"
    mesh = DeviceMesh(
        tuple(ref.to_device() for ref in spec.target.devices),
        (n_devices,),
        (axis_name,),
    )
    with default_device(mesh), default_dtype(DType.bfloat16):
        with F.lazy():
            return UnifiedDflashMiMoV2ModuleV3(
                spec,
                mask_embedding(),
                kv_params=kv_tree,
                enable_structured_output=structured_output,
                ep_batch_manager=ep_batch_manager,
            )


def fused_weights(
    config: MiMoV2Config, draft: DFlashMiMoV2Config
) -> dict[str, WeightData]:
    """The fused module's weights: the ``target.``/``draft.``-prefixed state."""
    target_state = adapted_target(config)
    draft_state = adapted_draft(draft, mask_embedding())
    return {
        **{f"target.{k}": v for k, v in target_state.items()},
        **{f"{DRAFT}.{k}": v for k, v in draft_state.items()},
    }
