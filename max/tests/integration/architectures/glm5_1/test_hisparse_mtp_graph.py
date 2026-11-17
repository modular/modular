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
"""The GLM-5.x ModuleV3 fused MTP graph's Hybrid HiSparse wiring, staged with
no GPU.

The whole fused speculative graph -- the target, its NextN draft and the
sequential driver between them -- is staged on a virtual GPU over the tiny
random config the plain-graph HiSparse test uses, twice, with HiSparse off
and on at MTP K = 2. Off, the graph must be exactly today's: the same inputs
and none of the HiSparse ops. On, the three HiSparse input groups sit
immediately after the EP inputs and before the speculative tail, only the
target's verify reads them, and each target ``full`` layer body resolves and
gathers the rows of its consecutive shared followers in one launch, never
in the draft.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from max import functional as F
from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn._graph_ops import HiSparseParams
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.legacy.quant_config import QuantConfig
from max.nn.legacy.transformer import ReturnHiddenStates
from max.pipelines.architectures.glm5_1_modulev3.glm5_1 import Glm5_1Decoder
from max.pipelines.architectures.glm5_1_modulev3.model_config import (
    Glm5_1Config,
)
from max.pipelines.architectures.glm5_1_modulev3.mtp import Glm5_1MTP
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _driver_call,
    _SpecDecodeModule,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
    decode_spec_decode_input_values,
)
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh

_TARGET_ARCH = "GlmMoeDsaForCausalLM"
"""The checkpoint's architecture name, as a served config resolves it."""

_LAYERS = 4
"""A dense-full layer, a shared-MoE layer and two full-MoE layers, so the
full MoE layers must share one subgraph body."""
_INDEXER_TYPES = ["full", "shared", "full", "full"]
_FULL_LAYERS = _INDEXER_TYPES.count("full")
_HEADS = 8
_HIDDEN = 256
_Q_LORA_RANK = 128
_KV_LORA_RANK = 128
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_INDEX_HEAD_DIM = 128
_INDEX_N_HEADS = 2
_INDEX_TOPK = 64
_PAGE_SIZE = 128
_EXPERTS = 4
_K = 2
"""The MTP draft count: the verify runs K + 1 query rows per request."""


def _huggingface_config() -> SimpleNamespace:
    """The tiny GLM checkpoint config the cache params are read from."""
    return SimpleNamespace(
        vocab_size=512,
        hidden_size=_HIDDEN,
        intermediate_size=256,
        moe_intermediate_size=128,
        moe_layer_freq=1,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=_EXPERTS,
        routed_scaling_factor=1.0,
        kv_lora_rank=_KV_LORA_RANK,
        q_lora_rank=_Q_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        norm_topk_prob=False,
        hidden_act="silu",
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleave=True,
        scoring_func="sigmoid",
        attention_bias=False,
        attention_dropout=0.0,
        index_head_dim=_INDEX_HEAD_DIM,
        index_n_heads=_INDEX_N_HEADS,
        index_topk=_INDEX_TOPK,
        indexer_types=list(_INDEXER_TYPES),
        indexer_rope_interleave=True,
    )


def _hisparse() -> HiSparseParams:
    """The sizes a ``--hisparse`` MTP export compiles in at K.

    The verify step runs ``K + 1`` query rows per request, so the hot buffer
    holds one top-k per query row plus one spare of LRU slack.
    """
    return HiSparseParams.for_top_k(
        top_k=_INDEX_TOPK,
        query_rows=1 + _K,
        page_size=_PAGE_SIZE,
        num_groups=_FULL_LAYERS,
    )


def _config(hisparse: HiSparseParams | None) -> Glm5_1Config:
    """A tiny GLM config, placed on one virtual GPU.

    Raises:
        ValueError: As :class:`Glm5_1Config` does, for a HiSparse deployment
            this config does not support.
    """
    devices = [DeviceRef.GPU(0)]
    hf = _huggingface_config()
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    # The pairing's config: the caches carry the draft slots the fused graph
    # reads, which a speculative export's construct_kv_params adds.
    pipeline_config.speculative = SimpleNamespace(
        speculative_method="mtp", num_speculative_tokens=_K
    )
    config = Glm5_1Config(
        dtype=DType.float8_e4m3fn,
        kv_params=Glm5_1Config.construct_kv_params(
            huggingface_config=hf,
            pipeline_config=cast(Any, pipeline_config),
            devices=devices,
            kv_cache_config=KVCacheConfig(),
            cache_dtype=DType.float8_e4m3fn,
        ),
        devices=devices,
        vocab_size=hf.vocab_size,
        hidden_size=hf.hidden_size,
        intermediate_size=hf.intermediate_size,
        moe_intermediate_size=hf.moe_intermediate_size,
        moe_layer_freq=hf.moe_layer_freq,
        num_hidden_layers=hf.num_hidden_layers,
        num_attention_heads=hf.num_attention_heads,
        num_key_value_heads=hf.num_key_value_heads,
        n_shared_experts=hf.n_shared_experts,
        n_routed_experts=hf.n_routed_experts,
        routed_scaling_factor=hf.routed_scaling_factor,
        kv_lora_rank=hf.kv_lora_rank,
        q_lora_rank=hf.q_lora_rank,
        qk_rope_head_dim=hf.qk_rope_head_dim,
        v_head_dim=hf.v_head_dim,
        qk_nope_head_dim=hf.qk_nope_head_dim,
        topk_method=hf.topk_method,
        n_group=hf.n_group,
        topk_group=hf.topk_group,
        num_experts_per_tok=hf.num_experts_per_tok,
        first_k_dense_replace=hf.first_k_dense_replace,
        norm_topk_prob=hf.norm_topk_prob,
        hidden_act=hf.hidden_act,
        max_position_embeddings=hf.max_position_embeddings,
        max_seq_len=hf.max_position_embeddings,
        rms_norm_eps=hf.rms_norm_eps,
        tie_word_embeddings=hf.tie_word_embeddings,
        rope_theta=hf.rope_theta,
        rope_scaling=hf.rope_scaling,
        rope_interleave=hf.rope_interleave,
        scoring_func=hf.scoring_func,
        attention_bias=hf.attention_bias,
        attention_dropout=hf.attention_dropout,
        graph_mode="decode",
        data_parallel_degree=1,
        index_head_dim=hf.index_head_dim,
        index_n_heads=hf.index_n_heads,
        index_topk=hf.index_topk,
        indexer_types=list(_INDEXER_TYPES),
        indexer_rope_interleave=hf.indexer_rope_interleave,
        quantization_encoding="float8_e4m3fn",
        unpadded_vocab_size=None,
        quant_config=QuantConfig.blockscaled_fp8(
            mlp_quantized_layers=range(_LAYERS),
            attn_quantized_layers=range(_LAYERS),
            embedding_output_dtype=DType.bfloat16,
        ),
        correction_bias_dtype=DType.float32,
        hisparse=hisparse,
    )
    # The runtime places the config on the deployment's mesh; one
    # tensor-parallel axis is what a HiSparse deployment runs.
    config.mesh = DeviceMesh(
        (devices[0].to_device(),),
        (1,),
        ("tp",),
    )
    return config


def _with_draft_tokens(params: Any) -> Any:
    """Stamps the draft width on every attention leaf, as the pairing does."""
    if isinstance(params, KVCacheParams):
        return replace(params, num_draft_tokens=_K)
    assert isinstance(params, MultiKVCacheParams)
    return MultiKVCacheParams.from_params(
        {
            name: _with_draft_tokens(child)
            for name, child in params.children.items()
        }
    )


def _draft_config(target: Glm5_1Config) -> Glm5_1Config:
    """The NextN layer's config: the target's, over one-layer caches.

    The layer computes its own top-k, so it indexes every step, which is what
    an empty indexer schedule means.
    """
    kv_params = target.kv_params
    assert isinstance(kv_params, MultiKVCacheParams)
    one_layer = MultiKVCacheParams.from_params(
        {
            name: replace(cast(KVCacheParams, leaf), num_layers=1)
            for name, leaf in kv_params.children.items()
        }
    )
    return replace(target, kv_params=one_layer, indexer_types=[])


def _unified_kv(
    target: Glm5_1Config, draft: Glm5_1Config
) -> MultiKVCacheParams:
    """The ``{"target", "draft"}`` KV tree the fused graph declares."""
    return MultiKVCacheParams.from_params(
        {
            "target": _with_draft_tokens(target.kv_params),
            "draft": _with_draft_tokens(draft.kv_params),
        }
    )


def _pipeline_config() -> Any:
    """What the pairing reads of a served config that prefers ModuleV3."""
    hf = _huggingface_config()
    huggingface_config: Any = SimpleNamespace(
        vocab_size=hf.vocab_size,
        hidden_size=hf.hidden_size,
        get_text_config=lambda: huggingface_config,
    )
    return SimpleNamespace(
        models=SimpleNamespace(main_architecture_name=_TARGET_ARCH),
        # The NextN layer ships in the target's checkpoint.
        draft_model=None,
        speculative=SpeculativeConfig(
            speculative_method="mtp",
            num_speculative_tokens=_K,
        ),
        needs_bitmask_constraints=False,
        runtime=SimpleNamespace(reasoning_parser=None, prefer_module_v3=True),
        model=SimpleNamespace(
            data_parallel_degree=1,
            pipeline_stage=None,
            huggingface_config=huggingface_config,
        ),
    )


def _fuse(
    hisparse: HiSparseParams | None,
) -> tuple[Any, list[TensorType | BufferType], Glm5_1Config]:
    """Builds the fused MTP graph the ModuleV3 pipeline model compiles.

    The target, the NextN draft and the driver stage as the one module the
    pairing compiles, over the tiny config, with the driver built through
    the production ``_driver_call`` so the target adapter sizes the HiSparse
    inputs the way an export does. Returns the staged graph, the input types
    its signature declares, and the target's config.
    """
    target_config = _config(hisparse)
    draft_config = _draft_config(target_config)
    kv_params = _unified_kv(target_config, draft_config)
    mesh = target_config.mesh
    assert mesh is not None
    # The pipeline model builds its modules under ``F.lazy()``, so the
    # random weight initialization defers instead of realizing eagerly.
    with (
        F.lazy(),
        default_device(mesh),
        default_dtype(DType.bfloat16),
    ):
        target = Glm5_1Decoder(target_config)
        draft = Glm5_1MTP(draft_config)
    # One device takes no signal buffers of its own; the trace appends the
    # mesh's past every declared input, as the pipeline model stages it.
    module = _SpecDecodeModule(
        target,
        draft,
        partial(
            _driver_call,
            _pipeline_config(),
            kv_params,
            list(target_config.devices),
            include_signal_buffers=False,
            target=target,
            target_config=target_config,
            draft=draft,
            draft_config=draft_config,
            hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
            trace_owns_signal_buffers=True,
        ),
    )
    call = module.driver_call()
    with default_dtype(DType.bfloat16), F.lazy():
        staged, _registry = module.stage(
            *call.input_types,
            name="UnifiedMTPGlm5_2",
            signal_buffer_index=call.driver.input_spec.signal_buffer_index,
        )
    return staged, list(call.input_types), target_config


def _symbol_count(staged: Any, symbol: str) -> int:
    return len(re.findall(rf'symbol\s*=\s*"{re.escape(symbol)}"', str(staged)))


def _custom_ops(text: str) -> list[tuple[str, list[str]]]:
    """Each custom op's symbol and operands, in print order."""
    ops: list[tuple[str, list[str]]] = []
    for part in text.split("mo.custom ")[1:]:
        start = part.find("](")
        if start == -1:
            continue
        symbol = re.search(r'symbol = "([^"]*)"', part[:start])
        if symbol is None:
            continue
        operands = part[start + 2 : part.index(")", start + 2)]
        ops.append((symbol.group(1), operands.split(", ")))
    return ops


def _subgraph_body(text: str, name: str) -> str:
    """The body of the one subgraph named ``name``, without its header."""
    header = re.search(rf"mo\.graph @{name}(?!\w)", text)
    assert header is not None, f"no mo.graph @{name} in the staged graph"
    rest = text[header.end() :]
    next_graph = re.search(r"mo\.graph @", rest)
    return rest[: next_graph.start()] if next_graph else rest


def _tail_start(input_types: Sequence[Any]) -> int:
    """The index of the first speculative-tail input.

    The tail opens with the draft tokens, the one ``[batch_size, num_steps]``
    int64 input.
    """
    for index, input_type in enumerate(input_types):
        if input_type == TensorType(
            DType.int64, ["batch_size", "num_steps"], device=DeviceRef.GPU(0)
        ):
            return index
    raise AssertionError("no draft-tokens input in the fused graph")


@pytest.fixture(autouse=True)
def _virtual_gpu() -> None:
    """One virtual B200, so the trace needs no GPU."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)


def test_hisparse_off_builds_today_graph() -> None:
    """Without HiSparse the fused graph takes no new input and no new op."""
    staged, input_types, _ = _fuse(hisparse=None)
    assert [value.type for value in staged.graph.inputs] == input_types
    assert _symbol_count(staged, "kv_cache.hisparse.resolve.paged") == 0
    assert _symbol_count(staged, "kv_cache.hisparse.gather_group.paged") == 0


def test_hisparse_on_inserts_its_inputs_ahead_of_the_tail() -> None:
    """With HiSparse the three input groups sit right before the tail.

    The host and hot tables are per-device device tensors staged each step
    like the MLA lookup table, whose page-count dim the host table shares,
    and the host mirror is one CPU tensor. They follow the batch context
    lengths -- the EP inputs follow those, and none are declared here -- and
    precede the speculative tail, so an export without them keeps every
    tail slot.
    """
    off_staged, off_types, _ = _fuse(hisparse=None)
    on_staged, on_types, config = _fuse(hisparse=_hisparse())

    tail = _tail_start(off_types)
    assert on_types[:tail] == off_types[:tail]
    assert on_types[tail + 3 :] == off_types[tail:]
    assert on_types[tail : tail + 3] == config.hisparse_input_types()
    host_table, hot_table, host_mirror = on_types[tail : tail + 3]
    assert host_table == TensorType(
        DType.int32,
        ["replica_0_batch_size", "replica_0_mla_max_num_pages"],
        device=DeviceRef.GPU(0),
    )
    assert hot_table == TensorType(
        DType.uint32,
        ["replica_0_batch_size", _hisparse().hot_pages + 1],
        device=DeviceRef.GPU(0),
    )
    assert host_mirror == TensorType(
        DType.uint8, ["hisparse_mirror_bytes"], device=DeviceRef.CPU()
    )
    assert [value.type for value in on_staged.graph.inputs] == on_types

    text = str(on_staged)
    # The target's two full MoE layers share one body, as they do without
    # HiSparse, and so does the one shared layer.
    assert len(re.findall(r"mo\.graph @moe_block_full(?!\w)", text)) == 1
    assert len(re.findall(r"mo\.call @moe_block_full(?!\w)", text)) == 2
    assert len(re.findall(r"mo\.graph @moe_block_shared(?!\w)", text)) == 1
    assert len(re.findall(r"mo\.call @moe_block_shared(?!\w)", text)) == 1

    # Every full target layer resolves -- the dense layer inline, the two
    # full MoE layers through the one body they share. The gather is grouped:
    # the dense layer fills its own and the shared follower's rows in one
    # launch, and the full MoE layers' body gathers itself, so three launches
    # cover four layers. The counts are the target's alone: the NextN draft's
    # layer stays fully resident, so one more of either would mean the draft
    # read the tables it never takes.
    assert _symbol_count(on_staged, "kv_cache.hisparse.resolve.paged") == 2
    assert _symbol_count(on_staged, "kv_cache.hisparse.gather_group.paged") == 2
    assert _symbol_count(off_staged, "kv_cache.hisparse.resolve.paged") == 0

    # The body the full MoE layers share resolves once, and its group operand
    # is the body's argument for the layer's cache slot, not a constant baked
    # into the body. The gather's width is an argument too: the two full MoE
    # layers of a long schedule differ in how many shared layers follow them.
    body = _subgraph_body(text, "moe_block_full")
    resolves = [
        operands
        for symbol, operands in _custom_ops(body)
        if symbol == "kv_cache.hisparse.resolve.paged"
    ]
    assert len(resolves) == 1
    assert re.fullmatch(r"%arg\d+", resolves[0][8]), resolves[0]
    gathers = [
        operands
        for symbol, operands in _custom_ops(body)
        if symbol == "kv_cache.hisparse.gather_group.paged"
    ]
    assert len(gathers) == 1
    for operand in gathers[0][4:6]:
        assert re.fullmatch(r"%arg\d+", operand), gathers[0]


def test_hisparse_follows_the_ep_inputs_in_the_signature() -> None:
    """The three inputs sit between the EP inputs and the tail, by name.

    The staged build above runs EP-free; this pins the position against a
    signature that declares EP inputs, the shape the TP8/EP8 export takes,
    without staging the EP ops themselves. The decode reads the group back
    by name, so the values the target's verify receives are the ones the
    signature put there.
    """
    devices = [DeviceRef.GPU(0)]
    ep_input_types = [
        TensorType(DType.uint64, ["ep_signal"], device=devices[0]),
        TensorType(DType.uint64, ["ep_counter"], device=devices[0]),
    ]
    resident = _config(None)
    kv_params = _unified_kv(resident, _draft_config(resident))
    hisparse_types = _config(_hisparse()).hisparse_input_types()
    spec = SpecDecodeInputTypeSpec(devices=devices, distributed=True)
    off_types = list(
        build_spec_decode_input_types(
            spec, kv_params=kv_params, ep_input_types=ep_input_types
        )
    )
    on_types = list(
        build_spec_decode_input_types(
            spec,
            kv_params=kv_params,
            ep_input_types=ep_input_types,
            hisparse_input_types=hisparse_types,
        )
    )

    tail = _tail_start(off_types)
    assert on_types[: tail - 2] == off_types[: tail - 2]
    assert on_types[tail - 2 : tail] == ep_input_types
    assert on_types[tail : tail + 3] == hisparse_types
    assert on_types[tail + 3 :] == off_types[tail:]

    with Graph("decode", input_types=on_types) as graph:
        decoded = decode_spec_decode_input_values(
            graph.inputs,
            spec,
            kv_params=kv_params,
            ep_input_types=ep_input_types,
            hisparse_input_types=hisparse_types,
        )
    assert decoded.ep_inputs == list(graph.inputs[tail - 2 : tail])
    assert decoded.hisparse == list(graph.inputs[tail : tail + 3])
