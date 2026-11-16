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
"""The GLM-5.x ModuleV3 graph's Hybrid HiSparse wiring, staged without a GPU.

The whole decoder is staged on a virtual GPU, the no-GPU lane the EP
construction tests use, over a tiny random config: twice, with HiSparse off
and on. Off, the graph must be exactly today's: the same inputs and none of
the HiSparse ops. On, the three HiSparse input groups follow the EP tail,
every ``full`` layer resolves its selections and gathers the rows of the
consecutive layers that reuse them in one launch, and the resolve keeps one
shared MoE subgraph body per indexer kind by taking the layer's group as a
runtime input rather than a constant baked into the body.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from max import functional as F
from max import tree
from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn._graph_ops import HiSparseParams
from max.nn.legacy.graph_groups import BatchContext
from max.nn.legacy.quant_config import QuantConfig
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    hisparse_gather_groups,
    resolve_indexer_types,
)
from max.pipelines.architectures.glm5_1_modulev3.glm5_1 import Glm5_1Decoder
from max.pipelines.architectures.glm5_1_modulev3.model_config import (
    Glm5_1Config,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.module_registration import (
    _DecoderGraph,
    decoder_input_types,
)
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh

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
    return HiSparseParams.for_top_k(
        top_k=_INDEX_TOPK,
        query_rows=1,
        page_size=_PAGE_SIZE,
        num_groups=_FULL_LAYERS,
    )


def _config(
    hisparse: HiSparseParams | None,
    *,
    data_parallel_degree: int = 1,
    cache_dtype: DType = DType.float8_e4m3fn,
) -> Glm5_1Config:
    """A tiny GLM config, placed on one virtual GPU per replica.

    Raises:
        ValueError: As :class:`Glm5_1Config` does, for a HiSparse deployment
            this config does not support.
    """
    devices = [DeviceRef.GPU(i) for i in range(data_parallel_degree)]
    hf = _huggingface_config()
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = data_parallel_degree
    pipeline_config.speculative = None
    config = Glm5_1Config(
        dtype=cache_dtype,
        kv_params=Glm5_1Config.construct_kv_params(
            huggingface_config=hf,
            pipeline_config=cast(Any, pipeline_config),
            devices=devices,
            kv_cache_config=KVCacheConfig(),
            cache_dtype=cache_dtype,
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
        data_parallel_degree=data_parallel_degree,
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
    # The runtime places the config on the deployment's mesh; one tensor-
    # parallel axis per replica is what a HiSparse deployment runs.
    config.mesh = DeviceMesh(
        tuple(d.to_device() for d in devices),
        (data_parallel_degree,),
        ("dp" if data_parallel_degree > 1 else "tp",),
    )
    return config


def _stage(config: Glm5_1Config) -> tuple[Any, list[TensorType | BufferType]]:
    """Stages the registered decoder graph used by ModuleV3 exports."""
    decoder_inputs = decoder_input_types(
        config.kv_params,
        config.devices[0],
        batch_context_lengths=BatchContext(
            length=(TensorType(DType.int32, [1], device=DeviceRef.CPU()),)
        ),
        hisparse=(
            config.hisparse_input_types() if config.hisparse is not None else ()
        ),
    )
    inputs = tree.leaves(decoder_inputs)
    assert config.mesh is not None
    with (
        F.lazy(),
        default_device(config.mesh),
        default_dtype(DType.bfloat16),
    ):
        model = _DecoderGraph(Glm5_1Decoder(config))
        staged, _registry = model.stage(decoder_inputs)
    return staged, inputs


def _input_types(staged: Any) -> list[Any]:
    return [value.type for value in staged.graph.inputs]


def _symbol_count(staged: Any, symbol: str) -> int:
    return len(re.findall(rf'symbol\s*=\s*"{re.escape(symbol)}"', str(staged)))


def _custom_ops(text: str) -> list[tuple[str, list[str]]]:
    """Each custom op's symbol and operands, in print order.

    An op prints as `mo.custom [symbol = "...", ...](operands) : (...)`: the
    attribute list ends at the first `](` and the operand list, free of
    parentheses, at the `)` after it.
    """
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


@pytest.fixture(autouse=True)
def _virtual_gpu() -> None:
    """One virtual B200, so the trace needs no GPU."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)


def test_hisparse_off_builds_today_graph() -> None:
    """Without HiSparse the graph takes no new input and runs no new op."""
    staged, inputs = _stage(_config(hisparse=None))
    assert _input_types(staged) == inputs
    assert _symbol_count(staged, "kv_cache.hisparse.resolve.paged") == 0
    assert _symbol_count(staged, "kv_cache.hisparse.gather_group.paged") == 0


def test_hisparse_on_appends_its_inputs_and_runs_its_ops() -> None:
    """With HiSparse the three input groups follow the EP tail, and the ops run.

    The host and hot tables are per-device device tensors staged each step
    like the MLA lookup table, whose page-count dim the host table shares, and
    the host mirror is one CPU tensor. Every ``full`` layer gathers the
    misses of its own and its consecutive shared followers in one launch, and
    every ``full`` layer resolves its selections, the
    dense layer inline and the MoE layers through the subgraph bodies they
    share. The resolve's group operand is the body's argument, not a
    constant, so the two full MoE layers keep their own LRU states while
    sharing one body.
    """
    hisparse = _hisparse()
    config = _config(hisparse=hisparse)
    _, off_inputs = _stage(_config(hisparse=None))
    staged, inputs = _stage(config)

    # The three input groups, in order, are the graph's last inputs: with EP
    # off they are the only extra inputs, and nothing (no signal buffer on a
    # single device) follows them.
    assert _input_types(staged) == inputs
    assert len(inputs) == len(off_inputs) + 3
    assert inputs[-3:] == config.hisparse_input_types()
    with_ep = decoder_input_types(
        config.kv_params,
        config.devices[0],
        ep=(TensorType(DType.int32, [1], device=DeviceRef.CPU()),),
        hisparse=config.hisparse_input_types(),
    )
    assert list(with_ep).index("hisparse") == list(with_ep).index("ep") + 1
    host_table, hot_table, host_mirror = inputs[-3:]
    assert host_table == TensorType(
        DType.int32,
        ["replica_0_batch_size", "replica_0_mla_max_num_pages"],
        device=DeviceRef.GPU(0),
    )
    assert hot_table == TensorType(
        DType.uint32,
        ["replica_0_batch_size", hisparse.hot_pages + 1],
        device=DeviceRef.GPU(0),
    )
    assert host_mirror == TensorType(
        DType.uint8, ["hisparse_mirror_bytes"], device=DeviceRef.CPU()
    )

    text = str(staged)
    # The two full MoE layers share one body, as they do without HiSparse, and
    # so does the one shared layer.
    assert len(re.findall(r"mo\.graph @moe_block_full(?!\w)", text)) == 1
    assert len(re.findall(r"mo\.call @moe_block_full(?!\w)", text)) == 2
    assert len(re.findall(r"mo\.graph @moe_block_shared(?!\w)", text)) == 1
    assert len(re.findall(r"mo\.call @moe_block_shared(?!\w)", text)) == 1

    # Every full layer resolves: the dense layer inline, and the two full MoE
    # layers through the one body they share. The dense layer's gather also
    # fills the shared layer that reuses its selections, so that layer skips
    # its own; the full MoE layers' selections are read by no other layer and
    # their body gathers itself. Two launches cover all four layers where one
    # launch per layer would take three.
    assert _symbol_count(staged, "kv_cache.hisparse.resolve.paged") == 2
    assert _symbol_count(staged, "kv_cache.hisparse.gather_group.paged") == 2

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

    # The shared layer's body gathers nothing: its rows are the full layer's.
    shared_body = _subgraph_body(text, "moe_block_shared")
    assert "kv_cache.hisparse.gather" not in shared_body


def test_gather_group_counts_cover_each_layer_once() -> None:
    """One launch fills a full layer and every shared layer after it.

    The tiny schedule's first full layer gathers itself and the shared layer
    that reuses it; the two full layers after it have no followers, so their
    own launch fills only themselves. Every layer is therefore filled exactly
    once, and no shared layer needs a launch of its own.
    """
    assert hisparse_gather_groups(_INDEXER_TYPES, _LAYERS) == [2, 1, 1, 1]


def test_gather_group_counts_cover_glm_5_3() -> None:
    """GLM-5.3's 78 layers are three singles and 18 groups of four.

    Its schedule is three full layers, then one full every four; each full
    layer from layer 2 on is followed by three shared ones, so a single
    grouped gather serves each block of four and the totals add up to the
    whole stack -- no layer is left ungathered.
    """
    schedule = resolve_indexer_types(
        SimpleNamespace(
            indexer_types=None,
            index_topk_pattern=None,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        ),
        78,
    )
    counts = hisparse_gather_groups(schedule, 78)
    assert counts[:6] == [1, 1, 4, 1, 1, 1]
    assert [i for i, n in enumerate(counts) if n != 1] == [
        2 + 4 * k for k in range(19)
    ]
    assert all(counts[i] == 4 for i in range(2, 78, 4))
    # Only full layers launch, so only their counts add up, to the whole
    # stack: every layer is filled by exactly one launch.
    assert sum(n for i, n in enumerate(counts) if schedule[i] == "full") == 78


def test_hisparse_refuses_an_nvfp4_latent_cache() -> None:
    """An NVFP4 latent's scale rows are not mirrored yet."""
    with pytest.raises(ValueError, match="NVFP4"):
        _config(_hisparse(), cache_dtype=DType.uint8)


def test_hisparse_tables_follow_each_data_parallel_replica() -> None:
    """Under data parallelism each replica stages its own rows' tables.

    The host and hot tables take their replica's lookup-table dims on that
    replica's device, and the one host mirror stays on the CPU.
    """
    hisparse = _hisparse()
    config = _config(hisparse, data_parallel_degree=2)
    types = config.hisparse_input_types()
    assert len(types) == 2 * 2 + 1
    for r in range(2):
        assert types[r] == TensorType(
            DType.int32,
            [f"replica_{r}_batch_size", f"replica_{r}_mla_max_num_pages"],
            device=DeviceRef.GPU(r),
        )
        assert types[2 + r] == TensorType(
            DType.uint32,
            [f"replica_{r}_batch_size", hisparse.hot_pages + 1],
            device=DeviceRef.GPU(r),
        )
    assert types[-1] == TensorType(
        DType.uint8, ["hisparse_mirror_bytes"], device=DeviceRef.CPU()
    )
