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
"""The fused GLM-5.3-Flash MTP graph's signature and its state ops.

The graph is built through ``build_spec_graph``, the code a served pairing
runs, over a tiny random-shape config. The state rides in the cache tree, so
the verify ring adds one pool and one row table per device to the cache group
and nothing past the canonical tail. The graph is staged, not compiled, so it
needs none of the new kernels. The base graph is staged beside it to pin that
it runs none of the speculative state ops.

Staging constructs ``Allreduce``, which instantiates an ``Accelerator``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.nn.legacy.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
    RecurrentStateParams,
)
from max.nn.legacy.layer import Module
from max.nn.legacy.quant_config import QuantConfig
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.glm5_next.glm5_next import Glm5Next
from max.pipelines.architectures.glm5_next.layers import kimi_delta_attention
from max.pipelines.architectures.glm5_next.model_config import Glm5NextConfig
from max.pipelines.architectures.glm5_next.mtp import Glm5NextMTP
from max.pipelines.architectures.glm5_next.quantization import (
    Glm5NextQuantScheme,
)
from max.pipelines.architectures.glm5_next.state_cache import (
    INDEXER_CACHE_KEY,
    MLA_CACHE_KEY,
    STATE_CACHE_KEY,
)
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecGraph,
    _with_num_draft_tokens,
    build_spec_graph,
)
from max.pipelines.speculative import SpeculativeConfig, spec_adapters

TARGET_ARCH = "Glm5NextForConditionalGeneration"
NUM_DRAFTS = 3
LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "deepseek_sparse_attention",
]
NUM_KDA = LAYER_TYPES.count("linear_attention")
NUM_SPARSE = LAYER_TYPES.count("deepseek_sparse_attention")
HIDDEN = 256
HEADS = 4
KDA_HEADS = 2
HEAD_DIM = 128


def _speculative() -> SpeculativeConfig:
    return SpeculativeConfig(
        speculative_method="mtp", num_speculative_tokens=NUM_DRAFTS
    )


def _kv_params(*, speculative: bool, num_devices: int) -> MultiKVCacheParams:
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = _speculative() if speculative else None
    params = Glm5NextConfig.construct_kv_params(
        huggingface_config=SimpleNamespace(
            num_hidden_layers=len(LAYER_TYPES),
            num_attention_heads=HEADS,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            index_head_dim=128,
            index_kpool=4,
            index_topk=64,
            num_nextn_predict_layers=1,
            layer_types=LAYER_TYPES,
            linear_attn_config={
                "num_heads": KDA_HEADS,
                "head_dim": HEAD_DIM,
                "short_conv_kernel_size": 4,
            },
        ),
        pipeline_config=pipeline_config,
        devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        kv_cache_config=KVCacheConfig(state_pool_dtype="float32"),
        cache_dtype=DType.bfloat16,
    )
    assert isinstance(params, MultiKVCacheParams)
    return params


def _without_ring(params: MultiKVCacheParams) -> MultiKVCacheParams:
    """Returns ``params`` with the state child's last leaf, the ring, cut."""
    state = params.children[STATE_CACHE_KEY]
    assert isinstance(state, RecurrentStateParams)
    return MultiKVCacheParams.from_params(
        {
            **params.children,
            STATE_CACHE_KEY: replace(state, regions=state.regions[:-1]),
        }
    )


def _config(kv_params: MultiKVCacheParams) -> Glm5NextConfig:
    """A GLM-5.3-Flash config small enough to stage the whole graph."""
    num_layers = len(LAYER_TYPES)
    all_layers = frozenset(range(num_layers + 1))
    sparse_layers = {
        i for i, kind in enumerate(LAYER_TYPES) if kind != "linear_attention"
    }
    # The MTP layer at ``num_layers`` is sparse MLA too.
    attn_layers = frozenset({*sparse_layers, num_layers})
    quant = QuantConfig.blockscaled_fp8(
        mlp_quantized_layers=all_layers,
        attn_quantized_layers=attn_layers,
        embedding_output_dtype=DType.bfloat16,
    )
    return Glm5NextConfig(
        dtype=DType.float8_e4m3fn,
        kv_params=kv_params,
        devices=list(kv_params.devices),
        use_subgraphs=False,
        max_seq_len=256,
        max_position_embeddings=256,
        vocab_size=512,
        hidden_size=HIDDEN,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=128,
        qk_rope_head_dim=0,
        qk_nope_head_dim=128,
        v_head_dim=128,
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        correction_bias_dtype=DType.float32,
        index_head_dim=128,
        # The checkpoint's count; gfx950's pooled scorer takes multiples of 32.
        index_n_heads=32,
        index_topk=64,
        indexer_types=["full"] * num_layers,
        layer_types=list(LAYER_TYPES),
        linear_num_heads=KDA_HEADS,
        linear_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=4,
        mla_latent_pad_to=576,
        kv_b_proj_dtype=DType.bfloat16,
        declared_dtype=DType.bfloat16,
        quant_config=quant,
        quant_scheme=Glm5NextQuantScheme(
            config=quant,
            mlp_layers=all_layers,
            attn_layers=attn_layers,
            compute_dtype=DType.bfloat16,
        ),
    )


def _name_weights(module: Module) -> None:
    """Gives every weight its full name, which a load would otherwise do.

    Without it the second sibling weight staged raises ``Weight 'weight'
    already exists``.
    """
    for name, weight in module.raw_state_dict().items():
        weight.name = name


def _zeros(module: Module) -> dict[str, Buffer]:
    return {
        name: Buffer.zeros(shape=w.shape.static_dims, dtype=w.dtype)
        for name, w in module.raw_state_dict().items()
    }


def _symbol_count(graph: Graph, symbol: str) -> int:
    return len(re.findall(rf'symbol\s*=\s*"{re.escape(symbol)}"', str(graph)))


def _pipeline_config() -> PipelineConfig:
    """What ``build_spec_graph`` reads of a served pipeline config."""
    text_config = SimpleNamespace(hidden_size=HIDDEN, vocab_size=512)
    return cast(
        PipelineConfig,
        SimpleNamespace(
            models=SimpleNamespace(main_architecture_name=TARGET_ARCH),
            draft_model=None,
            speculative=_speculative(),
            needs_bitmask_constraints=False,
            runtime=SimpleNamespace(
                reasoning_parser=None, prefer_module_v3=False
            ),
            model=SimpleNamespace(
                data_parallel_degree=1,
                huggingface_config=SimpleNamespace(
                    get_text_config=lambda: text_config
                ),
                pipeline_stage=None,
            ),
        ),
    )


def _stage_fused(load_weights: bool = False) -> SpecGraph:
    """Builds the fused graph the way a served pairing does.

    With ``load_weights`` every weight is zeros, enough to compile.
    """
    register_all_models()
    kv_params = _kv_params(speculative=True, num_devices=1)
    target_config = _config(kv_params)
    target_config.return_logits = ReturnLogits.VARIABLE
    target_config.return_hidden_states = ReturnHiddenStates.ALL_NORMALIZED
    target = Glm5Next(target_config, kv_params=kv_params)
    target.attach_layers(target.build_layers())
    draft_config = _config(kv_params)
    draft = Glm5NextMTP(draft_config)
    draft_weights = None
    if load_weights:
        target.load_state_dict(_zeros(target), weight_alignment=1)
        draft_weights = _zeros(draft)
    # The draft embeds and projects through the target's embedding and head.
    for slot in ("embed_tokens", "lm_head"):
        setattr(draft, slot, getattr(target, slot))
    return build_spec_graph(
        _pipeline_config(),
        MultiKVCacheParams.from_params(
            {"target": _with_num_draft_tokens(kv_params, NUM_DRAFTS)}
        ),
        list(kv_params.devices),
        include_signal_buffers=True,
        target=target,
        target_config=target_config,
        draft=draft,
        draft_config=draft_config,
        hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        draft_weights=draft_weights,
    )


def _stage_base() -> Graph:
    """Stages the base graph the way ``Glm5NextModel`` does."""
    kv_params = _kv_params(speculative=False, num_devices=1)
    model = Glm5Next(_config(kv_params))
    model.attach_layers(model.build_layers())
    _name_weights(model)
    with Graph("glm5_next", input_types=model.input_types(kv_params)) as g:
        (
            tokens,
            image_rows,
            image_indices,
            offsets,
            host_offsets,
            return_n_logits,
            _,
            *rest,
        ) = g.inputs
        it = iter(rest)
        signals = [next(it).buffer]
        kv_tree = kv_params.unflatten_kv_inputs(it)
        assert isinstance(kv_tree, dict)
        next(it)
        g.output(
            model(
                tokens.tensor,
                [image_rows.tensor],
                [image_indices.tensor],
                signals,
                tree.leaves(
                    kv_tree[MLA_CACHE_KEY], leaf=KVCacheInputsPerDevice
                ),
                tree.leaves(
                    kv_tree[INDEXER_CACHE_KEY], leaf=KVCacheInputsPerDevice
                ),
                return_n_logits.tensor,
                offsets.tensor,
                host_offsets.tensor,
                tree.leaves(
                    kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
                ),
            )
        )
    return g


@pytest.fixture(scope="module")
def fused() -> Graph:
    return _stage_fused().graph


@pytest.fixture(scope="module")
def base() -> Graph:
    return _stage_base()


def _kv_count(params: KVCacheParamInterface) -> int:
    return len(params.flattened_kv_inputs())


@pytest.mark.parametrize("num_devices", [1, 2])
def test_the_ring_adds_a_pool_and_a_row_table_per_device(
    num_devices: int,
) -> None:
    """Checks the state child's arity: the ring is the one leaf it adds."""
    spec = _kv_params(speculative=True, num_devices=num_devices)
    assert _kv_count(spec) - _kv_count(_without_ring(spec)) == 2 * num_devices


def test_the_state_rides_in_the_cache_tree() -> None:
    """Checks the graph declares the state among its KV inputs, so the
    target's call takes nothing past the canonical tail."""
    config = _config(_kv_params(speculative=True, num_devices=1))
    assert spec_adapters.tail_kv(TARGET_ARCH) is None
    assert not spec_adapters.trailing_input_types(TARGET_ARCH, "mtp", config)


def test_the_verify_records_every_kda_layer_into_the_ring(
    fused: Graph,
) -> None:
    assert _symbol_count(fused, "kda_verify_ring") == NUM_KDA
    assert _symbol_count(fused, "kda_decode") == 0
    # One launch per device folds every layer.
    assert _symbol_count(fused, "kda_state_fold") == 1


@pytest.mark.parametrize("servable", [True, False])
def test_only_a_draft_free_verify_takes_kda_chunk(
    monkeypatch: pytest.MonkeyPatch, servable: bool
) -> None:
    """Checks ``kda_chunk`` is staged beside the ring, never in its place.

    Staged as if on a GPU that runs ``kda_chunk`` and one that does not. Where
    it runs, each KDA layer stages it once, as the draft-free branch of a
    verify whose other branch records into the ring.
    """
    monkeypatch.setattr(
        kimi_delta_attention,
        "kda_chunk_servable",
        lambda head_dim, gate_mode, pool_dtype: servable,
    )
    graph = _stage_fused().graph
    assert _symbol_count(graph, "kda_chunk") == (NUM_KDA if servable else 0)
    assert _symbol_count(graph, "kda_verify_ring") == NUM_KDA
    assert _symbol_count(graph, "kda_decode") == 0


def test_the_conv_window_is_left_for_the_rollback(fused: Graph) -> None:
    # The verify's launch and the rollback's, per KDA layer.
    assert _symbol_count(fused, "gated_delta_conv1d_verify_fwd") == 2 * NUM_KDA
    assert _symbol_count(fused, "gated_delta_conv1d_fwd") == 0


def test_every_indexer_seeds_its_tail_twice(fused: Graph) -> None:
    """Checks each indexer over the window, the draft's included, gets a
    deferred seed and a replay."""
    indexers = NUM_SPARSE + 1
    assert _symbol_count(fused, "mo.mla.kpool.seed_tail") == 2 * indexers


def test_the_fused_graph_compiles() -> None:
    """Checks the fused graph compiles, the new state kernels included."""
    spec = _stage_fused(load_weights=True)
    session = InferenceSession(devices=[Accelerator()])
    assert (
        session.load(spec.graph, weights_registry=spec.weights_registry)
        is not None
    )


def test_the_base_graph_runs_no_speculative_state_op(base: Graph) -> None:
    for symbol in (
        "kda_verify_ring",
        "kda_state_fold",
        "gated_delta_conv1d_verify_fwd",
    ):
        assert _symbol_count(base, symbol) == 0, symbol
    assert _symbol_count(base, "gated_delta_conv1d_fwd") == NUM_KDA
    assert _symbol_count(base, "mo.mla.kpool.seed_tail") == NUM_SPARSE
