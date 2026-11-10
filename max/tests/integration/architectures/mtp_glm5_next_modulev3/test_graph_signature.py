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
"""The fused ModuleV3 GLM-5.3-Flash MTP graph's state ops.

The fused module is the one a served pairing compiles, built over a tiny
random-shape config and traced. Its verify records every KDA layer into the
verify ring, leaves every conv window and tail seed to the rollback, and only
a draft-free window may take ``kda_chunk``. Tracing constructs the
accelerator the mesh names.
"""

from __future__ import annotations

import re
from functools import partial
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from max import functional as F
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.outputs import ReturnHiddenStates
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.glm5_next.model_config import (
    Glm5NextConfig as GraphGlm5NextConfig,
)
from max.pipelines.architectures.glm5_next.quantization import (
    Glm5NextQuantScheme,
)
from max.pipelines.architectures.glm5_next_modulev3.glm5_next import Glm5Next
from max.pipelines.architectures.glm5_next_modulev3.layers import kda
from max.pipelines.architectures.glm5_next_modulev3.model_config import (
    Glm5NextConfig,
)
from max.pipelines.architectures.glm5_next_modulev3.mtp import Glm5NextMTP
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _driver_call,
    _graph_name,
    _Pairing,
    _SpecDecodeModule,
    _with_num_draft_tokens,
)
from max.pipelines.speculative import SpeculativeConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh

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
VOCAB = 512


def _speculative() -> SpeculativeConfig:
    return SpeculativeConfig(
        speculative_method="mtp", num_speculative_tokens=NUM_DRAFTS
    )


def _kv_params() -> MultiKVCacheParams:
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = _speculative()
    params = GraphGlm5NextConfig.construct_kv_params(
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
        devices=[DeviceRef.GPU()],
        kv_cache_config=KVCacheConfig(state_pool_dtype="float32"),
        cache_dtype=DType.bfloat16,
    )
    stamped = _with_num_draft_tokens(params, NUM_DRAFTS)
    assert isinstance(stamped, MultiKVCacheParams)
    return stamped


def _config(kv_params: MultiKVCacheParams, mesh: DeviceMesh) -> Glm5NextConfig:
    """A GLM-5.3-Flash config small enough to trace the whole graph."""
    num_layers = len(LAYER_TYPES)
    all_layers = frozenset(range(num_layers + 1))
    sparse = {
        i for i, kind in enumerate(LAYER_TYPES) if kind != "linear_attention"
    }
    # The MTP layer at ``num_layers`` is sparse MLA too.
    attn_layers = frozenset({*sparse, num_layers})
    quant = QuantConfig.blockscaled_fp8(
        mlp_quantized_layers=all_layers,
        attn_quantized_layers=attn_layers,
        embedding_output_dtype=DType.bfloat16,
    )
    return Glm5NextConfig(
        dtype=DType.float8_e4m3fn,
        kv_params=kv_params,
        devices=list(kv_params.devices),
        max_seq_len=256,
        max_position_embeddings=256,
        vocab_size=VOCAB,
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
        graph_mode="decode",
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
        declared_dtype=DType.bfloat16,
        quant_config=quant,
        quant_scheme=Glm5NextQuantScheme(
            config=quant,
            mlp_layers=all_layers,
            attn_layers=attn_layers,
            compute_dtype=DType.bfloat16,
        ),
        mesh=mesh,
    )


def _pipeline_config() -> PipelineConfig:
    """What the pairing reads of a served config that prefers ModuleV3."""
    text_config = SimpleNamespace(hidden_size=HIDDEN, vocab_size=VOCAB)
    return cast(
        PipelineConfig,
        SimpleNamespace(
            models=SimpleNamespace(main_architecture_name=TARGET_ARCH),
            draft_model=None,
            speculative=_speculative(),
            needs_bitmask_constraints=False,
            runtime=SimpleNamespace(
                reasoning_parser=None, prefer_module_v3=True
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


def _module() -> _SpecDecodeModule:
    """The fused module the pairing compiles, on one device."""
    register_all_models()
    mesh = DeviceMesh((Accelerator(),), (1,), ("tp",))
    kv_params = _kv_params()
    config = _config(kv_params, mesh)
    with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
        target = Glm5Next(config, kv_params)
        draft = Glm5NextMTP(config)
    return _SpecDecodeModule(
        target,
        draft,
        partial(
            _driver_call,
            _pipeline_config(),
            MultiKVCacheParams.from_params({"target": kv_params}),
            list(kv_params.devices),
            include_signal_buffers=True,
            target_config=config,
            draft_config=config,
            hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        ),
    )


def _graph_name_for_test() -> str:
    register_all_models()
    return _graph_name(_Pairing.from_pipeline_config(_pipeline_config()), "mtp")


def _trace() -> str:
    module = _module()
    call = module.driver_call()
    return str(module.trace(*call.input_types, name=_graph_name_for_test()))


def _symbol_count(graph: str, symbol: str) -> int:
    return len(re.findall(rf'symbol\s*=\s*"{re.escape(symbol)}"', graph))


@pytest.fixture(scope="module")
def fused() -> str:
    return _trace()


def test_the_fused_graph_keeps_the_graph_api_name() -> None:
    """Checks engines that bind the fused graph by name find it."""
    assert _graph_name_for_test() == "glm5_next_with_mtp_graph"


def test_the_verify_records_every_kda_layer_into_the_ring(fused: str) -> None:
    assert _symbol_count(fused, "kda_verify_ring") == NUM_KDA
    assert _symbol_count(fused, "kda_decode") == 0
    # One launch per device folds every layer.
    assert _symbol_count(fused, "kda_state_fold") == 1


@pytest.mark.parametrize("servable", [True, False])
def test_only_a_draft_free_verify_takes_kda_chunk(
    monkeypatch: pytest.MonkeyPatch, servable: bool
) -> None:
    """Checks ``kda_chunk`` is traced beside the ring, never in its place."""
    monkeypatch.setattr(kda, "accelerator_api", lambda: "hip")
    monkeypatch.setattr(
        kda,
        "kda_chunk_servable",
        lambda head_dim, gate_mode, pool_dtype: servable,
    )
    graph = _trace()
    assert _symbol_count(graph, "kda_chunk") == (NUM_KDA if servable else 0)
    assert _symbol_count(graph, "kda_verify_ring") == NUM_KDA
    assert _symbol_count(graph, "kda_decode") == 0


def test_the_conv_window_is_left_for_the_rollback(fused: str) -> None:
    # Only the verify's forward launch, one per KDA layer: the rollback places
    # the windows inside the fold's launch instead.
    assert _symbol_count(fused, "gated_delta_conv1d_verify_fwd") == NUM_KDA
    assert _symbol_count(fused, "gated_delta_conv1d_fwd") == 0


def test_the_rollback_folds_the_conv_windows_in_the_folds_launch(
    fused: str,
) -> None:
    """Checks the rollback holds one fold op and no per-layer conv replay."""
    assert _symbol_count(fused, "kda_state_fold") == 1
    # One forward conv-verify per layer is all the conv-verify ops the graph
    # holds; a per-layer replay in the roll-forward would double it.
    assert _symbol_count(fused, "gated_delta_conv1d_verify_fwd") == NUM_KDA


def test_every_indexer_seeds_its_tail_twice(fused: str) -> None:
    """Checks each indexer over the window, the draft's included, gets a
    deferred seed and a replay."""
    indexers = NUM_SPARSE + 1
    assert _symbol_count(fused, "attention.mla.kpool.seed_tail") == 2 * indexers


def test_the_fused_graph_compiles() -> None:
    """Checks the fused graph compiles, the state kernels included."""
    module = _module()
    call = module.driver_call()
    weights = {
        name: Buffer.zeros(
            shape=[int(d) for d in tensor.shape], dtype=tensor.dtype
        )
        for name, tensor in module.parameters
    }
    assert isinstance(next(iter(module.parameters))[1], Tensor)
    compiled = module.compile(
        *call.input_types, weights=weights, name=_graph_name_for_test()
    )
    assert compiled is not None
