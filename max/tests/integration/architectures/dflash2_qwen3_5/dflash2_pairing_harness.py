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


"""A tiny Qwen3.5 target beside a tiny DFlash2 draft, fused as served.

The pairing tests share this. It builds the fused graph through
``build_spec_graph``, the code a served pairing runs, over modules small
enough to compile in seconds, so the graph under test is the production one
and not a restatement of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
    recurrent_leaf,
)
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.dflash2_qwen3_5 import DFlash2Qwen3_5
from max.pipelines.architectures.dflash2_qwen3_5.model_config import (
    DRAFT_SLIDING_WINDOW,
    Dflash2DraftHFConfig,
    DFlash2Qwen3_5Config,
)
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.qwen3_5 import Qwen3_5
from max.pipelines.architectures.qwen3_5.state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    linear_state_regions,
    ring_len_for_window,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _with_num_draft_tokens,
    build_spec_graph,
)
from max.pipelines.speculative import SpeculativeConfig
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

TARGET_ARCH = "Qwen3_5ForConditionalGeneration"
DRAFT_ARCH = "DFlash2DraftModel"

CONV_KERNEL = 4
SELECTOR_TOP_K = 3
KV_HEAD_DIM = 16
FULL = "full_attention"
LINEAR = "linear_attention"


@dataclass(frozen=True)
class Geometry:
    """The sizes of a tiny target, its draft and the block between them."""

    layer_types: tuple[str, ...] = (LINEAR, FULL) * 2
    draft_layers: int = 2
    block: int = 8
    hidden: int = 64
    vocab: int = 128
    page_size: int = 32
    max_seq_len: int = 4096
    draft_kv_heads: int = 2
    draft_head_dim: int = 16
    dtype: DType = DType.float32
    device: DeviceRef = field(default_factory=DeviceRef.GPU)

    @property
    def num_linear(self) -> int:
        """Returns the target's linear-attention layer count."""
        return self.layer_types.count(LINEAR)

    @property
    def num_full(self) -> int:
        """Returns the target's full-attention layer count."""
        return self.layer_types.count(FULL)

    @property
    def tap_ids(self) -> list[int]:
        """Returns the target layers the draft reads: every one."""
        return list(range(len(self.layer_types)))


def target_config(geometry: Geometry) -> Qwen3_5Config:
    """Returns a target config whose cache holds its state and a block-long
    ring, set up to return the layers its draft reads."""
    g = geometry
    attn = MHAKVCacheParams(
        dtype=g.dtype,
        devices=[g.device],
        n_kv_heads=1,
        head_dim=KV_HEAD_DIM,
        num_layers=g.num_full,
        page_size=g.page_size,
    )
    config = Qwen3_5Config(
        hidden_size=g.hidden,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=len(g.layer_types),
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=g.max_seq_len,
        intermediate_size=g.hidden * 2,
        interleaved_rope_weights=True,
        vocab_size=g.vocab,
        dtype=g.dtype,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=attn,
        norm_dtype=g.dtype,
        rms_norm_eps=1e-6,
        attention_multiplier=KV_HEAD_DIM**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[g.device],
        clip_qkv=None,
        layer_types=list(g.layer_types),
        # The recurrence kernel is only compiled for 128 x 128 heads.
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=CONV_KERNEL,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
        state_pool_dtype=g.dtype,
    )
    config.kv_params = MultiKVCacheParams.from_params(
        {
            ATTN_CACHE_KEY: attn,
            STATE_CACHE_KEY: RecurrentStateParams(
                devices=[g.device],
                data_parallel_degree=1,
                regions=linear_state_regions(
                    num_linear_layers=g.num_linear,
                    key_head_dim=config.linear_key_head_dim,
                    num_key_heads=config.linear_num_key_heads,
                    value_head_dim=config.linear_value_head_dim,
                    num_value_heads=config.linear_num_value_heads,
                    conv_kernel_dim=config.linear_conv_kernel_dim,
                    dtype=config.state_dtype,
                    num_devices=1,
                    ring_len=ring_len_for_window(g.block),
                ),
            ),
        }
    )
    config.return_logits = ReturnLogits.VARIABLE
    config.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
    config.target_layer_ids = g.tap_ids
    return config


def draft_kv_params(geometry: Geometry) -> MHAKVCacheParams:
    """Returns the draft's windowed KV leaf."""
    g = geometry
    return MHAKVCacheParams(
        dtype=g.dtype,
        devices=[g.device],
        n_kv_heads=g.draft_kv_heads,
        head_dim=g.draft_head_dim,
        num_layers=g.draft_layers,
        page_size=g.page_size,
        window_size=DRAFT_SLIDING_WINDOW,
    )


def draft_config(geometry: Geometry) -> DFlash2Qwen3_5Config:
    """Returns the config the draft's module is built from."""
    g = geometry
    llama = Llama3Config(
        hidden_size=g.hidden,
        num_attention_heads=4,
        num_key_value_heads=g.draft_kv_heads,
        num_hidden_layers=g.draft_layers,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=g.max_seq_len,
        intermediate_size=g.hidden * 2,
        interleaved_rope_weights=False,
        vocab_size=g.vocab,
        dtype=g.dtype,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=draft_kv_params(g),
        rms_norm_eps=1e-6,
        attention_multiplier=g.draft_head_dim**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[g.device],
        clip_qkv=None,
        sliding_window=DRAFT_SLIDING_WINDOW,
    )
    return DFlash2Qwen3_5Config(
        llama,
        Dflash2DraftHFConfig(
            mask_token_id=g.vocab - 1,
            target_layer_ids=g.tap_ids,
            block_size=g.block,
            num_target_layers=None,
            conv_kernel_size=2,
            conv_group_size=8,
            selector_rank=4,
            selector_top_k=SELECTOR_TOP_K,
        ),
    )


def kv_tree(geometry: Geometry, target: Qwen3_5Config) -> MultiKVCacheParams:
    """Returns the tree a served pairing allocates: the target's cache, state
    included, beside the draft's leaf, each stamped with the draft width."""
    width = geometry.block - 1
    assert isinstance(target.kv_params, MultiKVCacheParams)
    return MultiKVCacheParams.from_params(
        {
            "target": _with_num_draft_tokens(target.kv_params, width),
            "draft": _with_num_draft_tokens(draft_kv_params(geometry), width),
        }
    )


def state_regions(target: Qwen3_5Config) -> tuple[RecurrentStateRegion, ...]:
    """Returns the target's state regions: conv, recurrence, then the ring."""
    state = recurrent_leaf(target.kv_params)
    assert state is not None
    return tuple(state.regions)


def _load_random_weights(module: Module, seed: int) -> None:
    rng = np.random.default_rng(seed)
    module.load_state_dict(
        {
            name: (
                rng.standard_normal(
                    [int(d) for d in weight.shape], dtype=np.float32
                )
                * 0.05
            )
            for name, weight in module.raw_state_dict().items()
        },
        weight_alignment=1,
        strict=False,
        override_quantization_encoding=True,
    )


class FusedGraph(NamedTuple):
    """The graph a pairing compiles and what it was built from."""

    graph: Graph
    weights_registry: dict[str, Any]
    input_spec: SpecDecodeInputTypeSpec
    target: Qwen3_5Config
    kv_tree: MultiKVCacheParams


def fused_graph(
    geometry: Geometry, *, structured_output: bool = True
) -> FusedGraph:
    """Builds the fused DFlash2 graph over a tiny Qwen3.5 pairing.

    Args:
        geometry: The sizes of the pairing.
        structured_output: Whether the pipeline needs the constrained-decoding
            bitmask, as a served Qwen3.5 pairing always does.

    Raises:
        ValueError: If the pairing can't be built under those settings.
    """
    g = geometry
    speculative = SpeculativeConfig(
        speculative_method="dflash2", num_speculative_tokens=g.block - 1
    )
    target_cfg = target_config(g)
    draft_cfg = draft_config(g)
    dflash = draft_cfg.dflash
    target = Qwen3_5(target_cfg)
    draft = DFlash2Qwen3_5(
        draft_cfg.llama,
        num_context_features=len(draft_cfg.target_layer_ids),
        block_size=dflash.block_size,
        conv_kernel_size=dflash.conv_kernel_size,
        conv_group_size=dflash.conv_group_size,
        selector_rank=dflash.selector_rank,
        selector_top_k=dflash.selector_top_k,
        layer_types=["sliding_attention"] * g.draft_layers,
    )
    _load_random_weights(target, seed=11)
    _load_random_weights(draft, seed=12)
    # The draft borrows the target's embedding and head, so it loads none.
    draft_weights = {
        name: value
        for name, value in draft.state_dict().items()
        if not name.startswith(("embed_tokens.", "lm_head."))
    }
    for slot in ("embed_tokens", "lm_head"):
        if hasattr(target, slot):
            setattr(draft, slot, getattr(target, slot))

    assert PIPELINE_REGISTRY.retrieve_architecture(TARGET_ARCH) is not None
    assert PIPELINE_REGISTRY.retrieve_architecture(DRAFT_ARCH) is not None
    tree = kv_tree(g, target_cfg)
    text_config = SimpleNamespace(hidden_size=g.hidden, vocab_size=g.vocab)
    pipeline_config = SimpleNamespace(
        models=SimpleNamespace(main_architecture_name=TARGET_ARCH),
        draft_model=SimpleNamespace(
            huggingface_config=SimpleNamespace(architectures=[DRAFT_ARCH])
        ),
        speculative=speculative,
        needs_bitmask_constraints=structured_output,
        runtime=SimpleNamespace(reasoning_parser="qwen3_5"),
        model=SimpleNamespace(
            data_parallel_degree=1,
            huggingface_config=SimpleNamespace(
                get_text_config=lambda: text_config
            ),
            pipeline_stage=None,
        ),
    )
    spec = build_spec_graph(
        cast("PipelineConfig", pipeline_config),
        tree,
        [g.device],
        # Qwen3.5 declares signal buffers whatever its device count.
        include_signal_buffers=True,
        target=target,
        target_config=target_cfg,
        draft=draft,
        draft_config=draft_cfg,
        hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
        draft_weights=draft_weights,
    )
    return FusedGraph(
        spec.graph, spec.weights_registry, spec.input_spec, target_cfg, tree
    )
