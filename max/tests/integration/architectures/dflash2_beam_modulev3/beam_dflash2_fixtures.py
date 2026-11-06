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
"""Builds the fused Beam + DFlash2 module on a CPU mesh, for no-GPU gates.

A tiny bfloat16 Beam target (four layers, the last full attention, layers
1-3 sparse) and a two-layer drafter tapping layers 1 and 3, joined through
the same driver call the pairing compiles. Nothing here executes: the stage
only emits the ops.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from max import functional as F
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.beam_modulev3.beam import Beam
from max.pipelines.architectures.beam_modulev3.model_config import BeamConfig
from max.pipelines.architectures.dflash2_beam_modulev3 import (
    DFlash2Beam,
    DFlash2BeamConfig,
    DFlash2BeamDraftConfig,
)
from max.pipelines.architectures.dflash2_beam_modulev3.model_config import (
    drafter_kv_params,
)
from max.pipelines.architectures.dflash_mimo_v2 import DrafterExport
from max.pipelines.kv_cache import KVCacheConfig
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.module_registration import device_mesh
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _driver_call,
    _graph_name,
    _Pairing,
    _SpecDecodeModule,
)
from max.pipelines.speculative import SpeculativeConfig
from max.tensor import default_device, default_dtype

TESTDATA = Path(__file__).parent / "testdata"
DRAFTS = 7
BLOCK = DRAFTS + 1
TAPS = [1, 3]
VOCAB = 512
SAMPLEABLE = 448
HIDDEN = 512


def checkpoint_config() -> dict[str, Any]:
    """The Beam DFlash2 drafter's own ``config.json``."""
    return json.loads((TESTDATA / "beam_dflash2_config.json").read_text())


def tiny_draft_hf_config() -> dict[str, Any]:
    """The drafter's config at the tiny target's width, two layers deep."""
    config = copy.deepcopy(checkpoint_config())
    config.update(
        hidden_size=HIDDEN,
        intermediate_size=1024,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        layer_types=["sliding_attention"] * 2,
        vocab_size=VOCAB,
        num_target_layers=4,
    )
    config["dflash_config"].update(
        target_layer_ids=TAPS, mask_token_id=VOCAB - 100, selector_rank=32
    )
    return config


def _target_hf_config() -> SimpleNamespace:
    """The target's ``config.json`` fields a Beam config reads, tiny."""
    return SimpleNamespace(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=1024,
        moe_intermediate_size=256,
        num_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        rms_norm_eps=1e-6,
        sliding_window=512,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        final_logit_softcapping=30.0,
        num_hidden_layers=4,
    )


def pipeline_config() -> PipelineConfig:
    """What the pairing reads of a served config that prefers ModuleV3."""
    return cast(
        PipelineConfig,
        SimpleNamespace(
            models=SimpleNamespace(main_architecture_name="BeamForCausalLM"),
            draft_model=SimpleNamespace(
                huggingface_config=SimpleNamespace(
                    architectures=["DFlash2BeamDraftModel"]
                )
            ),
            speculative=SpeculativeConfig(
                speculative_method="dflash2", num_speculative_tokens=DRAFTS
            ),
            needs_bitmask_constraints=False,
            # Beam serves with its reasoning parser, which must not add the
            # relaxed-acceptance input Mach's block adapter does not bind.
            runtime=SimpleNamespace(
                reasoning_parser="beam", prefer_module_v3=True
            ),
            model=SimpleNamespace(
                data_parallel_degree=1,
                pipeline_stage=None,
                huggingface_config=SimpleNamespace(
                    get_text_config=_target_hf_config
                ),
            ),
        ),
    )


def graph_name() -> str:
    """The name the pairing compiles its fused graph under."""
    return _graph_name(
        _Pairing.from_pipeline_config(pipeline_config()), "dflash2"
    )


def build_parts(
    devices: list[DeviceRef],
) -> tuple[BeamConfig, DFlash2BeamConfig, MultiKVCacheParams]:
    """The target's and drafter's configs and their KV tree."""
    hf = _target_hf_config()
    kv_cache = KVCacheConfig(kv_cache_page_size=128)
    pc = SimpleNamespace(
        speculative=SpeculativeConfig(
            speculative_method="dflash2", num_speculative_tokens=DRAFTS
        ),
        model=SimpleNamespace(data_parallel_degree=1),
    )
    target_kv = BeamConfig.construct_kv_params(
        cast(Any, hf), cast(Any, pc), devices, kv_cache, DType.bfloat16
    )
    draft_kv = drafter_kv_params(
        tiny_draft_hf_config(), cast(Any, pc), devices, kv_cache
    )
    # The pairing stamps every leaf with the block's width.
    target_kv = MultiKVCacheParams.from_params(
        {
            name: replace(leaf, num_draft_tokens=BLOCK)
            for name, leaf in target_kv.children.items()
            if isinstance(leaf, KVCacheParams)
        }
    )
    config = BeamConfig(
        vocab_size=hf.vocab_size,
        hidden_size=hf.hidden_size,
        num_attention_heads=hf.num_attention_heads,
        num_key_value_heads=hf.num_key_value_heads,
        head_dim=hf.head_dim,
        intermediate_size=hf.intermediate_size,
        moe_intermediate_size=hf.moe_intermediate_size,
        num_experts=hf.num_experts,
        num_experts_per_tok=hf.num_experts_per_tok,
        n_group=hf.n_group,
        topk_group=hf.topk_group,
        norm_topk_prob=hf.norm_topk_prob,
        routed_scaling_factor=hf.routed_scaling_factor,
        rms_norm_eps=hf.rms_norm_eps,
        sliding_window=hf.sliding_window,
        rope_theta=10000.0,
        layer_types=list(hf.layer_types),
        mlp_layer_types=list(hf.mlp_layer_types),
        final_logit_softcapping=hf.final_logit_softcapping,
        max_seq_len=1024,
        dtype=DType.bfloat16,
        devices=devices,
        kv_params=target_kv,
        # As the pairing builds the target: every row scores its logits.
        return_logits=ReturnLogits.VARIABLE,
        mesh=device_mesh([ref.to_device() for ref in devices], 1),
    )
    draft = DFlash2BeamConfig.from_huggingface_config(
        tiny_draft_hf_config(),
        block_size=BLOCK,
        devices=devices,
        kv_params=draft_kv,
        max_seq_len=1024 + 2 * BLOCK,
    )
    kv_tree = MultiKVCacheParams.from_params(
        {"target": target_kv, "draft": draft_kv}
    )
    return config, draft, kv_tree


def build_module(
    config: BeamConfig, draft: DFlash2BeamConfig, kv_tree: MultiKVCacheParams
) -> _SpecDecodeModule:
    """The fused module the pairing compiles, traced on one device."""
    assert config.mesh is not None
    with default_device(config.mesh), default_dtype(DType.bfloat16), F.lazy():
        target = Beam(config)
        drafter = DFlash2Beam(draft)
    draft_config = DFlash2BeamDraftConfig(
        module=draft,
        sampleable_vocab_size=SAMPLEABLE,
        drafter_export=DrafterExport(
            directory=Path(),
            target_layer_ids=list(draft.target_layer_ids),
            num_speculative_tokens=DRAFTS,
            spec_block_size=BLOCK,
        ),
    )
    return _SpecDecodeModule(
        target,
        drafter,
        partial(
            _driver_call,
            pipeline_config(),
            kv_tree,
            list(config.devices),
            include_signal_buffers=True,
            target_config=config,
            draft_config=draft_config,
            hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            trace_owns_signal_buffers=False,
            signal_buffer_index=None,
        ),
    )


def fake_weights(module: _SpecDecodeModule) -> dict[str, WeightData]:
    """A zero-filled weight for every parameter, at its declared shape."""
    weights = {}
    for name, param in module.parameters:
        shape = [int(d) for d in param.shape]
        weights[name] = WeightData(
            name=name,
            dtype=param.dtype,
            shape=Shape(shape),
            data=Buffer.zeros(tuple(shape), dtype=param.dtype),
        )
    return weights
