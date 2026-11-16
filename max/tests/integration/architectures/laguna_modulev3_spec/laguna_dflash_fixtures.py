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
"""Builds the fused Laguna + DFlash module on CPU, for no-GPU gates.

The tiny target of :mod:`laguna_spec_fixtures` and a two-layer drafter
tapping its layers 1 and 3, the last, as S-2.1's drafter taps its last
layer. They are joined through the driver call the pairing compiles, with
the arguments it passes on one device. Nothing here executes.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from laguna_parity import HEAD_DIM, HIDDEN_SIZE, MAX_LENGTH, VOCAB_SIZE
from laguna_spec_fixtures import hf_config, target_config
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.dflash_laguna_modulev3 import (
    DFlashLaguna,
    DFlashLagunaConfig,
    DFlashLagunaDraftConfig,
)
from max.pipelines.architectures.dflash_laguna_modulev3.model_config import (
    drafter_kv_params,
)
from max.pipelines.architectures.laguna_modulev3.laguna import Laguna
from max.pipelines.kv_cache import KVCacheConfig
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _driver_call,
    _graph_name,
    _Pairing,
    _SpecDecodeModule,
)
from max.pipelines.speculative import DrafterExport, SpeculativeConfig
from max.tensor import default_device, default_dtype

TESTDATA = Path(__file__).parents[1] / "dflash_laguna_modulev3" / "testdata"
DRAFTS = 7
BLOCK = DRAFTS + 1
TAPS = [1, 3]
SAMPLEABLE = 448


def tiny_draft_hf_config() -> dict[str, Any]:
    """The drafter's ``config.json`` at the tiny target's width."""
    config = copy.deepcopy(
        json.loads((TESTDATA / "laguna_dflash_config.json").read_text())
    )
    config.update(
        hidden_size=HIDDEN_SIZE,
        head_dim=HEAD_DIM,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=2,
        layer_types=["sliding_attention"] * 2,
        sliding_windows=[512] * 2,
        vocab_size=VOCAB_SIZE,
        draft_vocab_size=VOCAB_SIZE,
    )
    config["dflash_config"].update(target_layer_ids=TAPS, num_target_layers=4)
    return config


def pipeline_config() -> PipelineConfig:
    """What the pairing reads of a served config that prefers ModuleV3."""
    speculative = SpeculativeConfig(
        speculative_method="dflash", num_speculative_tokens=DRAFTS
    )
    return cast(
        PipelineConfig,
        SimpleNamespace(
            models=SimpleNamespace(main_architecture_name="LagunaForCausalLM"),
            draft_model=SimpleNamespace(
                huggingface_config=SimpleNamespace(
                    architectures=["DFlashLagunaForCausalLM"]
                )
            ),
            speculative=speculative,
            needs_bitmask_constraints=False,
            # Laguna serves with its reasoning parser, which must not add the
            # relaxed-acceptance input Mach's block adapter does not bind.
            runtime=SimpleNamespace(
                reasoning_parser="laguna", prefer_module_v3=True
            ),
            model=SimpleNamespace(
                data_parallel_degree=1,
                pipeline_stage=None,
                huggingface_config=SimpleNamespace(get_text_config=hf_config),
            ),
        ),
    )


def graph_name() -> str:
    """The name the pairing compiles its fused graph under."""
    return _graph_name(
        _Pairing.from_pipeline_config(pipeline_config()), "dflash"
    )


def build_module() -> _SpecDecodeModule:
    """The fused module the pairing compiles, traced on one device."""
    devices = [DeviceRef.CPU()]
    target = target_config(
        ModelOutputConfig(
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            target_layer_ids=tuple(TAPS),
        )
    )
    # The pairing stamps every leaf with its method and the block's width.
    assert isinstance(target.kv_params, MultiKVCacheParams)
    target_kv = MultiKVCacheParams.from_params(
        {
            name: replace(
                leaf, speculative_method="dflash", num_draft_tokens=BLOCK
            )
            for name, leaf in target.kv_params.children.items()
            if isinstance(leaf, KVCacheParams)
        }
    )
    target = replace(target, kv_params=target_kv)
    draft_kv = drafter_kv_params(
        tiny_draft_hf_config(),
        pipeline_config(),
        devices,
        KVCacheConfig(kv_cache_page_size=128),
    )
    draft = DFlashLagunaConfig.from_huggingface_config(
        tiny_draft_hf_config(),
        block_size=BLOCK,
        devices=devices,
        kv_params=draft_kv,
        max_seq_len=MAX_LENGTH + 2 * BLOCK,
    )
    with F.lazy(), default_dtype(DType.bfloat16), default_device(CPU()):
        target_module = Laguna(target)
        drafter = DFlashLaguna(draft)
    draft_config = DFlashLagunaDraftConfig(
        module=draft,
        sampleable_vocab_size=SAMPLEABLE,
        drafter_export=DrafterExport(
            directory=Path(),
            target_layer_ids=list(TAPS),
            num_speculative_tokens=DRAFTS,
            spec_block_size=BLOCK,
        ),
    )
    return _SpecDecodeModule(
        target_module,
        drafter,
        partial(
            _driver_call,
            pipeline_config(),
            MultiKVCacheParams.from_params(
                {"target": target_kv, "draft": draft_kv}
            ),
            devices,
            include_signal_buffers=True,
            target_config=target,
            draft_config=draft_config,
            hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            # As the pairing compiles a one-device module whose adapter
            # names no signal-buffer slot: the trace declares none.
            trace_owns_signal_buffers=True,
            signal_buffer_index=None,
        ),
    )
