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
"""Checks the Inkling 2 loop config and the visit order it puts the decoder in.

The geometry is the looped checkpoint's: 24 layers, every sixth one global,
and ``text_config.loop_config`` over layers [6, 18) run three times. Both
caches and the conv pool are sized from that visit list, so the counts here
are the ones a serve allocates.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.dtype import DType
from max.pipelines.architectures.hf_config_shims import Inkling2LoopHFConfig
from max.pipelines.architectures.inkling2.arch import (
    inkling2_arch,
    inkling2_loop_arch,
)
from max.pipelines.architectures.inkling2.model_config import (
    Inkling2Config,
    Inkling2LoopConfig,
    Inkling2TextConfig,
)
from max.pipelines.architectures.inkling2.state_cache import (
    ConvSite,
    ConvStateLayout,
)

NUM_HIDDEN_LAYERS = 24
"""The looped checkpoint's layer count. ``num_hidden_layers`` is the target's
and does not move when the core repeats."""

LOCAL_LAYER_IDS = tuple(i for i in range(NUM_HIDDEN_LAYERS) if i % 6 != 5)
"""Every layer but each sixth, which attends over the whole sequence."""

SHIPPED_LOOP: dict[str, Any] = {
    "core_start": 6,
    "core_end": 18,
    "num_passes": 3,
    "gate_rank": 64,
    "is_varloop": False,
}


def _loop(**overrides: Any) -> Inkling2LoopConfig:
    return Inkling2LoopConfig(**{**SHIPPED_LOOP, **overrides})


def _text_config(
    loop: Inkling2LoopConfig | None = None,
) -> Inkling2TextConfig:
    return Inkling2TextConfig(
        hidden_size=4608,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        rms_norm_eps=1e-6,
        vocab_size=201024,
        unpadded_vocab_size=200058,
        num_attention_heads=48,
        head_dim=128,
        num_key_value_heads=8,
        swa_num_attention_heads=48,
        swa_head_dim=128,
        swa_num_key_value_heads=8,
        sliding_window_size=1024,
        d_rel=64,
        rel_extent=4096,
        log_scaling_alpha=0.1,
        log_scaling_n_floor=4096,
        qk_l2_clip_threshold=1.0,
        rel_l2_clip_threshold=1.0,
        sconv_kernel_size=4,
        local_layer_ids=LOCAL_LAYER_IDS,
        loop_config=loop,
        n_layers_per_attn_res_block=6,
        attn_res_normalize_values=True,
        n_routed_experts=256,
        num_experts_per_tok=8,
        intermediate_size=1536,
        route_scale=1.0,
        mlp_gate_softcap=30.0,
        mlp_up_softcap=30.0,
        final_logit_softcapping=30.0,
        model_max_length=262144,
    )


def test_unlooped_visits_every_layer_once() -> None:
    """The merged checkpoints declare no loop and must not move."""
    config = _text_config()
    assert config.visit_layer_ids() == tuple(range(NUM_HIDDEN_LAYERS))
    assert config.num_local_visits == 20
    assert config.num_global_visits == 4


def test_looped_walks_prelude_then_passes_then_coda() -> None:
    """The prelude runs once, each pass runs the whole core, the coda runs
    once: a pass is a whole sweep of the core, not a layer's repeats."""
    core = tuple(range(6, 18))
    assert _text_config(_loop()).visit_layer_ids() == (
        *range(6),
        *(core * 3),
        *range(18, NUM_HIDDEN_LAYERS),
    )


def test_visit_counts_size_both_caches() -> None:
    config = _text_config(_loop())
    visits = config.visit_layer_ids()
    # 6 prelude + 12 core layers x 3 passes + 6 coda
    assert len(visits) == 48
    # Two of the twelve core layers are global (11 and 17), so the three passes
    # add six global and thirty local rows to the unlooped 4 and 20.
    assert config.num_global_visits == 8
    assert config.num_local_visits == 40


def test_core_past_the_last_layer_is_refused() -> None:
    with pytest.raises(ValueError, match="core layers"):
        _text_config(_loop(core_end=NUM_HIDDEN_LAYERS + 6))


@pytest.mark.parametrize(
    "overrides",
    [
        {"core_start": 12, "core_end": 12},
        {"num_passes": 0},
        {"gate_rank": 0},
        {"is_varloop": True},
    ],
)
def test_a_loop_the_decoder_cannot_run_is_refused(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _loop(**overrides)


@pytest.mark.parametrize("kind", ["mapping", "config"])
def test_reads_the_loop_the_checkpoint_declares(kind: str) -> None:
    """A released checkpoint writes a mapping; a release that parses this
    family itself would hand over a config object."""
    block: Any = (
        SHIPPED_LOOP if kind == "mapping" else SimpleNamespace(**SHIPPED_LOOP)
    )
    loop = Inkling2LoopConfig.from_hf(block)
    assert loop == _loop()
    assert loop.core_layer_ids == tuple(range(6, 18))
    assert loop.is_core_layer(17)
    assert not loop.is_core_layer(18)


def test_shim_nests_the_looped_checkpoints_blocks() -> None:
    """`Inkling2TextConfig.from_hf` reads `loop_config` off the nested text
    block, so the shim has to hand it a config object, aliases applied."""
    config = Inkling2LoopHFConfig.from_dict(
        {
            "model_type": "inkling2_loop",
            "text_config": {
                "loop_config": SHIPPED_LOOP,
                "conv_kernel_size": 4,
            },
            "vision_config": {"patch_size": 20},
        }
    )
    assert not isinstance(config.text_config, Mapping)
    assert config.text_config.loop_config == SHIPPED_LOOP
    assert config.text_config.sconv_kernel_size == 4
    assert config.vision_config.native_patch_size == 20


def test_core_start_inside_a_block_is_refused() -> None:
    """A pass reads the blocks the prelude left behind, so the core starts
    where one of them begins."""
    with pytest.raises(ValueError, match="attn_res block"):
        _text_config(_loop(core_start=8))


def _state_dict(*, ships_controllers: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        f"layers.{layer}.mlp.experts.w13_weight": SimpleNamespace(
            dtype=DType.uint8
        )
        for layer in range(NUM_HIDDEN_LAYERS)
    }
    if ships_controllers:
        for layer in range(6, 18):
            state[f"layers.{layer}.controllers.attn.down_weight"] = (
                SimpleNamespace(dtype=DType.bfloat16)
            )
    return state


@pytest.mark.parametrize("ships_controllers", [True, False])
def test_weights_and_config_must_agree_about_the_loop(
    ships_controllers: bool,
) -> None:
    """The loop is switched on by the config alone, so a checkpoint whose two
    halves disagree would serve the wrong graph rather than fail."""
    config = object.__new__(Inkling2Config)
    config.text_config = _text_config(None if ships_controllers else _loop())
    with pytest.raises(ValueError, match="disagree about the loop"):
        Inkling2Config.finalize(
            config,
            cast(Any, None),
            _state_dict(ships_controllers=ships_controllers),
        )


def test_the_looped_arch_serves_no_drafter() -> None:
    """Mini ships a DSpark drafter whose taps are the unlooped decoder's;
    until they are validated against the loop's walk, the looped arch names
    no draft so a speculative request fails instead of drafting off the wrong
    activations."""
    assert inkling2_loop_arch.checkpoint_drafts == {}
    assert inkling2_loop_arch.spec_decode_graph_names == {}
    assert inkling2_arch.checkpoint_drafts == {
        "dspark": "Inkling2DSparkDraftModel_ModuleV3"
    }


def test_unlooped_checkpoint_keeps_smalls_walk_and_conv_layout() -> None:
    """The loop edits the shared decoder, so Small's structure is asserted
    here: one visit per layer, and a conv layout the decoder can still index
    by layer. The looped config is where the two diverge, 48 visits against
    the checkpoint's 24 layers."""
    config = _text_config()
    layout = ConvStateLayout.from_config(config)
    assert len(layout.layers) == NUM_HIDDEN_LAYERS
    assert layout.is_local == tuple(
        config.is_local_attention(layer) for layer in range(NUM_HIDDEN_LAYERS)
    )
    # Every unlooped layer still addresses its own conv row at each site.
    for site in ConvSite:
        rows = [
            layout.row_for(layer, site) for layer in range(NUM_HIDDEN_LAYERS)
        ]
        assert len(set(rows)) == NUM_HIDDEN_LAYERS
    looped = ConvStateLayout.from_config(_text_config(_loop()))
    assert len(looped.layers) == 48
