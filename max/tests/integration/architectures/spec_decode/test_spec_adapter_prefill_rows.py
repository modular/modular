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
"""Which spec pairings declare a prefill row or a successor-replay head.

An engine mixes a prompt row into a verify step only on an export whose
record carries the prefill-row declaration, and replays a cached prompt
row's boundary token only on one whose record carries the replay
declaration, so a pairing that declares either without its graph keeping
the promise corrupts that row's state.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from max.pipelines.architectures import register_all_models
from max.pipelines.lib import PIPELINE_REGISTRY
from max.pipelines.speculative import proposers, spec_adapters

# The pairings whose verify commits a prefill row riding a verify step,
# as the engine registry's mixed-verify list holds them.
PREFILL_ROW_PAIRS = {
    ("Gemma4ForConditionalGeneration", "mtp"),
    ("Gemma4UnifiedForConditionalGeneration", "mtp"),
    ("Qwen3_5ForCausalLM", "mtp"),
    ("Qwen3_5ForCausalLM", "dflash2"),
    ("Qwen3_5ForConditionalGeneration", "mtp"),
    ("Qwen3_5ForConditionalGeneration", "dflash2"),
    ("DeepseekV41ForCausalLM_ModuleV3", "dspark"),
    ("GlmMoeDsaForCausalLM_ModuleV3", "mtp"),
    ("NemotronHForCausalLM_ModuleV3", "dspark"),
    ("MiMoV2ForCausalLM", "dflash"),
    ("LagunaForCausalLM_ModuleV3", "dflash"),
}

# The pairings whose engine replays a cached prompt row's boundary token,
# as the engine registry's fused-NextN drafter list holds them.
SUCCESSOR_REPLAY_PAIRS = {
    ("GlmMoeDsaForCausalLM_ModuleV3", "mtp"),
}


def _target_config() -> Any:
    """Returns a config an adapter's condition reads its answer off."""
    return SimpleNamespace(
        decode_context_parallel_degree=1, data_parallel_degree=4
    )


def test_the_declared_pairings_take_prefill_rows() -> None:
    for arch, method in sorted(PREFILL_ROW_PAIRS):
        assert spec_adapters.verify_takes_prefill_rows(
            arch, method, _target_config()
        ), f"{arch} {method} declares no prefill row"


def test_the_declared_pairings_replay_a_successor() -> None:
    for arch, method in sorted(SUCCESSOR_REPLAY_PAIRS):
        assert spec_adapters.successor_replay(arch, method), (
            f"{arch} {method} declares no successor replay"
        )


def test_the_engine_registry_pairs_all_declare_prefill_rows() -> None:
    """Every mixed-verify pairing the registry serves declares a prefill
    row."""
    # The trunk adapter declares no fact at this base; MAX's layout-blind
    # pairing flag is a separate divergence, tracked.
    exempt = {("DeepseekV41ForCausalLM", "dspark")}
    register_all_models()
    for arch in PIPELINE_REGISTRY.all_architectures():
        if not arch.supports_spec_decode_mixed_batches:
            continue
        methods = {
            method
            for method in set(arch.checkpoint_drafts)
            | set(arch.draft_architectures)
            | set(arch.spec_decode_graph_names)
            if method in proposers.BLOCK_METHODS
        }
        for method in methods:
            if (arch.name, method) in exempt:
                continue
            assert spec_adapters.verify_takes_prefill_rows(
                arch.name, method, _target_config()
            ), (
                f"{arch.name} serves mixed batches under {method}, but its"
                " verify declares no prefill row"
            )


def test_every_other_pairing_leaves_prefill_rows_out() -> None:
    """No pairing outside the table declares a prefill row."""
    register_all_models()
    declared = {
        (arch, method)
        for arch, method in _registered_pairs()
        if spec_adapters.verify_takes_prefill_rows(
            arch, method, _target_config()
        )
    }
    assert declared == PREFILL_ROW_PAIRS


def test_every_other_pairing_leaves_successor_replay_out() -> None:
    """No pairing outside the table declares a successor-replay head."""
    register_all_models()
    declared = {
        (arch, method)
        for arch, method in _registered_pairs()
        if spec_adapters.successor_replay(arch, method)
    }
    assert declared == SUCCESSOR_REPLAY_PAIRS


def _registered_pairs() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for arch in PIPELINE_REGISTRY.all_architectures():
        if not spec_adapters.has_adapter(arch.name):
            continue
        for method in (
            set(arch.checkpoint_drafts)
            | set(arch.draft_architectures)
            | set(arch.spec_decode_graph_names)
        ):
            if method in proposers.SEQUENTIAL_METHODS | proposers.BLOCK_METHODS:
                pairs.add((arch.name, method))
    assert pairs, "no registered pairing found at this base"
    return pairs


@pytest.mark.parametrize("dcp", [2, 8])
def test_decode_context_parallel_refuses_prefill_rows(dcp: int) -> None:
    """A decode-context-parallel target never declares a prefill row."""
    config = SimpleNamespace(
        decode_context_parallel_degree=dcp, data_parallel_degree=4
    )
    assert not spec_adapters.verify_takes_prefill_rows(
        "GlmMoeDsaForCausalLM_ModuleV3", "mtp", config
    )


@pytest.mark.parametrize("dp", [1, 4])
def test_the_data_parallel_degree_decides_prefill_rows(dp: int) -> None:
    """Only the data-parallel (DEP) layout declares a prefill row."""
    config = SimpleNamespace(
        decode_context_parallel_degree=1, data_parallel_degree=dp
    )
    assert spec_adapters.verify_takes_prefill_rows(
        "DeepseekV41ForCausalLM_ModuleV3", "dspark", config
    ) is (dp > 1)
