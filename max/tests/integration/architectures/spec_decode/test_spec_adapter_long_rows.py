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
"""Which spec targets declare that their verify commits a long row.

An engine mixes a prompt row into a verify step only on an export whose
sidecar carries this, so a target that declares it without committing a long
row in every recurrent leaf corrupts that row's state.
"""

from __future__ import annotations

import pytest
from max.pipelines.architectures import register_all_models
from max.pipelines.lib import PIPELINE_REGISTRY
from max.pipelines.speculative import proposers, spec_adapters


@pytest.mark.parametrize(
    ("target_arch", "method"),
    [
        ("Qwen3_5ForConditionalGeneration", "mtp"),
        ("Qwen3_5ForConditionalGeneration", "dflash2"),
        ("NemotronHForCausalLM_ModuleV3", "dspark"),
    ],
)
def test_a_recurrent_target_commits_long_rows(
    target_arch: str, method: str
) -> None:
    assert spec_adapters.verify_commits_long_rows(target_arch, method)


@pytest.mark.parametrize(
    ("target_arch", "method"),
    [
        # Its indexer tail ring has no proof for a long row.
        ("Glm5NextForConditionalGeneration", "mtp"),
        ("LlamaForCausalLM", "eagle"),
        ("HYV4ForCausalLM", "mtp"),
    ],
)
def test_other_targets_do_not_declare_long_row_commits(
    target_arch: str, method: str
) -> None:
    assert not spec_adapters.verify_commits_long_rows(target_arch, method)


def test_recurrent_targets_serve_mixed_batches_only_with_long_row_commits() -> (
    None
):
    """A recurrent target that serves mixed batches must commit a long row."""
    register_all_models()
    recurrent = [
        arch
        for arch in PIPELINE_REGISTRY.all_architectures()
        if arch.checkpoints_recurrent_state
    ]
    assert recurrent, "no architecture walks a recurrent state at this base"
    for arch in recurrent:
        if not arch.supports_spec_decode_mixed_batches:
            continue
        methods = {
            method
            for method in set(arch.checkpoint_drafts)
            | set(arch.draft_architectures)
            if method in proposers.BLOCK_METHODS
        }
        for method in methods:
            assert spec_adapters.verify_commits_long_rows(arch.name, method), (
                f"{arch.name} keeps recurrent state and serves mixed"
                f" batches, but its {method} verify does not commit a long"
                " row in every recurrent leaf"
            )
