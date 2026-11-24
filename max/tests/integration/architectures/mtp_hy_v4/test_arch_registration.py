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
"""Pins how a Hy4 target and its MTP layer pair."""

from __future__ import annotations

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    spec_decode_architecture,
)


def test_draft_arch_is_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture("MTPHYV4DraftModel")
    assert arch is not None
    assert arch.pipeline_model is None


def test_pairs_with_the_modulev3_target() -> None:
    # The MTP layer ships in the target's checkpoint, which names no draft.
    arch = spec_decode_architecture("HYV4ForCausalLM", "mtp", None)
    assert arch is not None
    assert arch.name == "HYV4ForCausalLM_ModuleV3+mtp"
