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
"""Tests that a speculative pairing over a recurrent state disables prefix
caching.

The recurrent cache group checkpoints only when the context length lands
exactly on a page boundary, and under the overlap pipeline that length does
not yet include a speculative step's accepted drafts. Prefix caching would
publish checkpoints at the wrong depth, so it must stay off.
"""

from __future__ import annotations

import pytest
from max.pipelines.architectures import register_all_models
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    spec_decode_architecture,
)


@pytest.mark.parametrize(
    "target",
    ["InklingForConditionalGeneration", "Qwen3_5ForConditionalGeneration"],
)
def test_mtp_over_a_recurrent_target_disables_prefix_caching(
    target: str,
) -> None:
    register_all_models()
    fused = spec_decode_architecture(target, "mtp", None)
    assert fused is not None
    assert fused.required_arguments["enable_prefix_caching"] is False
