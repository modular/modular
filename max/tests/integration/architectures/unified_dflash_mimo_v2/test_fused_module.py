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
"""Tests building the fused MiMo-V2 DFlash module on virtual B200s."""

from __future__ import annotations

import pytest
from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.graph import DeviceRef
from max.nn.kv_cache import MultiKVCacheParams
from max.pipelines.architectures.unified_dflash_mimo_v2 import (
    UnifiedDflashMiMoV2,
    fused_graph,
)
from max.pipelines.architectures.unified_dflash_mimo_v2.model_config import (
    DRAFT,
)
from max.pipelines.lib import SpeculativeConfig
from mimo_dflash_harness import Model, tiny_configs

REFS = [DeviceRef.GPU(0)]


@pytest.fixture(scope="module")
def model() -> Model:
    # The module's collectives create their devices; virtual B200s serve.
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(len(REFS))
    return tiny_configs()


def test_greedy_acceptance_refuses_a_synthetic_rate(model: Model) -> None:
    spec = model.spec(REFS, 7, sampleable_vocab_size=1000)
    spec.speculative_config = SpeculativeConfig(
        speculative_method="dflash",
        num_speculative_tokens=7,
        use_greedy_acceptance=True,
        synthetic_acceptance_rate=0.5,
    )
    with pytest.raises(ValueError, match="synthetic_acceptance_rate"):
        UnifiedDflashMiMoV2(spec, model.mask_embedding)


def test_the_graph_returns_the_three_outputs_serving_reads(
    model: Model,
) -> None:
    # A fourth output would reach the serving model as draft probabilities.
    spec = model.spec(REFS, 7, sampleable_vocab_size=1000)
    nn_model = UnifiedDflashMiMoV2(spec, model.mask_embedding)
    for name, weight in nn_model.raw_state_dict().items():
        weight.name = name
    target_kv = spec.target.kv_params
    assert isinstance(target_kv, MultiKVCacheParams)
    graph = fused_graph(
        nn_model,
        MultiKVCacheParams.from_params(
            {"target": target_kv, DRAFT: spec.draft.kv_params}
        ),
    )
    assert len(graph.output_types) == 3


@pytest.mark.parametrize("sampleable", [0, 1025])
def test_the_sampleable_vocab_fits_the_head(
    model: Model, sampleable: int
) -> None:
    spec = model.spec(REFS, 7, sampleable_vocab_size=sampleable)
    with pytest.raises(ValueError, match="sampleable_vocab_size"):
        UnifiedDflashMiMoV2(spec, model.mask_embedding)
