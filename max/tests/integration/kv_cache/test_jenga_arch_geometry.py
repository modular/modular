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
"""Pins the Jenga geometry each checked-in architecture config plans.

A leaf page with a prime factor the others lack can grow the huge block from
megabytes to gigabytes, stranding whole blocks per leaf.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.deepseekV2.model_config import (
    DeepseekV2Config,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)
from max.pipelines.architectures.gemma3.model_config import Gemma3Config
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.architectures.glm5_1.model_config import Glm5_1Config
from max.pipelines.architectures.gpt_oss.model_config import GptOssConfig
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.kv_cache import JengaKVCacheManager
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    JengaGeometry,
    _stranding_warning,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative.config import SpeculativeConfig
from transformers import AutoConfig

ARCHITECTURES = Path(__file__).parents[1] / "architectures"

# One B200's share of DeepSeek-V4.1's 101.57 GiB DEP4 KV budget.
BUDGET = int(101.57 / 4 * 2**30)

# DeepSeek-V4.1 serves with DSpark, whose five drafts give the state ring
# eight rows.
SPECULATIVE = {
    "deepseekV41": SpeculativeConfig(
        speculative_method="dspark", num_speculative_tokens=5
    )
}

# (constructor, testdata dir, use text_config, huge block bytes, huge blocks,
# padded leaf strides)
CASES = [
    (
        DeepseekV41Config.construct_kv_params,
        "deepseekV41",
        False,
        52_310_016,
        521,
        {
            "c2a_state.scratch_group": 99_072,
            "idx_c2a.full_group": 13_824,
            "idx_c1a.full_group": 9_216,
        },
    ),
    (
        DeepseekV2Config.construct_kv_params,
        "deepseekV2",
        False,
        3_981_312,
        6848,
        {},
    ),
    (Gemma3Config.construct_kv_params, "gemma3", True, 5_767_168, 4727, {}),
    (
        Gemma4ForConditionalGenerationConfig.construct_kv_params,
        "gemma4",
        False,
        104_857_600,
        260,
        {},
    ),
    (Glm5_1Config.construct_kv_params, "glm5_1", False, 589_824, 46225, {}),
    (GptOssConfig.construct_kv_params, "gpt-oss", False, 3_145_728, 8667, {}),
    (Llama3Config.construct_kv_params, "llama3", False, 8192, 3_328_245, {}),
    (Qwen3Config.construct_kv_params, "qwen3", False, 14_680_064, 1857, {}),
]


@pytest.mark.parametrize(
    ("construct", "arch_dir", "text", "huge", "blocks", "strides"),
    CASES,
    ids=[case[1] for case in CASES],
)
def test_geometry_is_pinned(
    construct: Callable[..., KVCacheParamInterface],
    arch_dir: str,
    text: bool,
    huge: int,
    blocks: int,
    strides: dict[str, int],
) -> None:
    hf_config = AutoConfig.from_pretrained(
        str(ARCHITECTURES / arch_dir / "testdata" / "config.json")
    )
    if text:
        hf_config = hf_config.get_text_config()
    pipeline_config = MagicMock()
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.model.pipeline_stage = None
    pipeline_config.speculative = SPECULATIVE.get(arch_dir)
    params = construct(
        huggingface_config=hf_config,
        pipeline_config=pipeline_config,
        devices=[DeviceRef.GPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
    )
    geometry = JengaKVCacheManager._plan_geometry(params, BUDGET)
    leaves = params.leaves()
    padded = {
        leaf_id: size
        for leaf_id, size in geometry.padded_sizes.items()
        if size != leaves[leaf_id].bytes_per_page
    }
    assert (geometry.huge_page_bytes, geometry.num_huge_blocks, padded) == (
        huge,
        blocks,
        strides,
    )
    assert _stranding_warning(geometry, len(leaves), BUDGET) is None


def test_stranding_warning_fires_past_five_percent() -> None:
    # DeepSeek-V4.1's former 1.07 GiB huge block, over six leaves.
    coarse = JengaGeometry(23, 1_149_763_584, {}, {})
    warning = _stranding_warning(coarse, 6, BUDGET)
    assert warning is not None and "29.5%" in warning
