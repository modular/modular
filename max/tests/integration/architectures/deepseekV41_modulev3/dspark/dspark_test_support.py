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
"""Shared, side-effect-free config/data helpers for the DSpark test suites.

Only checkpoint text/config readers, the real and shrunk configs they
resolve, and the CPU tensor conversion helpers live here. This module
must stay inert at import: no device, mesh, session, or graph is ever
constructed here, so any test process can import it before its first
accelerator exists.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor

# The reference's block-mask constants.
WINDOW = 8
BLOCK = 5

TESTDATA = Path(__file__).resolve().parent.parent / "testdata"
"""The checkpoint config fixture directory (``config_full.json``)."""


def _to_max(t: torch.Tensor) -> Tensor:
    return Tensor(t, device=CPU())


def _text() -> SimpleNamespace:
    return SimpleNamespace(
        **json.load(open(TESTDATA / "config_full.json"))["text_config"]
    )


def _full_hf() -> dict[str, object]:
    """The checkpoint's own config mapping, as the interface reads it."""
    return json.load(open(TESTDATA / "config_full.json"))


def _real_config() -> DeepseekV41Config:
    # The real KV-leaf tree: several drafter tests read leaf names and
    # table extents off the real interface's params.
    kv_params = build_kv_params(
        V41Config.from_hf(_full_hf()),
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [DeviceRef.CPU()],
    )
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv_params,
        devices=[DeviceRef.CPU()],
        max_seq_len=1024,
        **DeepseekV41Config.resolve_text_fields(_text()),
    )


def _small_config() -> DeepseekV41Config:
    """The real config's structure at test-sized widths.

    Every MXFP8 input width stays a multiple of 32 (the block-scaled
    matmuls' requirement) and the noise token stays inside the shrunk
    vocabulary so the draft loop's post-stop gathers stay in bounds.
    """
    return dataclasses.replace(
        _real_config(),
        vocab_size=97,
        hidden_size=128,
        num_attention_heads=2,
        head_dim=64,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        moe_intermediate_size=64,
        sliding_window=64,
        dspark_n_routed_experts=8,
        dspark_num_experts_per_tok=2,
        dspark_markov_rank=16,
        dspark_noise_token_id=96,
    )
