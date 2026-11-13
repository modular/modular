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
"""CPU checks of the draft checkpoint contract. No graph, no GPU.

The draft loads through weight rules, which pass every tensor through as
stored. ``testdata/dspark_checkpoint.json`` holds the published drafter's
safetensors header (names, dtypes and shapes) and ``config.json`` at the
revision the Hugging Face cache lock pins, so a misspelled parameter or a
tensor the module stops reading fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

from max.pipelines.architectures.dspark_qwen3.dspark_qwen3 import (
    DSparkQwen3DraftConfig,
    parse_dspark_draft_config,
)
from max.pipelines.architectures.dspark_qwen3.weight_adapters import (
    dspark_qwen3_weight_rules,
    expected_draft_weight_names,
)

_MANIFEST = json.loads(
    (Path(__file__).parent / "testdata" / "dspark_checkpoint.json").read_text()
)

CONFIG = DSparkQwen3DraftConfig(
    hidden_size=4,
    intermediate_size=8,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=2,
    rms_norm_eps=1e-6,
    vocab_size=8,
    rope_theta=10000.0,
    sliding_window=16,
    max_position_embeddings=64,
    block_size=4,
    mask_token_id=3,
    target_layer_ids=(0, 1),
    markov_rank=2,
    sample_from_anchor=False,
    quantized=False,
)


def test_the_rules_keep_every_tensor_as_stored() -> None:
    assert tuple(dspark_qwen3_weight_rules(_MANIFEST["config"])) == ()


def test_the_module_reads_every_published_tensor() -> None:
    """The draft module reads each tensor of the published drafter, by the
    name the checkpoint stores, and nothing it lacks."""
    config = parse_dspark_draft_config(_MANIFEST["config"])
    assert expected_draft_weight_names(config) == set(_MANIFEST["tensors"])


def test_the_mask_row_comes_from_the_stored_table() -> None:
    """The module takes the mask token's row from ``embed_tokens.weight``,
    which the published drafter stores whole, in BF16."""
    names = expected_draft_weight_names(CONFIG)
    assert "embed_tokens.weight" in names
    assert "mask_embedding" not in names
    config = _MANIFEST["config"]
    dtype, shape = _MANIFEST["tensors"]["embed_tokens.weight"]
    assert (dtype, shape) == (
        "BF16",
        [config["vocab_size"], config["hidden_size"]],
    )
    assert 0 <= config["mask_token_id"] < config["vocab_size"]
