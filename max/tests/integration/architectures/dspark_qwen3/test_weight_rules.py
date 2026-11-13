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
revision the Hugging Face cache lock pins. The draft module is built lazily
on the host, so its declared parameters are compared without compiling a
graph, and a misspelled parameter, a wrong stored dtype or shape, or a tensor
the module stops reading fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.dspark_qwen3.dspark_qwen3 import (
    DSparkQwen3,
    DSparkQwen3DraftConfig,
    parse_dspark_draft_config,
)
from max.pipelines.architectures.dspark_qwen3.weight_adapters import (
    dspark_qwen3_weight_rules,
    expected_draft_weight_names,
)
from max.tensor import default_dtype

_MANIFEST = json.loads(
    (Path(__file__).parent / "testdata" / "dspark_checkpoint.json").read_text()
)

_SAFETENSORS_DTYPES = {
    "BF16": DType.bfloat16,
    "F32": DType.float32,
    "F8_E4M3": DType.float8_e4m3fn,
    "U8": DType.uint8,
}

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


def _draft(config: DSparkQwen3DraftConfig) -> DSparkQwen3:
    """Builds the draft module on the host, lazily, as the draft
    registration builds it: BF16 by default, one device."""
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_layers=config.num_hidden_layers,
        devices=[DeviceRef.CPU()],
    )
    with F.lazy(), default_dtype(DType.bfloat16):
        return DSparkQwen3(
            config,
            kv_params,
            num_devices=1,
            max_seq_len=4096,
            device=CPU(),
        )


def _declared(module: DSparkQwen3) -> dict[str, tuple[DType, tuple[int, ...]]]:
    return {
        name: (param.dtype, tuple(int(dim) for dim in param.shape))
        for name, param in module.parameters
    }


def test_the_rules_keep_every_tensor_as_stored() -> None:
    assert tuple(dspark_qwen3_weight_rules(_MANIFEST["config"])) == ()


def test_the_module_declares_every_published_tensor_as_stored() -> None:
    """The draft module declares each tensor of the published drafter, by
    the name, dtype and shape the checkpoint stores, and nothing it lacks."""
    config = parse_dspark_draft_config(_MANIFEST["config"])
    declared = _declared(_draft(config))
    stored = {
        name: (_SAFETENSORS_DTYPES[dtype], tuple(shape))
        for name, (dtype, shape) in _MANIFEST["tensors"].items()
    }

    assert sorted(declared.keys() - stored.keys()) == []
    assert sorted(stored.keys() - declared.keys()) == []
    assert {
        name: (stored[name], declared[name])
        for name in stored
        if stored[name] != declared[name]
    } == {}
    assert expected_draft_weight_names(config) == declared.keys()


def test_the_mask_row_comes_from_the_stored_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module takes the mask token's row from ``embed_tokens.weight``,
    which the published drafter stores whole, in BF16, and keeps on the
    host; it declares no separate mask embedding."""
    stored_dtype, stored_shape = _MANIFEST["tensors"]["embed_tokens.weight"]
    config = _MANIFEST["config"]
    assert (stored_dtype, stored_shape) == (
        "BF16",
        [config["vocab_size"], config["hidden_size"]],
    )
    assert 0 <= config["mask_token_id"] < config["vocab_size"]

    module = _draft(CONFIG)
    declared = _declared(module)
    assert declared["embed_tokens.weight"] == (
        DType.bfloat16,
        (CONFIG.vocab_size, CONFIG.hidden_size),
    )
    assert not any("mask" in name for name in declared)

    table = module.embed_tokens
    rows: list[int] = []
    forward = type(table).forward

    def recording_forward(self: Any, row: int) -> Any:
        assert self is table
        rows.append(row)
        return forward(self, row)

    monkeypatch.setattr(type(table), "forward", recording_forward)
    with F.lazy():
        mask_row = module.mask_embedding
    assert rows == [CONFIG.mask_token_id]
    assert mask_row.dtype == DType.bfloat16
    assert tuple(int(dim) for dim in mask_row.shape) == (CONFIG.hidden_size,)
    assert mask_row.device.is_host
