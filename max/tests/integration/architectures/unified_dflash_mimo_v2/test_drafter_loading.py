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
"""Tests reading the MiMo-V2 DFlash drafter out of a checkpoint directory,
and the KV trees and drafter records the two pipeline models build from it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVConnectorType,
    MHAKVCacheParams,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.pipelines.architectures.dflash_mimo_v2 import DFlashMiMoV2Config
from max.pipelines.architectures.dflash_mimo_v2.weight_adapters import (
    DRAFT_WEIGHTS_FILE,
    MASK_EMBEDDING_FILE,
    drafter_tensor_shapes,
)
from max.pipelines.architectures.mimo_v2.model_config import (
    FULL,
    SLIDING,
    MiMoV2Config,
)
from max.pipelines.architectures.unified_dflash_mimo_v2 import (
    MiMoV2DFlashContextModel,
    UnifiedDflashMiMoV2Model,
)
from max.pipelines.architectures.unified_dflash_mimo_v2.drafter import (
    load_drafter,
)
from max.pipelines.architectures.unified_dflash_mimo_v2.model_config import (
    DRAFT,
    MiMoV2DFlashContextConfig,
    UnifiedDflashMiMoV2Config,
    repo_file,
)
from max.pipelines.kv_cache import KVConnectorConfig
from max.pipelines.lib import KVCacheConfig, SpeculativeConfig
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.weights import HuggingFaceRepo
from mimo_dflash_harness import (
    Model,
    drafter_kv_params,
    tiny_configs,
    tiny_dflash_config,
)
from safetensors.torch import save_file

DFLASH = tiny_dflash_config([0, 1, 2])
REFS = [DeviceRef.GPU(0)]
MAX_SEQ_LEN = 4096
K = 5
SAMPLEABLE = 1000
WRITER_WEIGHTS = {
    "fc.weight",
    "hidden_norm.weight",
    *(
        f"layers.{i}.self_attn.{w}.weight"
        for i in range(2)
        for w in ("k_proj", "v_proj", "k_norm")
    ),
}
"""What the context writer reads of the drafter: its projection, its norm
and each layer's K/V projections."""


def _config() -> DFlashMiMoV2Config:
    devices = [DeviceRef.CPU()]
    return DFlashMiMoV2Config.from_dflash_config(
        DFLASH,
        devices=devices,
        kv_params=drafter_kv_params(DFLASH, devices, speculative=False),
        max_seq_len=4096,
    )


def _write_drafter(directory: Path, dflash: dict[str, Any]) -> dict[str, Any]:
    """Writes a random drafter's three files; returns its tensors' bits."""
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(dflash))
    config = _config()
    rng = np.random.default_rng(0)
    bits = {
        name: rng.integers(0, 2**15, shape, dtype=np.uint16)
        for name, shape in drafter_tensor_shapes(config).items()
    }
    save_file(
        {
            name: torch.from_numpy(b.view(np.int16)).view(torch.bfloat16)
            for name, b in bits.items()
        },
        str(directory / DRAFT_WEIGHTS_FILE),
    )
    torch.save(
        {
            "mask_token_id": config.mask_token_id,
            "embedding": torch.zeros(config.hidden_size, dtype=torch.bfloat16),
        },
        directory / MASK_EMBEDDING_FILE,
    )
    return bits


@pytest.mark.parametrize("subfolder", [None, "dflash"])
def test_repo_file_is_under_the_repo_folder(
    tmp_path: Path, subfolder: str | None
) -> None:
    repo = HuggingFaceRepo(str(tmp_path), subfolder=subfolder)
    folder = tmp_path / subfolder if subfolder else tmp_path
    assert repo_file(repo, "config.json") == folder / "config.json"


@pytest.mark.parametrize("subfolder", [None, "dflash"])
def test_load_drafter_reads_the_repo_folder(
    tmp_path: Path, subfolder: str | None
) -> None:
    folder = tmp_path / subfolder if subfolder else tmp_path / "drafter"
    bits = _write_drafter(folder, DFLASH)
    # Without a subfolder the drafter is its own repo; with one it sits in
    # the target's.
    root = tmp_path if subfolder else folder
    repo = HuggingFaceRepo(str(root), subfolder=subfolder)

    state, mask, directory = load_drafter(repo, _config())

    assert directory == folder
    assert state.keys() == bits.keys()
    for name, want in bits.items():
        got = np.from_dlpack(state[name].to_buffer().view(DType.uint16))
        np.testing.assert_array_equal(got, want)
    assert mask.dtype == DType.bfloat16


def _pipeline_config(
    model_repo: HuggingFaceRepo, draft_repo: HuggingFaceRepo | None
) -> MagicMock:
    config = MagicMock()
    config.model.huggingface_weight_repo = model_repo
    config.model.data_parallel_degree = 1
    config.draft_model = (
        None
        if draft_repo is None
        else MagicMock(huggingface_weight_repo=draft_repo)
    )
    config.speculative = (
        None
        if draft_repo is None
        else SpeculativeConfig(
            speculative_method="dflash", num_speculative_tokens=K
        )
    )
    return config


def _kv_params(
    config_cls: type[MiMoV2Config], pipeline_config: MagicMock, model: Model
) -> MultiKVCacheParams:
    return config_cls.construct_kv_params(
        model.hf, pipeline_config, REFS, KVCacheConfig(), DType.bfloat16
    )


def _target_from(
    model: Model, monkeypatch: pytest.MonkeyPatch, *, speculative: bool
) -> list[int]:
    """Makes ``MiMoV2Config.initialize`` build the tiny target; returns the
    ``max_seq_len`` of each call."""
    lengths: list[int] = []

    def initialize(
        cls: type[MiMoV2Config],
        pipeline_config: object,
        model_config: object = None,
        *,
        max_seq_len: int,
    ) -> MiMoV2Config:
        lengths.append(max_seq_len)
        return model.target(REFS, speculative=speculative)

    monkeypatch.setattr(MiMoV2Config, "initialize", classmethod(initialize))
    return lengths


@pytest.mark.parametrize("subfolder", [None, "dflash"])
def test_the_fused_model_loads_its_draft_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subfolder: str | None
) -> None:
    folder = tmp_path / subfolder if subfolder else tmp_path / "drafter"
    bits = _write_drafter(folder, DFLASH)
    draft_repo = HuggingFaceRepo(
        str(tmp_path if subfolder else folder), subfolder=subfolder
    )
    pipeline_config = _pipeline_config(
        HuggingFaceRepo(str(tmp_path)), draft_repo
    )
    model = tiny_configs()
    kv_params = _kv_params(UnifiedDflashMiMoV2Config, pipeline_config, model)

    assert set(kv_params.children) == {"target", DRAFT}
    target_kv = kv_params.children["target"]
    draft_kv = kv_params.children[DRAFT]
    assert isinstance(target_kv, MultiKVCacheParams)
    assert isinstance(draft_kv, MHAKVCacheParams)
    assert set(target_kv.children) == {SLIDING, FULL}
    assert draft_kv.n_kv_heads == DFLASH["num_key_value_heads"]
    assert draft_kv.head_dim == DFLASH["head_dim"]
    assert draft_kv.num_layers == DFLASH["num_hidden_layers"]
    assert draft_kv.window_size == DFLASH["sliding_window"]
    # Every group of one tree carries the block's width as its draft count.
    for leaf in (*target_kv.children.values(), draft_kv):
        assert isinstance(leaf, KVCacheParams)
        assert leaf.num_draft_tokens == DFLASH["block_size"]

    lengths = _target_from(model, monkeypatch, speculative=True)
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "get_active_tokenizer",
        lambda repo: range(SAMPLEABLE),
    )
    fused = UnifiedDflashMiMoV2Model.__new__(UnifiedDflashMiMoV2Model)
    fused.pipeline_config = pipeline_config
    fused.kv_params = kv_params
    fused.memory_plan = MagicMock(planned_max_length=MAX_SEQ_LEN)
    target = fused._create_model_config({})

    # The verify rows and the block reach past the last committed position.
    assert lengths == [MAX_SEQ_LEN + spec_decode_cache_slack(kv_params)]
    assert target.kv_params is target_kv
    spec = fused._spec
    assert spec.num_speculative_tokens == K
    assert spec.sampleable_vocab_size == SAMPLEABLE
    assert spec.draft.kv_params is draft_kv
    assert fused._draft_state.keys() == bits.keys()
    export = fused.drafter_export
    assert export.directory == folder
    assert (
        export.target_layer_ids == DFLASH["dflash_config"]["target_layer_ids"]
    )
    assert export.num_speculative_tokens == K
    assert export.spec_block_size == DFLASH["block_size"]


@pytest.mark.parametrize("subfolder", [None, "text"])
def test_base_ctx_writes_with_the_checkpoints_own_drafter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subfolder: str | None
) -> None:
    folder = tmp_path / subfolder if subfolder else tmp_path
    _write_drafter(folder / "dflash", DFLASH)
    pipeline_config = _pipeline_config(
        HuggingFaceRepo(str(tmp_path), subfolder=subfolder), None
    )
    model = tiny_configs()
    kv_params = _kv_params(MiMoV2DFlashContextConfig, pipeline_config, model)

    assert list(kv_params.children) == [SLIDING, FULL, DRAFT]
    draft_kv = kv_params.children[DRAFT]
    assert isinstance(draft_kv, KVCacheParams)
    assert draft_kv.num_draft_tokens == 0
    assert draft_kv.window_size == DFLASH["sliding_window"]

    _target_from(model, monkeypatch, speculative=False)
    base_ctx = MiMoV2DFlashContextModel.__new__(MiMoV2DFlashContextModel)
    base_ctx.pipeline_config = pipeline_config
    base_ctx.kv_params = kv_params
    base_ctx.memory_plan = MagicMock(planned_max_length=MAX_SEQ_LEN)
    base_ctx.return_logits = MagicMock()
    config = base_ctx._create_model_config({})
    assert (
        config.target_layer_ids == DFLASH["dflash_config"]["target_layer_ids"]
    )

    assert base_ctx._tap_hook(config) is not None
    assert base_ctx._writer_registry.keys() == {
        f"{DRAFT}.{name}" for name in WRITER_WEIGHTS
    }
    export = base_ctx.drafter_export
    assert export.directory == folder / "dflash"
    assert export.num_speculative_tokens is None
    assert export.spec_block_size is None


def test_the_fused_tree_refuses_dkv(tmp_path: Path) -> None:
    folder = tmp_path / "drafter"
    _write_drafter(folder, DFLASH)
    pipeline_config = _pipeline_config(
        HuggingFaceRepo(str(tmp_path)), HuggingFaceRepo(str(folder))
    )
    dkv = KVCacheConfig(
        enable_prefix_caching=True,
        kv_connector_config=KVConnectorConfig(
            type=KVConnectorType.dkv, block_store_endpoint="localhost:1"
        ),
    )
    with pytest.raises(ValueError, match="dKV"):
        UnifiedDflashMiMoV2Config.construct_kv_params(
            tiny_configs().hf, pipeline_config, REFS, dkv, DType.bfloat16
        )
