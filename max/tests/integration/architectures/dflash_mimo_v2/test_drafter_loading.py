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
"""Tests the MiMo-V2.6-Flash DFlash drafter's verify width and the drafter
files it reads out of a checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.dflash_mimo_v2 import DFlashMiMoV2Config
from max.pipelines.architectures.dflash_mimo_v2.drafter import (
    load_drafter,
    mimo_dflash_draft_width,
    repo_file,
)
from max.pipelines.architectures.dflash_mimo_v2.weight_adapters import (
    DRAFT_WEIGHTS_FILE,
    MASK_EMBEDDING_FILE,
    drafter_tensor_shapes,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.weights import HuggingFaceRepo
from safetensors.torch import save_file
from tiny_mimo_dflash import (
    drafter_kv_params,
    tiny_dflash_config,
)

DRAFT_HF = {"dflash_config": {"mask_token_id": 151675, "target_layer_ids": [0]}}


def _width(k: int | None, block: int = 8) -> int:
    return mimo_dflash_draft_width(
        SpeculativeConfig(
            speculative_method="dflash", num_speculative_tokens=k
        ),
        None,
        {**DRAFT_HF, "block_size": block},
    )


def test_verify_width_is_decoupled_from_the_block() -> None:
    assert _width(None) == 7
    for k in range(1, 8):
        assert _width(k) == k
    for k in (0, 8):
        with pytest.raises(ValueError, match="verifies 1 to 7"):
            _width(k)


DFLASH = tiny_dflash_config([0, 1, 2])


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
