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
"""The Laguna DFlash drafter's config, width and checkpoint naming.

Read from checked-in copies of the checkpoint's ``config.json`` and tensor
header, so a module that drifts from them fails here rather than as lost
acceptance on a GPU.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.dflash_laguna_modulev3 import (
    DFlashLaguna,
    DFlashLagunaConfig,
    dflash_laguna_draft_width,
)
from max.pipelines.architectures.dflash_laguna_modulev3.model_config import (
    drafter_kv_params,
)
from max.pipelines.architectures.hf_config_shims import LagunaHFConfig
from max.pipelines.kv_cache import KVCacheConfig
from max.pipelines.lib.module_registration import device_mesh
from max.pipelines.speculative import SpeculativeConfig
from max.tensor import default_device, default_dtype

TESTDATA = Path(__file__).parent / "testdata"


def _config() -> dict[str, Any]:
    return json.loads((TESTDATA / "laguna_dflash_config.json").read_text())


def _width(num_speculative_tokens: int | None) -> int:
    return dflash_laguna_draft_width(
        SpeculativeConfig(
            speculative_method="dflash",
            num_speculative_tokens=num_speculative_tokens,
        ),
        None,
        _config(),
    )


def _kv(k: int) -> KVCacheParams:
    pipeline_config = SimpleNamespace(
        speculative=SpeculativeConfig(
            speculative_method="dflash", num_speculative_tokens=k
        ),
        model=SimpleNamespace(data_parallel_degree=1),
    )
    return drafter_kv_params(
        _config(),
        cast(Any, pipeline_config),
        [DeviceRef.CPU()],
        KVCacheConfig(kv_cache_page_size=128),
    )


def _parse(config: Any, block_size: int = 8) -> DFlashLagunaConfig:
    return DFlashLagunaConfig.from_huggingface_config(
        config,
        block_size=block_size,
        devices=[DeviceRef.CPU()],
        kv_params=_kv(7),
        max_seq_len=4096,
    )


def test_the_width_truncates_the_trained_block() -> None:
    """Unset serves the model card's K of 7; the block is causal, so any K
    below the trained 16 serves at ``K + 1``."""
    assert _width(None) == 7
    assert _width(1) == 1
    assert _width(15) == 15
    with pytest.raises(ValueError, match="below the block of 16"):
        _width(16)


def test_the_shim_parses_the_checkpoint_config_unchanged() -> None:
    """The pipeline loads the config through its AutoConfig shim."""
    assert _parse(LagunaHFConfig(**_config())) == _parse(_config())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dflash_config", "causal"), False),
        (("layer_types",), ["full_attention"] + ["sliding_attention"] * 5),
        (("draft_vocab_size",), 100353),
        (("gating",), "per-element"),
        (("architectures",), ["DFlashDraftModel"]),
    ],
)
def test_a_convention_the_module_lacks_is_refused(
    path: tuple[str, ...], value: object
) -> None:
    config = copy.deepcopy(_config())
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="Laguna DFlash"):
        _parse(config)


def test_a_block_past_the_trained_one_is_refused() -> None:
    with pytest.raises(ValueError, match="serves a block of 2 to 16"):
        _parse(_config(), block_size=17)


def test_the_module_names_every_checkpoint_tensor() -> None:
    """Checks parameter names, shapes and dtypes against the pinned tensor
    header."""
    header = json.loads((TESTDATA / "laguna_dflash_tensors.json").read_text())
    with (
        F.lazy(),
        default_dtype(DType.bfloat16),
        default_device(device_mesh([CPU()], 1)),
    ):
        drafter = DFlashLaguna(_parse(_config()))
    params = {
        name: (param.dtype, [int(d) for d in param.shape])
        for name, param in drafter.parameters
    }
    assert len(header) == 69
    assert set(params) == set(header)
    for name, (dtype, shape) in params.items():
        assert header[name]["dtype"] == "BF16", name
        assert dtype == DType.bfloat16, name
        assert header[name]["shape"] == shape, name
