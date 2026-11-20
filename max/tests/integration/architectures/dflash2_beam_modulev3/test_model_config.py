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
"""The Beam DFlash2 drafter's config, width and checkpoint naming.

Read from the checkpoint's own ``config.json`` and its tensor header, both
checked in, so a revised checkpoint or a renamed parameter fails here rather
than as lost acceptance on a GPU.
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
from max.pipelines.architectures.dflash2_beam_modulev3 import (
    SERVING_BLOCK_SIZE,
    DFlash2Beam,
    DFlash2BeamConfig,
    dflash2_beam_draft_width,
)
from max.pipelines.architectures.dflash2_beam_modulev3.model_config import (
    drafter_kv_params,
)
from max.pipelines.kv_cache import KVCacheConfig
from max.pipelines.lib.module_registration import device_mesh
from max.pipelines.speculative import SpeculativeConfig
from max.tensor import default_device, default_dtype

TESTDATA = Path(__file__).parent / "testdata"


def _config() -> dict[str, Any]:
    return json.loads((TESTDATA / "beam_dflash2_config.json").read_text())


def _width(num_speculative_tokens: int | None) -> int:
    return dflash2_beam_draft_width(
        SpeculativeConfig(
            speculative_method="dflash2",
            num_speculative_tokens=num_speculative_tokens,
        ),
        None,
        _config(),
    )


def test_the_width_defaults_to_the_model_cards_serving_block() -> None:
    """Unset serves block 8, as vLLM's and SGLang's recipes do; any width
    below the trained 16 serves at ``K + 1``."""
    assert SERVING_BLOCK_SIZE == 8
    assert _width(None) == 7
    assert _width(1) == 1
    assert _width(15) == 15
    with pytest.raises(ValueError, match="below the block of 16"):
        _width(16)


def _kv(k: int) -> Any:
    pipeline_config = SimpleNamespace(
        speculative=SpeculativeConfig(
            speculative_method="dflash2", num_speculative_tokens=k
        ),
        model=SimpleNamespace(data_parallel_degree=1),
    )
    return drafter_kv_params(
        _config(),
        cast(Any, pipeline_config),
        [DeviceRef.CPU()],
        KVCacheConfig(kv_cache_page_size=128),
    )


def test_the_context_cache_is_the_drafters_own_windowed_bf16_leaf() -> None:
    """Five layers of 8 KV heads at 128, windowed at 4096 and carrying the
    block's width; Mach's registry declares the same group."""
    kv = _kv(7)
    assert (kv.num_layers, kv.n_kv_heads, kv.head_dim) == (5, 8, 128)
    assert kv.dtype == DType.bfloat16
    assert kv.window_size == 4096
    assert kv.num_draft_tokens == 8


def test_the_checkpoint_config_parses() -> None:
    config = DFlash2BeamConfig.from_huggingface_config(
        _config(),
        block_size=8,
        devices=[DeviceRef.CPU()],
        kv_params=_kv(7),
        max_seq_len=4096,
    )
    assert config.target_layer_ids == [4, 16, 28, 40, 46]
    assert config.dflash.num_target_layers == 52
    assert config.mask_token_id == 151669
    assert config.dflash.block_size == 16
    assert config.block_size == 8
    assert config.final_logit_softcapping == 30.0
    assert config.rope_theta == 10000.0
    assert config.sliding_window == 4096
    assert (
        config.dflash.conv_kernel_size,
        config.dflash.conv_group_size,
        config.dflash.selector_rank,
        config.dflash.selector_top_k,
    ) == (2, 16, 256, 16)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("is_causal", True),
        ("layer_types", ["full_attention"] + ["sliding_attention"] * 4),
        ("architectures", ["DFlash2DraftModel"]),
        ("hidden_act", "gelu"),
    ],
)
def test_a_convention_the_module_lacks_is_refused(
    key: str, value: object
) -> None:
    config = copy.deepcopy(_config())
    config[key] = value
    with pytest.raises(ValueError, match="Beam DFlash2"):
        DFlash2BeamConfig.from_huggingface_config(
            config,
            block_size=8,
            devices=[DeviceRef.CPU()],
            kv_params=_kv(7),
            max_seq_len=4096,
        )


def test_a_block_past_the_trained_one_is_refused() -> None:
    with pytest.raises(ValueError, match="serves a block of 2 to 16"):
        DFlash2BeamConfig.from_huggingface_config(
            _config(),
            block_size=17,
            devices=[DeviceRef.CPU()],
            kv_params=_kv(7),
            max_seq_len=4096,
        )


def test_the_module_names_every_checkpoint_tensor() -> None:
    """A compiled module drops a weight no parameter names, so names, shapes
    and dtypes must match the checkpoint's header exactly, both ways."""
    header = json.loads((TESTDATA / "beam_dflash2_tensors.json").read_text())
    config = DFlash2BeamConfig.from_huggingface_config(
        _config(),
        block_size=8,
        devices=[DeviceRef.CPU()],
        kv_params=_kv(7),
        max_seq_len=4096,
    )
    with (
        F.lazy(),
        default_dtype(DType.bfloat16),
        default_device(device_mesh([CPU()], 1)),
    ):
        drafter = DFlash2Beam(config)
    params = {
        name: (param.dtype, [int(d) for d in param.shape])
        for name, param in drafter.parameters
    }
    assert set(params) == set(header)
    for name, (dtype, shape) in params.items():
        assert header[name]["dtype"] == "BF16", name
        assert dtype == DType.bfloat16, name
        assert header[name]["shape"] == shape, name
