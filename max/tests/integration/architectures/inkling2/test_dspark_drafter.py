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
"""Checks the Inkling 2 DSpark drafter module against its checkpoint.

``testdata/mtp_safetensors_header.json`` is the header of
thinkingmachines/Inkling-2-Small's ``mtp.safetensors``: every tensor's name,
dtype and shape. ``Module.compile`` drops a weight no parameter matches, so
the module's parameters and the header must be equal in both directions.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.inkling2.arch import (
    inkling2_arch,
    inkling2_dspark_draft_width,
)
from max.pipelines.architectures.inkling2.dspark import (
    DSPARK_GLOBAL_ATTENTION,
    DSPARK_LOCAL_ATTENTION,
    Inkling2DSpark,
    construct_dspark_kv_params,
)
from max.pipelines.architectures.inkling2.dspark_config import (
    Inkling2DSparkConfig,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import default_dtype

_HEADER: dict[str, dict[str, Any]] = json.loads(
    (
        Path(__file__).parent / "testdata" / "mtp_safetensors_header.json"
    ).read_text()
)

_SAFETENSORS_DTYPES = {"BF16": DType.bfloat16, "F32": DType.float32}

MTP_CONFIG: dict[str, Any] = {
    "hidden_size": 4608,
    "num_hidden_layers": 5,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 18432,
    "max_position_embeddings": 1048576,
    "torch_dtype": "bfloat16",
    "q_bias": False,
    "o_bias": False,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "local_layer_ids": [0, 1, 3, 4],
    "sliding_window_size": 1024,
    "block_size": 8,
    "markov_rank": 256,
    "markov_head_type": "rnn_hidden",
    "target_layer_ids": [5, 11, 17, 23, 29, 35, 41, 48],
}


def _hf_config(**mtp_overrides: Any) -> SimpleNamespace:
    return SimpleNamespace(
        mtp_config={**MTP_CONFIG, **mtp_overrides},
        text_config=SimpleNamespace(
            hidden_size=4608, num_hidden_layers=48, vocab_size=201024
        ),
    )


def _config() -> Inkling2DSparkConfig:
    return Inkling2DSparkConfig.from_hf(_hf_config())


def _kv_params(config: Inkling2DSparkConfig) -> Any:
    return construct_dspark_kv_params(
        config, KVCacheConfig(), [DeviceRef.CPU()], DType.bfloat16
    )


def _drafter(config: Inkling2DSparkConfig) -> Inkling2DSpark:
    """Builds the drafter on the host, lazily, in bfloat16."""
    with F.lazy(), default_dtype(DType.bfloat16):
        return Inkling2DSpark(
            config, _kv_params(config), max_seq_len=4096, device=CPU()
        )


def _declared(
    module: Inkling2DSpark,
) -> dict[str, tuple[DType, tuple[int, ...]]]:
    return {
        name: (param.dtype, tuple(int(dim) for dim in param.shape))
        for name, param in module.parameters
    }


def test_the_module_declares_every_checkpoint_tensor() -> None:
    shipped = {
        name: (_SAFETENSORS_DTYPES[entry["dtype"]], tuple(entry["shape"]))
        for name, entry in _HEADER.items()
    }
    declared = _declared(_drafter(_config()))
    assert sorted(set(shipped) - set(declared)) == []
    assert sorted(set(declared) - set(shipped)) == []
    assert declared == shipped


def test_the_config_describes_the_same_tensors() -> None:
    shapes = _config().weight_shapes()
    assert shapes == {
        name: tuple(entry["shape"]) for name, entry in _HEADER.items()
    }


def test_layers_split_across_the_two_draft_caches() -> None:
    drafter = _drafter(_config())
    local, full = DSPARK_LOCAL_ATTENTION, DSPARK_GLOBAL_ATTENTION
    assert drafter.layer_cache_keys == [local, local, full, local, local]
    assert [layer.self_attn.layer_idx_in_cache for layer in drafter.layers] == [
        0,
        1,
        0,
        2,
        3,
    ]
    assert [layer.self_attn.window for layer in drafter.layers] == [
        1032,
        1032,
        None,
        1032,
        1032,
    ]


def test_the_draft_caches_size_one_full_and_four_windowed_layers() -> None:
    children = _kv_params(_config()).children
    assert set(children) == {DSPARK_GLOBAL_ATTENTION, DSPARK_LOCAL_ATTENTION}
    full, local = (
        children[DSPARK_GLOBAL_ATTENTION],
        children[DSPARK_LOCAL_ATTENTION],
    )
    assert (full.num_layers, full.window_size) == (1, None)
    # A local layer reads the window before the block and the block.
    assert (local.num_layers, local.window_size) == (4, 1024 + 8)
    for params in (full, local):
        assert (params.n_kv_heads, params.head_dim) == (8, 128)
        assert params.num_draft_tokens == 8


def test_a_drafter_with_only_full_layers_has_one_cache() -> None:
    config = Inkling2DSparkConfig.from_hf(_hf_config(local_layer_ids=[]))
    assert set(_kv_params(config).children) == {DSPARK_GLOBAL_ATTENTION}


@pytest.mark.parametrize("requested", [None, 7])
def test_the_draft_width_is_the_checkpoint_block(requested: int | None) -> None:
    speculative = SimpleNamespace(num_speculative_tokens=requested)
    assert inkling2_dspark_draft_width(speculative, _hf_config(), None) == 7


def test_another_draft_width_is_refused() -> None:
    speculative = SimpleNamespace(num_speculative_tokens=3)
    with pytest.raises(ValueError, match="exactly 7 tokens"):
        inkling2_dspark_draft_width(speculative, _hf_config(), None)


def test_the_architecture_registers_the_draft_width() -> None:
    assert inkling2_arch.checkpoint_draft_width is inkling2_dspark_draft_width
