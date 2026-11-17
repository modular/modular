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
"""The text load path's vision split, against real safetensors shards.

S3 put the vision subtree back on the load path -- the vision graph binds
it -- so the split moved from the safetensors view to the graph specs:
the adapter parses the whole checkpoint (the vision table's BF16 check
and the Q/K rope-layout permutation included), the decoder's
``_prepare_state_dict`` drops the vision names at bind time, and the
DSpark ``checkpoint_weights`` still filters its input before the adapter
so the fused graph never materializes a vision tensor. Both filters are
product code, pinned here against two synthetic shards:

* the adapter emits the vision names under the vision module names, the
  conv patch embed's rename included;
* ``_prepare_state_dict`` drops exactly the vision names;
* the DSpark ``checkpoint_weights`` drops the vision names before the
  adapter ever sees them.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from max.graph.weights import SafetensorWeights, load_weights
from max.pipelines.architectures.mistral_large4_modulev3.model import (
    MistralLarge4Model,
)
from max.pipelines.architectures.mistral_large4_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.architectures.mistral_large4_modulev3.weight_contract import (
    is_vision_name,
)
from transformers import PretrainedConfig

_TEXT_SHARD0 = "layers.0.attention.wo.weight"
_DSPARK = "mtp_dspark.backbone.target_norm.weight"
_VISION = "vision_encoder.ln_pre.weight"
_PATCH_CONV = "vision_encoder.patch_conv.weight"
_TEXT_SHARD1 = "layers.1.attention.wv.weight"


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    """Writes one real safetensors shard, bytes verbatim.

    The header spellings are safetensors' own, so ``load_weights`` reads
    each tensor back at its stored dtype -- the lazy view a real
    checkpoint load hands the adapter, never a hand-built mapping.
    """
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", len(blob)))
        file.write(blob)
        for chunk in blobs:
            file.write(chunk)


def _checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """Two shards: text and drafter names in the first, vision in the second."""
    shard0 = tmp_path / "consolidated-00001.safetensors"
    _write_safetensors(
        shard0,
        {
            _TEXT_SHARD0: ("F32", (4, 4), np.arange(16, dtype="<f4").tobytes()),
            _DSPARK: (
                "F32",
                (8,),
                np.arange(8, 0, -1, dtype="<f4").tobytes(),
            ),
        },
    )
    shard1 = tmp_path / "consolidated-00002.safetensors"
    _write_safetensors(
        shard1,
        {
            # BF16 bits, as the vision subtree stores them: the tower
            # norm, and the conv patch embed whose `.weight` suffix the
            # vision table strips.
            _VISION: ("BF16", (16,), np.arange(16, dtype="<u2").tobytes()),
            _PATCH_CONV: (
                "BF16",
                (2,),
                np.arange(2, 0, -1, dtype="<u2").tobytes(),
            ),
            _TEXT_SHARD1: (
                "F32",
                (4, 4),
                np.arange(100, 116, dtype="<f4").tobytes(),
            ),
        },
    )
    return shard0, shard1


def test_the_load_splits_the_vision_subtree_at_bind_time(
    tmp_path: Path,
) -> None:
    """The adapter emits the vision names; the decoder drops them.

    The adapter parses the whole shard set, so the vision names come out
    under the vision module names (the conv patch embed's one rename
    included), and the decoder's bind-time filter drops exactly those --
    the language graph never meets a weight it does not declare.
    """
    shard0, shard1 = _checkpoint(tmp_path)
    lazy = load_weights([shard0, shard1])
    assert isinstance(lazy, SafetensorWeights)
    adapted = convert_safetensor_state_dict(dict(lazy.items()), None)
    assert sorted(name for name in adapted if is_vision_name(name)) == sorted(
        [
            "vision_encoder.ln_pre.weight",
            "vision_encoder.patch_conv",
        ]
    )
    bound = MistralLarge4Model._prepare_state_dict(
        cast(Any, None), adapted, cast(Any, None)
    )
    assert not any(is_vision_name(name) for name in bound)
    assert sorted(bound) == sorted(
        [
            "language_model.layers.0.self_attn.wo.weight",
            "language_model.layers.1.self_attn.wv.weight",
        ]
    )
    # One kept text tensor is the bytes it was written with.
    got = np.from_dlpack(
        bound["language_model.layers.1.self_attn.wv.weight"].data
    )
    np.testing.assert_array_equal(
        got.reshape(4, 4), np.arange(100, 116, dtype=np.float32).reshape(4, 4)
    )


def test_dspark_checkpoint_weights_drop_the_vision_subtree(
    tmp_path: Path,
) -> None:
    """The fused graph's weights never include a vision name."""
    shard0, shard1 = _checkpoint(tmp_path)
    lazy = load_weights([shard0, shard1])
    context = SimpleNamespace(
        adapter=convert_safetensor_state_dict,
        weights=lazy,
        pipeline_config=SimpleNamespace(
            model=SimpleNamespace(huggingface_config=PretrainedConfig())
        ),
    )
    weights = MistralLarge4Model.checkpoint_weights(
        cast(Any, context), cast(Any, None)
    )
    names = sorted(weights)
    assert "language_model.layers.0.self_attn.wo.weight" in names
    assert "language_model.layers.1.self_attn.wv.weight" in names
    assert not any(is_vision_name(name) for name in names)
    assert not any(name.startswith("mtp_dspark.") for name in names)
