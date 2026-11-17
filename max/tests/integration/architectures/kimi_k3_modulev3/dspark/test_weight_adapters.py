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
"""CPU checks of the DSpark draft's checkpoint contract. No graph, no GPU.

Every tensor name below comes from the published checkpoint's safetensors
header (read header-only; the file itself is 7 GB), so the round-trip pins
the real names, and the module-tree check pins the real shapes.
"""

from __future__ import annotations

import glob
import json
import os
import struct

import numpy as np
import pytest
from _dspark_test_data import DRAFT_CONFIG, REAL_HEADER
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, Shape
from max.graph.weights import WeightData
from max.nn.kv_cache import MLAKVCacheParams
from max.pipelines.architectures.kimi_k3_modulev3.dspark.config import (
    parse_k3_dspark_config,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.dspark import K3DSpark
from max.pipelines.architectures.kimi_k3_modulev3.dspark.weight_adapters import (
    convert_k3_dspark_state_dict,
    expected_k3_dspark_weight_names,
)
from max.tensor import default_device, default_dtype

CONFIG = parse_k3_dspark_config(DRAFT_CONFIG)
CHECKPOINT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Inferact--Kimi-K3-DSpark/snapshots"
)


def _weight(name: str) -> WeightData:
    """A small BF16 tensor standing in for one checkpoint entry."""
    shape = (2, 2)
    bits = np.zeros(shape, dtype=np.uint16)
    return WeightData(
        data=Buffer.from_numpy(bits.reshape(-1).view(np.uint8)).view(
            DType.bfloat16, shape
        ),
        name=name,
        dtype=DType.bfloat16,
        shape=Shape(shape),
    )


def _checkpoint() -> dict[str, WeightData]:
    return {name: _weight(name) for name in REAL_HEADER}


def test_round_trips_every_real_checkpoint_name() -> None:
    weights = convert_k3_dspark_state_dict(_checkpoint(), CONFIG)
    expected = expected_k3_dspark_weight_names(CONFIG)
    assert set(weights) == expected
    # No renames: every loaded tensor keeps the name the checkpoint gave it.
    for name, value in weights.items():
        assert value.name == name


def test_drops_the_confidence_head() -> None:
    """The driver consumes no confidence head, so its tensors never load."""
    weights = convert_k3_dspark_state_dict(_checkpoint(), CONFIG)
    assert not any(name.startswith("confidence_head.") for name in weights)


def test_expected_names_match_the_real_header() -> None:
    expected = expected_k3_dspark_weight_names(CONFIG)
    assert expected == {
        name for name in REAL_HEADER if not name.startswith("confidence_head.")
    }


def test_refuses_a_missing_tensor() -> None:
    state = _checkpoint()
    del state["layers.3.self_attn.o_proj.weight"]
    with pytest.raises(ValueError, match="Missing"):
        convert_k3_dspark_state_dict(state, CONFIG)


def test_refuses_an_unknown_tensor() -> None:
    state = _checkpoint()
    state["lm_head.weight"] = _weight("lm_head.weight")
    with pytest.raises(ValueError, match="unexpected"):
        convert_k3_dspark_state_dict(state, CONFIG)


def test_module_tree_matches_the_real_header_shapes() -> None:
    # The drafter serves only through the fused MLA kernels, so the leaf
    # its module builds against is the latent one the kernels require,
    # which also pins the latent-width contract the checkpoint must carry.
    with F.lazy(), default_dtype(DType.bfloat16), default_device(CPU()):
        draft = K3DSpark(
            CONFIG,
            MLAKVCacheParams(
                dtype=DType.bfloat16,
                head_dim=CONFIG.kv_lora_rank + CONFIG.qk_rope_head_dim,
                num_layers=CONFIG.num_hidden_layers,
                devices=[DeviceRef.CPU()],
                page_size=128,
                num_q_heads=CONFIG.num_attention_heads,
            ),
            max_seq_len=64,
            mla_buffer_size=128,
        )
        state = dict(draft.parameters)
    assert set(state) == expected_k3_dspark_weight_names(CONFIG)
    for name, (_, shape) in REAL_HEADER.items():
        if name.startswith("confidence_head."):
            continue
        got = tuple(int(dim) for dim in state[name].shape)
        assert got == shape, f"{name}: module {got} vs checkpoint {shape}"


def test_real_header_matches_the_embedded_one() -> None:
    """Cross-checks the embedded header against the real file, if present."""
    files = glob.glob(os.path.join(CHECKPOINT, "*", "model.safetensors"))
    if not files:
        pytest.skip("the Inferact/Kimi-K3-DSpark checkpoint is not cached")
    with open(files[0], "rb") as fh:
        count = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(count))
    real = {
        name: (entry["dtype"], tuple(entry["shape"]))
        for name, entry in header.items()
        if name != "__metadata__"
    }
    assert real == REAL_HEADER
