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
header (read header-only; the file itself is 7 GB), so the rules see the
real names, and the module-tree check pins the real shapes.
"""

from __future__ import annotations

import glob
import json
import os
import struct
from dataclasses import fields

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
from max.pipelines.architectures.kimi_k3_modulev3.dspark.model_config import (
    K3DSparkDraft,
    K3DSparkModuleConfig,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.weight_adapters import (
    k3_dspark_weight_rules,
)
from max.pipelines.lib import WeightTransform
from max.pipelines.lib.module_registration import construct_module, device_mesh

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


def test_checkpoint_covers_the_registered_draft() -> None:
    """The rules load the checkpoint as stored, minus the confidence head:
    name for name and shape for shape, the parameters of the draft the
    framework constructs."""
    loaded = WeightTransform(k3_dspark_weight_rules(None)).apply_data(
        {name: _weight(name) for name in REAL_HEADER}
    )
    # The drafter serves only through the fused MLA kernels, so the leaf
    # its module builds against is the latent one the kernels require.
    config = K3DSparkModuleConfig(
        **{f.name: getattr(CONFIG, f.name) for f in fields(CONFIG)},
        kv_params=MLAKVCacheParams(
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
    with F.lazy():
        draft = construct_module(
            K3DSparkDraft, config, device_mesh([CPU()], data_parallel_degree=1)
        )
    shapes = {
        name: tuple(int(dim) for dim in value.shape)
        for name, value in draft.parameters
    }
    assert shapes == {name: REAL_HEADER[name][1] for name in loaded}


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
