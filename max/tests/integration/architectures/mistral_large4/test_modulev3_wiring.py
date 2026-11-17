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
"""Config-to-layer wiring and the LeRoPE frequency table.

Both are worth a unit test because both are silent-wrong-answer traps. A layer
that rotates its key when its KV source does not, or a cache leaf whose slot
count disagrees with the schedule, produces plausible-looking wrong logits. And
``low = 3`` / ``high = 16`` with the twelve blended frequencies is the whole
numerical contract of LeRoPE over YaRN, so the values are pinned here from
``reference-math.md`` §1.3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.pipelines.architectures.mistral_large4_modulev3 import (
    NOPE_MLA_GROUP,
    ROPE_MLA_GROUP,
    indexer_slot_map,
    kv_slot_map,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers.rope import (
    MistralLarge4RotaryEmbedding,
    lerope_inv_freq,
    yarn_inv_freq,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    MistralLarge4NormalizedConfig,
)
from max.tensor import Tensor, default_device

TESTDATA = Path("max/tests/integration/architectures/mistral_large4/testdata")
"""Where the M0 checkpoint snapshots live, relative to the runfiles root."""

_YARN = dict(
    theta=10000.0,
    factor=2048.0,
    beta_fast=32.0,
    beta_slow=1.0,
    original_max_position_embeddings=512,
)

# The 8-layer fixture's schedule: NoPE at 3 and 7, layer 7 reads layer 3, and
# layers 5 and 6 carry layer 4's top-k forward (fixture.md §2).
_FIXTURE_ROPE = [True, True, True, False, True, True, True, False]
_FIXTURE_KV_SOURCE = [0, 1, 2, 3, 4, 5, 6, 3]
_FIXTURE_INDEXER = ["full"] * 5 + ["shared", "shared", "full"]


def test_kv_slot_map_rejects_a_rope_reader_of_a_nope_source() -> None:
    """A layer that rotates its key may only read a layer that also rotates."""
    with pytest.raises(ValueError):
        kv_slot_map([True, False], [0, 0])


def test_mla_slot_map_is_dense_per_leaf() -> None:
    """Rope slots run 0..5 for layers 0,1,2,4,5,6; NoPE slot 0 is layer 3."""
    kv = kv_slot_map(
        [True, True, True, False, True, True, True, False],
        [0, 1, 2, 3, 4, 5, 6, 3],
    )
    assert kv[ROPE_MLA_GROUP] == {
        0: 0,
        1: 1,
        2: 2,
        4: 3,
        5: 4,
        6: 5,
    }
    # The reader takes its source's slot, so it adds no storage.
    assert kv[NOPE_MLA_GROUP] == {3: 0, 7: 0}


def test_indexer_slot_map_counts_full_layers_only() -> None:
    """A ``shared`` layer holds no indexer slot."""

    assert indexer_slot_map(
        ["full", "full", "full", "full", "full", "shared", "shared", "full"]
    ) == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        7: 5,
    }


def test_full_checkpoint_schedule_counts() -> None:
    """The 72-layer checkpoint: 44 indexers, 63 KV owners, 18 NoPE layers."""
    params = json.loads((TESTDATA / "ml4_nvfp4_params.json").read_text())
    normalized = MistralLarge4NormalizedConfig.from_params_json(params)
    assert normalized.n_layers == 72
    assert len(normalized.layers.nope_layers) == 18
    assert len(normalized.layers.indexer_owner_layers) == 44
    assert len(normalized.layers.indexer_sharer_layers) == 28
    assert len(normalized.layers.kv_owner_layers) == 63
    assert normalized.layers.kv_sharer_layers == tuple(range(39, 72, 4))
    assert normalized.yarn.correction_range == (3, 16)


def test_rope_ramp_bounds_and_table() -> None:
    """Pins low 3 / high 16 and the blended frequencies of §1.3."""
    inv_freq = yarn_inv_freq(
        theta=10000.0,
        factor=2048.0,
        beta_fast=32.0,
        beta_slow=1.0,
        dim=64,
        original_max_position_embeddings=512,
    )
    assert inv_freq.shape == (32,)
    expected = [
        1.0000000000,
        0.7498942093,
        0.5623413252,
        0.4216965034,
        0.2919144308,
        0.2006725120,
        0.1368107616,
        0.0923407494,
        0.0615572416,
        0.0403958186,
        0.0259690001,
        0.0162317675,
        0.0097407749,
        0.0054813078,
        0.0027431616,
        0.0010317962,
    ]
    for index, value in enumerate(expected):
        assert abs(float(inv_freq[index]) - value) < 1e-7, index
    # Dims 0-3 are unscaled, dim 16 and beyond are divided by the factor.
    assert abs(float(inv_freq[0]) - 1.0) < 1e-7
    assert abs(float(inv_freq[3]) - 10000.0 ** (-6 / 64)) < 1e-7
    assert abs(float(inv_freq[16]) - 10000.0 ** (-32 / 64) / 2048.0) < 1e-7


def test_lerope_multiplies_the_finished_frequencies() -> None:
    """LeRoPE is exp(alpha) on YaRN's output, never on the ramp bounds."""
    yarn_kwargs = dict(
        theta=10000.0,
        factor=2048.0,
        beta_fast=32.0,
        beta_slow=1.0,
    )
    base = yarn_inv_freq(
        dim=64, original_max_position_embeddings=512, **yarn_kwargs
    )
    zeros = lerope_inv_freq(
        np.zeros(32, dtype=np.float32),
        original_max_position_embeddings=512,
        **yarn_kwargs,
    )
    assert np.allclose(base, zeros)
    doubled = lerope_inv_freq(
        np.full(32, np.log(2.0), dtype=np.float32),
        original_max_position_embeddings=512,
        **yarn_kwargs,
    )
    assert np.allclose(doubled, base * 2.0)


def test_rope_table_shape_and_position_rows() -> None:
    """``freqs_cis`` is ``[2 * max_seq_len, 64]``; position 0 is identity."""

    rope = MistralLarge4RotaryEmbedding(
        dim=64,
        max_seq_len=16,
        theta=10000.0,
        factor=2048.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_position_embeddings=512,
    )
    with default_device(CPU()):
        table = rope.freqs_cis_from(
            Tensor.zeros((32,), dtype=DType.bfloat16)
        ).to_numpy()
    assert table.shape == (32, 64)
    # The table is cos/sin interleaved per angle: position 0's rotation is the
    # identity, so its cos columns are 1 and its sin columns 0.
    assert np.allclose(table[0, 0::2], 1.0, atol=1e-3)
    assert np.allclose(table[0, 1::2], 0.0, atol=1e-3)
