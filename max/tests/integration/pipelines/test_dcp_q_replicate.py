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
"""The decode context-parallel query layout and position interleave."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    DeepseekV3_2Config,
    resolve_dcp_q_replicate,
)
from max.pipelines.lib import MAXModelConfig


def test_unset_replicates_the_query() -> None:
    """No flag means a replicated query, as vLLM defaults GLM-5.x."""
    assert MAXModelConfig.model_fields["dcp_q_replicate"].default is None
    assert resolve_dcp_q_replicate(None) is True


@pytest.mark.parametrize("value", [True, False])
def test_an_explicit_value_wins(value: bool) -> None:
    assert resolve_dcp_q_replicate(value) is value


def _config(
    *, dcp: int, interleave: int, page_size: int = 128
) -> DeepseekV3_2Config:
    mla = MLAKVCacheParams(
        dtype=DType.float32,
        head_dim=1,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.CPU()],
        num_q_heads=64,
    )
    return DeepseekV3_2Config(
        dtype=DType.float32,
        kv_params=MultiKVCacheParams.from_params({"mla": mla, "indexer": mla}),
        devices=[DeviceRef.CPU()],
        decode_context_parallel_degree=dcp,
        decode_context_parallel_interleave=interleave,
        hidden_size=1,
        max_seq_len=128,
    )


def test_interleave_defaults_to_token_striping() -> None:
    field = MAXModelConfig.model_fields["decode_context_parallel_interleave"]
    assert field.default == 1
    assert (
        DeepseekV3_2Config.__dataclass_fields__[
            "decode_context_parallel_interleave"
        ].default
        == 1
    )


@pytest.mark.parametrize(
    ("dcp", "interleave", "page_size"),
    [(1, 1, 128), (8, 1, 128), (8, 64, 128), (8, 128, 128), (4, 256, 256)],
)
def test_a_layout_that_divides_the_page_is_accepted(
    dcp: int, interleave: int, page_size: int
) -> None:
    config = _config(dcp=dcp, interleave=interleave, page_size=page_size)
    assert config.decode_context_parallel_interleave == interleave


@pytest.mark.parametrize("interleave", [0, -1])
def test_a_non_positive_interleave_is_rejected(interleave: int) -> None:
    with pytest.raises(ValueError, match="interleave must be positive"):
        _config(dcp=8, interleave=interleave)


def test_an_interleave_needs_a_dcp_group() -> None:
    with pytest.raises(ValueError, match="needs"):
        _config(dcp=1, interleave=128)


@pytest.mark.parametrize("interleave", [3, 96, 256])
def test_an_interleave_must_divide_the_page(interleave: int) -> None:
    with pytest.raises(ValueError, match=r"divide the KV page size \(128\)"):
        _config(dcp=8, interleave=interleave)
