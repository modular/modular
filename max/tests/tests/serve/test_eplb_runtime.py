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
"""Tests the EplbRuntime helper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from max.dtype import DType
from max.nn.comm.ep import EPConfig
from max.nn.layer import LayerList
from max.pipelines.lib.eplb_runtime import EPLB_STATS_ENV, EplbRuntime
from max.pipelines.lib.eplb_stats import EplbStatsMetadata, EplbStatsSnapshot
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig

NUM_LOGICAL = 16
EP_SIZE = 4

_METADATA = EplbStatsMetadata(
    num_layers=3,
    num_moe_layers=2,
    moe_layer_indices=(1, 2),
    num_logical_experts=NUM_LOGICAL,
    num_experts_per_token=2,
)


def _ep_config() -> EPConfig:
    return EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=64,
        top_k=2,
        n_experts=NUM_LOGICAL,
        max_tokens_per_rank=16,
        n_gpus_per_node=EP_SIZE,
        n_nodes=1,
    )


def _write_snapshot(
    path: Path, metadata: EplbStatsMetadata = _METADATA
) -> Path:
    histogram = np.zeros(
        (metadata.num_layers, metadata.num_logical_experts), dtype=np.int64
    )
    for row in metadata.moe_layer_indices:
        histogram[row] = 1
        histogram[row, row] = 10_000
    snap = EplbStatsSnapshot.from_array(metadata, histogram)
    path.write_text(json.dumps(snap.to_dict()))
    return path


def test_no_snapshot_leaves_config_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EPLB_STATS_ENV, raising=False)
    runtime = EplbRuntime(PipelineRuntimeConfig(), _METADATA, devices=[])
    config = _ep_config()

    assert runtime.install_placement(config, n_groups=1) is None
    assert not config.eplb_enabled
    assert config.n_experts == NUM_LOGICAL
    state_dict = {"w": np.zeros(1)}
    assert runtime.remap_weights(LayerList([]), state_dict) is state_dict


def test_snapshot_installs_per_layer_placement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        EPLB_STATS_ENV, str(_write_snapshot(tmp_path / "snap.json"))
    )
    runtime = EplbRuntime(
        PipelineRuntimeConfig(eplb_redundant_experts_per_gpu=1),
        _METADATA,
        devices=[],
    )
    config = _ep_config()

    placement = runtime.install_placement(config, n_groups=1)

    assert placement is not None
    assert config.eplb_enabled
    assert config.n_experts == NUM_LOGICAL + EP_SIZE
    assert config.num_logical_experts == NUM_LOGICAL
    assert config.num_moe_layers == _METADATA.num_layers
    assert config.max_replicas == placement.max_replicas
    assert config.eplb_phy2log_plan is placement.phy2log
    # Each MoE layer replicates its own hot expert.
    assert placement.logcnt[1, 1] > 1 and placement.logcnt[1, 2] == 1
    assert placement.logcnt[2, 2] > 1 and placement.logcnt[2, 1] == 1


def test_mismatched_snapshot_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other = EplbStatsMetadata(
        num_layers=3,
        num_moe_layers=2,
        moe_layer_indices=(1, 2),
        num_logical_experts=NUM_LOGICAL * 2,
        num_experts_per_token=2,
    )
    monkeypatch.setenv(
        EPLB_STATS_ENV, str(_write_snapshot(tmp_path / "snap.json", other))
    )
    runtime = EplbRuntime(PipelineRuntimeConfig(), _METADATA, devices=[])
    with pytest.raises(ValueError, match="doesn't match model"):
        runtime.install_placement(_ep_config(), n_groups=1)


@pytest.mark.parametrize(
    ("redundant", "profile", "folded"),
    [(0, False, True), (1, False, False), (0, True, False)],
)
def test_router_fold_needs_a_permutation_and_no_profiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    redundant: int,
    profile: bool,
    folded: bool,
) -> None:
    monkeypatch.setenv(
        EPLB_STATS_ENV, str(_write_snapshot(tmp_path / "snap.json"))
    )
    runtime = EplbRuntime(
        PipelineRuntimeConfig(
            eplb_redundant_experts_per_gpu=redundant, eplb_profile=profile
        ),
        _METADATA,
        devices=[],
    )
    config = _ep_config()
    runtime.install_placement(config, n_groups=1)
    assert config.eplb_router_folded is folded
