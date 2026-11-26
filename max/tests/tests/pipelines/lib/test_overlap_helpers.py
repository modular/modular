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
"""Tests the overlap helpers an export ships: which ones, how they are
sized, and that an artifact stores them as hashed ``model`` roles."""

from __future__ import annotations

import json
import unittest.mock
from pathlib import Path

import numpy as np
import pytest
from max._artifact import overlap_helpers
from max._artifact.model_artifact import ModelArtifact
from max._artifact.overlap_helpers import (
    REALIZE_INPLACE_MEF,
    REALIZE_MEF,
    SPEC_REALIZE_INPLACE_MEF,
    SPEC_REALIZE_MIXED_INPLACE_MEF,
    HelperRole,
    SpecFeed,
    artifact_helper_builds,
    attention_cache_count,
    compile_helpers,
    helper_builds,
)
from max._artifact.pipeline_modules import PipelineModules
from max._artifact.serialization import deserialize, serialize
from max._artifact.value_types import ArtifactError
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
)
from max.tensor.sharding import BufferLayout

_PLAIN = 5


def _tree(**children: object) -> MultiKVCacheParams:
    tree = unittest.mock.MagicMock(spec=MultiKVCacheParams)
    tree.children = children
    return tree


def test_the_plain_helpers_are_shipped_without_a_spec_feed() -> None:
    names = [helper.name for helper in helper_builds(None)]
    assert len(names) == _PLAIN
    assert REALIZE_MEF in names
    assert REALIZE_INPLACE_MEF in names
    assert all(helper.n_devices == 1 for helper in helper_builds(None))


def test_a_registry_spec_feed_spans_the_export_devices() -> None:
    builds = helper_builds(
        SpecFeed(tp=2, kv_groups=3, dp=2, next_draft_dtype=DType.int32)
    )
    by_name = {helper.name: helper for helper in builds}
    assert by_name[SPEC_REALIZE_INPLACE_MEF].n_devices == 2
    assert by_name[SPEC_REALIZE_MIXED_INPLACE_MEF].n_devices == 2


def test_attention_caches_are_counted_without_recurrent_state() -> None:
    params = _tree(
        target=_tree(
            sliding=unittest.mock.MagicMock(spec=MHAKVCacheParams),
            full=unittest.mock.MagicMock(spec=MHAKVCacheParams),
            mamba=unittest.mock.MagicMock(spec=RecurrentStateParams),
        ),
        draft=unittest.mock.MagicMock(spec=MLAKVCacheParams),
    )
    assert attention_cache_count(params) == 3


def test_a_plain_artifact_ships_the_token_feeds() -> None:
    builds = artifact_helper_builds(None)
    assert [helper.name for helper in builds] == [
        HelperRole.REALIZE_FUTURE_TOKENS.value,
        HelperRole.REALIZE_FUTURE_TOKENS_INPLACE.value,
    ]
    assert all(helper.n_devices == 1 for helper in builds)


@pytest.mark.parametrize("dp", [0, 1, 4])
def test_a_speculative_artifact_ships_every_feed_sliced_by_replica(
    dp: int,
) -> None:
    builds = artifact_helper_builds(SpecFeed(tp=4, kv_groups=2, dp=dp))
    by_name = {helper.name: helper for helper in builds}
    assert sorted(by_name) == sorted(role.value for role in HelperRole)
    devices = [DeviceRef.CPU()] * 4
    for role in (
        HelperRole.SPEC_REALIZE_INPLACE,
        HelperRole.SPEC_REALIZE_MIXED_INPLACE,
    ):
        helper = by_name[role.value]
        assert helper.n_devices == 4
        graph = helper.build(devices)
        # Every layout takes the replica bounds last, one more than the
        # replicas: a tensor-parallel export is one replica.
        bounds = graph.inputs[-1].tensor
        assert int(bounds.shape[0]) == max(dp, 1) + 1
        # Tokens, drafts, then each (group, device) cache length.
        caches = [value.buffer for value in graph.inputs[2 : 2 + 2 * 4]]
        assert all(cache.dtype == DType.uint32 for cache in caches)


def test_the_artifact_feeds_advance_positions_for_an_mrope_graph() -> None:
    builds = artifact_helper_builds(
        SpecFeed(tp=2, kv_groups=1, dp=2, positions=True)
    )
    for helper in builds:
        if helper.name not in (
            HelperRole.SPEC_REALIZE_INPLACE.value,
            HelperRole.SPEC_REALIZE_MIXED_INPLACE.value,
        ):
            continue
        graph = helper.build([DeviceRef.CPU()] * 2)
        # The positions follow the two cache lengths.
        positions = graph.inputs[4].buffer
        assert positions.dtype == DType.int64
        assert int(positions.shape[0]) == 3


def _executed(graph: object, *buffers: np.ndarray) -> None:
    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)  # type: ignore[arg-type]
    model.execute(*[Buffer.from_numpy(value) for value in buffers])


def test_the_mixed_feed_lands_each_row_at_its_token_offset() -> None:
    """The mixed feed lands each bonus token at its packed offset, widens
    int32 drafts, and advances only the verify row's position window, which
    a longer prompt chunk ahead of it moves off the equal split."""
    feed = SpecFeed(
        tp=1, kv_groups=1, dp=1, next_draft_dtype=DType.int32, positions=True
    )
    (mixed,) = [
        helper
        for helper in artifact_helper_builds(feed)
        if helper.name == HelperRole.SPEC_REALIZE_MIXED_INPLACE.value
    ]
    graph = mixed.build([DeviceRef.from_device(CPU())])
    # Row 0 prefills a 5-token chunk plus its 2 drafts; row 1 verifies its
    # bonus token and 2 drafts from token offset 7.
    tokens = np.zeros(10, dtype=np.int64)
    drafts = np.zeros((2, 2), dtype=np.int64)
    cache = np.full(2, 10, dtype=np.uint32)
    positions = np.tile(np.arange(10, dtype=np.int64), (3, 1))
    _executed(
        graph,
        tokens,
        drafts,
        cache,
        positions,
        np.array([1, 0], dtype=np.int64),  # accepted
        np.array([7, 8], dtype=np.int64),  # next tokens
        np.array([[1, 2], [3, 4]], dtype=np.int32),  # next drafts
        np.array([1, 2], dtype=np.int64),  # previous row -> current row
        np.array([7, 10], dtype=np.int64),  # token offsets; 10 is the skip
        np.array([0, 2], dtype=np.int64),  # replica bounds
    )
    np.testing.assert_array_equal(tokens, [0] * 7 + [7, 0, 0])
    np.testing.assert_array_equal(drafts, [[0, 0], [1, 2]])
    np.testing.assert_array_equal(cache, [10, 12])
    expected = np.arange(10, dtype=np.int64)
    expected[7:] += 2
    np.testing.assert_array_equal(positions, np.tile(expected, (3, 1)))


def _stub_compile(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replaces the engine so a helper compiles to a MEF writer, and returns
    the device count of each session it opens."""
    sessions: list[int] = []

    class FakeSession:
        def __init__(self, devices: list[CPU]) -> None:
            sessions.append(len(devices))

        def compile(self, graph: Graph) -> object:
            name = graph.name

            def write(path: str | Path) -> None:
                Path(path).write_text(name)

            return unittest.mock.Mock(export_mef=write)

    monkeypatch.setattr(overlap_helpers, "InferenceSession", FakeSession)
    return sessions


def test_compiled_helpers_are_hashed_model_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = _stub_compile(monkeypatch)
    devices = [CPU()]
    helpers = compile_helpers(
        devices, artifact_helper_builds(SpecFeed(tp=1, kv_groups=2, dp=1))
    )
    assert sorted(helpers) == sorted(role.value for role in HelperRole)
    assert sessions == [1] * len(HelperRole)
    inplace = helpers[HelperRole.SPEC_REALIZE_INPLACE.value]
    # Tokens, drafts and two cache lengths are written in place.
    assert [
        isinstance(layout, BufferLayout) for layout in inplace.in_layouts
    ] == [True] * 4 + [False] * 5
    assert inplace.out_layouts == []

    dest = serialize(
        ModelArtifact(model=PipelineModules(helpers)), tmp_path / "a"
    )
    files = json.loads((dest / "metadata.json").read_text())["files"]
    for role in HelperRole:
        assert f"{role.value}/model.mef" in files, role
    loaded = deserialize(dest).model
    assert loaded is not None
    assert sorted(loaded.roles) == sorted(helpers)
    realize = loaded.roles[HelperRole.REALIZE_FUTURE_TOKENS.value]
    assert len(realize.in_layouts) == 3
    assert len(realize.out_layouts) == 1


def test_a_helper_wider_than_the_export_is_refused() -> None:
    with pytest.raises(ArtifactError, match="spans 8 devices"):
        compile_helpers(
            [CPU()], artifact_helper_builds(SpecFeed(tp=8, kv_groups=1))
        )
