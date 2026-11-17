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
"""Tests the overlap helper component of a ``.m`` artifact: which helpers an
export ships, how they are sized, and where they are stored."""

from __future__ import annotations

import unittest.mock
from pathlib import Path

import pytest
from max._artifact import overlap_helpers
from max._artifact.model_artifact import ModelArtifact
from max._artifact.overlap_helpers import (
    REALIZE_INPLACE_MEF,
    REALIZE_MEF,
    SPEC_REALIZE_INPLACE_MEF,
    SPEC_REALIZE_SHARED_INPLACE_MEF,
    OverlapHelpers,
    SpecFeed,
    attention_cache_count,
    export_spec_feeds,
    helper_builds,
    spec_realize_inplace_graph_name,
)
from max._artifact.serialization import deserialize, serialize
from max._artifact.value_types import ArtifactError
from max.driver import CPU
from max.dtype import DType
from max.graph import Graph
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
)

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


def test_a_spec_feed_spans_the_export_devices() -> None:
    builds = helper_builds(
        SpecFeed(tp=2, kv_groups=3), SpecFeed(tp=8, kv_groups=1, m3_dp=1)
    )
    by_name = {helper.name: helper for helper in builds}
    assert by_name[SPEC_REALIZE_INPLACE_MEF].n_devices == 2
    assert by_name[SPEC_REALIZE_SHARED_INPLACE_MEF].n_devices == 8


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


@pytest.mark.parametrize("dp", [1, 2, 4])
def test_a_fused_mtp_export_also_gets_the_shared_cache_feed(dp: int) -> None:
    feed, shared = export_spec_feeds(
        unittest.mock.MagicMock(spec=MLAKVCacheParams),
        n_devices=8,
        data_parallel_degree=dp,
        next_draft_dtype=DType.int64,
        may_share_cache_lengths=True,
    )
    assert feed.m3_dp == 0
    assert shared == SpecFeed(tp=8, kv_groups=1, m3_dp=dp)


def test_the_shared_cache_feed_needs_eight_devices_and_a_fused_drafter() -> (
    None
):
    kv = unittest.mock.MagicMock(spec=MLAKVCacheParams)

    def shared(n_devices: int, may_share: bool) -> SpecFeed | None:
        return export_spec_feeds(
            kv,
            n_devices=n_devices,
            data_parallel_degree=2,
            next_draft_dtype=DType.int64,
            may_share_cache_lengths=may_share,
        )[1]

    assert shared(8, False) is None
    assert shared(2, True) is None
    assert shared(8, True) is not None


def test_the_dp_feed_follows_the_graph_draft_dtype() -> None:
    kv = _tree(
        **{
            f"g{i}": unittest.mock.MagicMock(spec=MLAKVCacheParams)
            for i in range(6)
        }
    )
    feed, shared = export_spec_feeds(
        kv,
        n_devices=8,
        data_parallel_degree=8,
        next_draft_dtype=DType.int32,
        may_share_cache_lengths=False,
    )
    assert feed == SpecFeed(
        tp=8, kv_groups=6, dp=8, next_draft_dtype=DType.int32
    )
    assert shared is None
    assert feed.mixed_build() is not None


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


def test_compile_stores_each_helper_under_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = _stub_compile(monkeypatch)
    builds = helper_builds(SpecFeed(tp=2, kv_groups=2))
    helpers = OverlapHelpers.compile([CPU(), CPU()], builds)
    assert sorted(helpers.files) == sorted(b.name for b in builds)
    assert sorted(sessions) == sorted(b.n_devices for b in builds)

    artifact = ModelArtifact(overlap_helpers=helpers)
    dest = serialize(artifact, tmp_path / "a")
    for name in helpers.files:
        assert (dest / name).is_file(), name
    assert (dest / SPEC_REALIZE_INPLACE_MEF).read_text() == (
        spec_realize_inplace_graph_name(2, 2)
    )

    loaded = deserialize(dest, ModelArtifact)
    assert loaded.overlap_helpers is not None
    for name, path in loaded.overlap_helpers.files.items():
        assert isinstance(path, Path)
        assert path == dest / name


def test_a_helper_wider_than_the_export_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_compile(monkeypatch)
    with pytest.raises(ArtifactError, match="spans 8 devices"):
        OverlapHelpers.compile(
            [CPU()], helper_builds(SpecFeed(tp=8, kv_groups=1))
        )
