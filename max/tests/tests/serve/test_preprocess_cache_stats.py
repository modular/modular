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
"""The serve half of the preprocess-cache seam: cumulative counts to deltas."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from unittest.mock import MagicMock, call, patch

from max.pipelines.lib.vision_preprocess_cache import VisionPreprocessCache
from max.pipelines.modeling.types import VisionPreprocessCacheStats
from max.serve.pipelines.preprocess_cache_stats import (
    PreprocessCacheStatsRecorder,
)


def _stats(
    *,
    hits: int = 0,
    misses: int = 0,
    evictions: int = 0,
    size_bytes: int = 0,
    capacity_bytes: int = 1024,
) -> VisionPreprocessCacheStats:
    return VisionPreprocessCacheStats(
        hits=hits,
        misses=misses,
        evictions=evictions,
        size_bytes=size_bytes,
        capacity_bytes=capacity_bytes,
    )


class _CachingTokenizer:
    """An architecture that caches preprocessed media and no more."""

    def __init__(
        self, *snapshots: Mapping[str, VisionPreprocessCacheStats]
    ) -> None:
        self._snapshots = list(snapshots)

    def preprocess_cache_stats(
        self,
    ) -> Mapping[str, VisionPreprocessCacheStats]:
        # Every call but the last advances, so a test can script a sequence
        # of observations and then keep reading the final one.
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]


class _ProbingCachingTokenizer(_CachingTokenizer):
    """M3's shape: the admission probe *and* a cache underneath it."""

    def preprocessed_image_mask(
        self, images: list[bytes], messages: list[object]
    ) -> Sequence[bool]:
        return [False] * len(images)


class _RealCacheTokenizer:
    """Reads a real cache, the way an architecture's method does."""

    def __init__(self, cache: VisionPreprocessCache[str]) -> None:
        self._cache = cache

    def preprocess_cache_stats(
        self,
    ) -> Mapping[str, VisionPreprocessCacheStats]:
        return {"image": self._cache.stats()}


class _TextTokenizer:
    """An architecture with no media cache at all."""


def _recorder(
    tokenizer: object,
) -> tuple[PreprocessCacheStatsRecorder, MagicMock]:
    return PreprocessCacheStatsRecorder(tokenizer), MagicMock()


def _record(
    recorder: PreprocessCacheStatsRecorder,
    metrics: MagicMock,
    *,
    carried_media: bool = True,
) -> None:
    target = "max.serve.pipelines.preprocess_cache_stats.METRICS"
    with patch(target, metrics):
        recorder.record(carried_media=carried_media)


class TestDeltas:
    def test_first_observation_publishes_the_whole_cumulative_count(
        self,
    ) -> None:
        """Nothing was reported before it, so all of it is the delta."""
        recorder, metrics = _recorder(
            _CachingTokenizer({"image": _stats(hits=5, misses=3, evictions=2)})
        )

        _record(recorder, metrics)

        metrics.media_preprocess_cache_evictions.assert_called_once_with(
            2, "image"
        )
        metrics.vision_preprocess_cache_hits.assert_called_once_with(5)
        metrics.vision_preprocess_cache_misses.assert_called_once_with(3)

    def test_second_observation_publishes_only_what_accrued(self) -> None:
        recorder, metrics = _recorder(
            _CachingTokenizer(
                {"image": _stats(hits=5, misses=3, evictions=2)},
                {"image": _stats(hits=9, misses=4, evictions=6)},
            )
        )

        _record(recorder, metrics)
        metrics.reset_mock()
        _record(recorder, metrics)

        metrics.media_preprocess_cache_evictions.assert_called_once_with(
            4, "image"
        )
        metrics.vision_preprocess_cache_hits.assert_called_once_with(4)
        metrics.vision_preprocess_cache_misses.assert_called_once_with(1)

    def test_a_restart_rebases_rather_than_emitting_a_negative(self) -> None:
        """``VisionPreprocessCache.__setstate__`` zeroes the counters.

        A cache unpickled into another process starts counting from zero, so
        differencing against the last snapshot from before the boundary
        would hand an OTel counter a negative add. The post-restart value is
        the whole of the unreported delta instead.
        """
        recorder, metrics = _recorder(
            _CachingTokenizer(
                {"image": _stats(hits=40, misses=10, evictions=7)},
                # Same process object, counters zeroed then re-accumulated.
                {"image": _stats(hits=2, misses=1, evictions=0)},
            )
        )

        _record(recorder, metrics)
        metrics.reset_mock()
        _record(recorder, metrics)

        metrics.media_preprocess_cache_evictions.assert_called_once_with(
            0, "image"
        )
        metrics.vision_preprocess_cache_hits.assert_called_once_with(2)
        metrics.vision_preprocess_cache_misses.assert_called_once_with(1)

    def test_a_restart_that_moved_only_one_counter_still_rebases(self) -> None:
        """Any decrease condemns the whole snapshot.

        Evictions can legitimately stay flat across a restart, so a
        per-counter rule would difference hits against a stale baseline and
        still go negative.
        """
        recorder, metrics = _recorder(
            _CachingTokenizer(
                {"image": _stats(hits=40, misses=10, evictions=0)},
                {"image": _stats(hits=2, misses=1, evictions=0)},
            )
        )

        _record(recorder, metrics)
        metrics.reset_mock()
        _record(recorder, metrics)

        for emitted in (
            metrics.vision_preprocess_cache_hits.call_args,
            metrics.vision_preprocess_cache_misses.call_args,
            metrics.media_preprocess_cache_evictions.call_args,
        ):
            assert emitted.args[0] >= 0

    def test_gauges_publish_the_absolute_reading_not_a_delta(self) -> None:
        recorder, metrics = _recorder(
            _CachingTokenizer(
                {"image": _stats(size_bytes=400, capacity_bytes=1024)},
                {"image": _stats(size_bytes=900, capacity_bytes=1024)},
            )
        )

        _record(recorder, metrics)
        metrics.reset_mock()
        _record(recorder, metrics)

        metrics.media_preprocess_cache_size.assert_called_once_with(
            900, "image"
        )
        metrics.media_preprocess_cache_capacity.assert_called_once_with(
            1024, "image"
        )


class TestGates:
    def test_a_text_only_request_publishes_nothing(self) -> None:
        """Tokenization runs for every request, media or not."""
        recorder, metrics = _recorder(
            _CachingTokenizer({"image": _stats(hits=5, evictions=2)})
        )

        _record(recorder, metrics, carried_media=False)

        metrics.media_preprocess_cache_evictions.assert_not_called()
        metrics.media_preprocess_cache_size.assert_not_called()
        metrics.media_preprocess_cache_capacity.assert_not_called()
        metrics.vision_preprocess_cache_hits.assert_not_called()
        metrics.vision_preprocess_cache_misses.assert_not_called()

    def test_an_architecture_without_the_capability_is_silent(self) -> None:
        """A typed absence, not a raise and not a getattr miss."""
        recorder, metrics = _recorder(_TextTokenizer())

        _record(recorder, metrics)

        metrics.media_preprocess_cache_evictions.assert_not_called()
        metrics.media_preprocess_cache_size.assert_not_called()
        metrics.vision_preprocess_cache_hits.assert_not_called()

    def test_a_non_callable_stats_attribute_is_ignored(
        self,
        caplog,  # noqa: ANN001
    ) -> None:
        """The protocol proves the attribute exists, not that it is callable."""

        class BadTokenizer:
            preprocess_cache_stats = {"image": None}

        with caplog.at_level(logging.WARNING, logger="max.serve"):
            recorder, metrics = _recorder(BadTokenizer())
        assert "not callable" in caplog.text

        _record(recorder, metrics)

        metrics.media_preprocess_cache_size.assert_not_called()


class TestAFailingProbeCannotFailTheRequest:
    """``record`` runs inside ``next_token_chunk``'s re-raising ``try``.

    Everything it calls is tokenizer-supplied, and nothing downstream needs
    the answer, so a raise here would turn every media request into a 500
    for the life of the process in exchange for a metric.
    """

    def test_a_stats_call_returning_a_non_mapping_is_survived(self) -> None:
        """The two gates prove the attribute exists and is callable only."""

        class BadTokenizer:
            def preprocess_cache_stats(self) -> Mapping[str, object]:
                # An out-of-tree tokenizer that builds its video cache only
                # when a video processor is configured returns this shape.
                return None  # type: ignore[return-value]

        recorder, metrics = _recorder(BadTokenizer())

        _record(recorder, metrics)

        metrics.media_preprocess_cache_size.assert_not_called()

    def test_a_raising_stats_call_is_survived(self) -> None:
        class RaisingTokenizer:
            def preprocess_cache_stats(
                self,
            ) -> Mapping[str, VisionPreprocessCacheStats]:
                raise RuntimeError("cache went away")

        recorder, metrics = _recorder(RaisingTokenizer())

        _record(recorder, metrics)

        metrics.media_preprocess_cache_size.assert_not_called()

    def test_the_recorder_disables_itself_and_logs_once(
        self,
        caplog,  # noqa: ANN001
    ) -> None:
        """Per-request logging of a permanent failure is its own outage."""
        calls = 0

        class RaisingTokenizer:
            def preprocess_cache_stats(
                self,
            ) -> Mapping[str, VisionPreprocessCacheStats]:
                nonlocal calls
                calls += 1
                raise RuntimeError("cache went away")

        recorder, metrics = _recorder(RaisingTokenizer())

        with caplog.at_level(logging.WARNING, logger="max.serve"):
            for _ in range(3):
                _record(recorder, metrics)

        assert calls == 1
        assert caplog.text.count("publishing no preprocess-cache metrics") == 1

    def test_a_partial_snapshot_still_publishes_what_it_reached(self) -> None:
        """The failure costs the metric, not the reading before it."""

        class HalfBadTokenizer:
            def preprocess_cache_stats(self) -> Mapping[str, object]:
                return {"image": _stats(size_bytes=7), "video": "not a stat"}

        recorder, metrics = _recorder(HalfBadTokenizer())

        _record(recorder, metrics)

        metrics.media_preprocess_cache_size.assert_called_once_with(7, "image")


class TestHitMissSurfacing:
    """One cache, one pair of names -- and never two increments per image."""

    def test_an_architecture_without_the_admission_probe_fills_the_pair(
        self,
    ) -> None:
        """This pair has been structurally empty everywhere but M3."""
        recorder, metrics = _recorder(
            _CachingTokenizer({"image": _stats(hits=5, misses=3)})
        )

        _record(recorder, metrics)

        metrics.vision_preprocess_cache_hits.assert_called_once_with(5)
        metrics.vision_preprocess_cache_misses.assert_called_once_with(3)

    def test_an_architecture_with_the_admission_probe_does_not_double_count(
        self,
    ) -> None:
        """The direct detector for the production double-count on M3.

        The route already records the pair from the admission peek, which
        deliberately does not touch the cache's own counters. Publishing
        both vantages under one name would count a single cached image
        twice, and ``minimax-m3-mxfp8`` is a live modelapp.
        """
        recorder, metrics = _recorder(
            _ProbingCachingTokenizer({"image": _stats(hits=5, misses=3)})
        )

        _record(recorder, metrics)

        metrics.vision_preprocess_cache_hits.assert_not_called()
        metrics.vision_preprocess_cache_misses.assert_not_called()
        # The capacity-pressure metrics are not the probe's to report, so
        # they are published on M3 like anywhere else.
        metrics.media_preprocess_cache_size.assert_called_once_with(0, "image")

    def test_a_probe_the_route_rejects_does_not_mute_this_vantage(
        self,
        caplog,  # noqa: ANN001
    ) -> None:
        """One predicate decides both vantages, so they cannot disagree.

        ``preprocessed_image_probe`` rejects a non-callable attribute, so the
        route publishes nothing from the admission peek. Gating this recorder
        on ``isinstance`` alone would leave the pair empty from both ends --
        the structural zero this seam exists to remove.
        """

        class UnusableProbeTokenizer(_CachingTokenizer):
            preprocessed_image_mask = [True, False]

        with caplog.at_level(logging.WARNING, logger="max.serve"):
            recorder, metrics = _recorder(
                UnusableProbeTokenizer({"image": _stats(hits=5, misses=3)})
            )

        _record(recorder, metrics)

        metrics.vision_preprocess_cache_hits.assert_called_once_with(5)
        metrics.vision_preprocess_cache_misses.assert_called_once_with(3)

    def test_video_lookups_do_not_feed_the_image_pair(self) -> None:
        """``maxserve.vision.preprocess_cache_*`` counts images and is untagged."""
        recorder, metrics = _recorder(
            _CachingTokenizer(
                {
                    "image": _stats(hits=5, misses=3),
                    "video": _stats(hits=11, misses=13),
                }
            )
        )

        _record(recorder, metrics)

        metrics.vision_preprocess_cache_hits.assert_called_once_with(5)
        metrics.vision_preprocess_cache_misses.assert_called_once_with(3)

    def test_each_media_kind_gets_its_own_capacity_series(self) -> None:
        recorder, metrics = _recorder(
            _CachingTokenizer(
                {
                    "image": _stats(size_bytes=10, capacity_bytes=100),
                    "video": _stats(size_bytes=20, capacity_bytes=200),
                }
            )
        )

        _record(recorder, metrics)

        assert metrics.media_preprocess_cache_capacity.call_args_list == [
            call(100, "image"),
            call(200, "video"),
        ]


class TestAgainstARealCache:
    """End of the seam: a real cache overflowing, read the way serve reads it."""

    def test_a_forced_overflow_reports_evictions_and_a_full_cache(
        self,
    ) -> None:
        # Room for two entries, so a third evicts one.
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(20)
        recorder, metrics = _recorder(_RealCacheTokenizer(cache))

        cache.put(1, "one", 10)
        cache.put(2, "two", 10)
        cache.put(3, "three", 10)
        cache.get(3)
        _record(recorder, metrics)

        metrics.media_preprocess_cache_evictions.assert_called_once_with(
            1, "image"
        )
        metrics.media_preprocess_cache_size.assert_called_once_with(20, "image")
        metrics.media_preprocess_cache_capacity.assert_called_once_with(
            20, "image"
        )
        metrics.vision_preprocess_cache_hits.assert_called_once_with(1)
