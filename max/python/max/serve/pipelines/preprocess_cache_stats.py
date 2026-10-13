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

"""Publishes a tokenizer's preprocess-cache counters as ``maxserve.media.*``.

The cache lives in ``max.pipelines``, which cannot import the serve telemetry
stack (see ``max/tests/tests/serve/test_pipelines_serve_layering.py``), so it
only accumulates. This is the serve half: it reads a snapshot after
tokenization and turns the cumulative counts into the deltas an OTel counter
takes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

from max.pipelines.modeling.types import (
    PreprocessCacheStatsProbe,
    PreprocessedImageProbe,
    TextGenerationRequestMessage,
    VisionPreprocessCacheStats,
)
from max.serve.telemetry.metrics import METRICS

logger = logging.getLogger("max.serve")

_StatsFn = Callable[[], Mapping[str, VisionPreprocessCacheStats]]

PreprocessedImageMask = Callable[
    [list[bytes], list[TextGenerationRequestMessage]], Sequence[bool]
]


def preprocessed_image_probe(
    tokenizer: object,
) -> PreprocessedImageMask | None:
    """The tokenizer's preprocessed-image probe, if it offers a usable one.

    Opt-in: an architecture with no preprocessed-image cache does not
    implement :class:`PreprocessedImageProbe`, and every image is decoded as
    before. Callability is checked on top of ``isinstance`` because a
    runtime-checkable protocol only proves the attribute exists -- an
    architecture that defined it as a property or a list would otherwise reach
    the call site and fail there, turning every image request for that model
    into a misleading 400 about the request body.

    Lives here rather than in ``openai_routes`` because two call sites depend
    on giving the same answer: the route decodes against it, and
    :class:`PreprocessCacheStatsRecorder` stays quiet about hits when it is
    not ``None``. Two spellings of the gate could disagree, and the
    disagreement mutes both vantages at once.
    """
    if not isinstance(tokenizer, PreprocessedImageProbe):
        return None
    probe = tokenizer.preprocessed_image_mask
    if not callable(probe):
        logger.warning(
            "%s declares preprocessed_image_mask but it is not callable;"
            " decoding every image.",
            type(tokenizer).__name__,
        )
        return None
    return probe


def _stats_fn(tokenizer: object) -> _StatsFn | None:
    """The tokenizer's cache-stats accessor, if it offers a usable one.

    Callability is checked on top of ``isinstance`` for the same reason as
    the preprocessed-image probe in ``openai_routes``: a runtime-checkable
    protocol only proves the attribute exists, and an architecture that
    defined it as a plain value would otherwise fail inside the request path
    for the sake of a metric.
    """
    if not isinstance(tokenizer, PreprocessCacheStatsProbe):
        return None
    stats = tokenizer.preprocess_cache_stats
    if not callable(stats):
        logger.warning(
            "%s declares preprocess_cache_stats but it is not callable;"
            " publishing no preprocess-cache metrics.",
            type(tokenizer).__name__,
        )
        return None
    return stats


def _counter_deltas(
    last: VisionPreprocessCacheStats | None, now: VisionPreprocessCacheStats
) -> tuple[int, int, int]:
    """Hits, misses and evictions accrued since ``last``.

    A count that went backwards means the cache restarted at zero rather
    than that traffic was negative: ``VisionPreprocessCache.__setstate__``
    rebuilds it empty in any process it is unpickled into. The whole of the
    new value is then the unreported delta, which is also the right answer
    the first time a cache is seen.
    """
    if (
        last is None
        or now.hits < last.hits
        or now.misses < last.misses
        or now.evictions < last.evictions
    ):
        return now.hits, now.misses, now.evictions
    return (
        now.hits - last.hits,
        now.misses - last.misses,
        now.evictions - last.evictions,
    )


class PreprocessCacheStatsRecorder:
    """Records one tokenizer's preprocess-cache counters, per media request.

    Holds the previous snapshot per media kind, because the cache counts
    cumulatively and an OTel counter takes a delta. One instance per
    pipeline, so the snapshot it differences against is always from the same
    process as the cache it read.
    """

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer_name = type(tokenizer).__name__
        self._stats = _stats_fn(tokenizer)
        # An architecture whose probe the route can use already has
        # maxserve.vision.preprocess_cache_hits/_misses recorded for it, from
        # the admission peek in openai_routes. The peek and the tokenizer's
        # own lookup see the same image, so publishing both would count one
        # cached image twice. Every other architecture leaves that pair
        # empty, which is what this fills. The same predicate the route
        # gates on, so a probe it rejects does not silence both vantages.
        self._surface_hits = preprocessed_image_probe(tokenizer) is None
        self._last: dict[str, VisionPreprocessCacheStats] = {}

    def record(self, *, carried_media: bool) -> None:
        """Publishes the caches' counters, unless this request had no media.

        Args:
            carried_media: Whether the request carried any image or video.
                Tokenization runs for every request, and a text-only one must
                stay silent across the whole ``maxserve.media.*`` family.
        """
        if self._stats is None or not carried_media:
            return
        try:
            self._publish(self._stats())
        except Exception:
            # Tokenizer-supplied code, called from inside next_token_chunk's
            # try block whose only handler re-raises, for a read nothing
            # downstream needs. A protocol check proves the attribute exists
            # and the callability check proves it can be called; neither says
            # the call returns a mapping of snapshots. Disabling the recorder
            # costs the metric and nothing else, where raising would 500
            # every media request for the life of the process. Logged once,
            # because self._stats is what gets us here again.
            logger.warning(
                "%s.preprocess_cache_stats() failed; publishing no"
                " preprocess-cache metrics.",
                self._tokenizer_name,
                exc_info=True,
            )
            self._stats = None

    def _publish(
        self, snapshot: Mapping[str, VisionPreprocessCacheStats]
    ) -> None:
        for media_kind, now in snapshot.items():
            hits, misses, evictions = _counter_deltas(
                self._last.get(media_kind), now
            )
            self._last[media_kind] = now
            METRICS.media_preprocess_cache_evictions(evictions, media_kind)
            METRICS.media_preprocess_cache_size(now.size_bytes, media_kind)
            METRICS.media_preprocess_cache_capacity(
                now.capacity_bytes, media_kind
            )
            # The existing pair counts images and carries no media_kind, so
            # only the image cache feeds it; a video hit would otherwise be
            # indistinguishable from an image one.
            if self._surface_hits and media_kind == "image":
                METRICS.vision_preprocess_cache_hits(hits)
                METRICS.vision_preprocess_cache_misses(misses)
