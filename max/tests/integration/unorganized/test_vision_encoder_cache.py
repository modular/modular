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

"""Unit tests for VisionEncoderCache."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.interfaces.context import GenerationStatus, SamplingParams
from max.interfaces.eos_tracking import EOSTracker
from max.interfaces.pipeline_variants.text_generation import (
    ImageMetadata,
    LogProbabilities,
    SpecDecodingState,
    TextGenerationOutput,
)
from max.interfaces.request import RequestID
from max.pipelines.core.context import TokenBuffer
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices


def _make_buffer(rows: int, cols: int = 4) -> Buffer:
    """Create a host Buffer with deterministic data."""
    arr = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    return Buffer.from_numpy(arr)


def _make_image_meta(
    start: int, end: int, image_hash: int | None = None
) -> ImageMetadata:
    """Create an ImageMetadata with minimal pixel data."""
    return ImageMetadata(
        start_idx=start,
        end_idx=end,
        pixel_values=np.zeros((1, 3), dtype=np.float32),
        image_hash=image_hash,
    )


def _make_token_buffer(
    total_length: int, processed_length: int = 0
) -> TokenBuffer:
    """Create a TokenBuffer of *total_length* tokens with *processed_length* already processed.

    The active window covers ``[processed_length, total_length)``.
    """
    buf = TokenBuffer(np.zeros(total_length, dtype=np.int64))
    if processed_length > 0:
        # skip_processing advances the active-window start, which sets processed_length.
        buf.skip_processing(processed_length)
    return buf


class FakeContext:
    """Test context implementing VLMTextGenerationContext protocol."""

    def __init__(
        self,
        request_id: RequestID,
        images: list[ImageMetadata] | None = None,
        needs_vision: bool = True,
        image_token_indices: npt.NDArray[np.int32] | None = None,
        processed_length: int = 0,
        active_length: int = 0,
    ) -> None:
        self._eos_tracker = EOSTracker()
        self._request_id = request_id
        self.images: list[ImageMetadata] = images or []
        self._needs_vision = needs_vision
        self.status = GenerationStatus.ACTIVE
        self.image_token_indices: npt.NDArray[np.int32] = (
            image_token_indices
            if image_token_indices is not None
            else np.empty(0, dtype=np.int32)
        )
        total_length = processed_length + active_length
        self.tokens: TokenBuffer = _make_token_buffer(
            max(total_length, 1), processed_length
        )
        self.cached_prefix_length: int | None = None
        self.in_reasoning_phase: bool = False

    @property
    def request_id(self) -> RequestID:
        return self._request_id

    @property
    def image_idx(self) -> int:
        return 0 if self._needs_vision else len(self.images)

    @property
    def needs_vision_encoding(self) -> bool:
        return self._needs_vision

    @property
    def next_images(self) -> list[ImageMetadata]:
        return self.images if self._needs_vision else []

    def compute_image_aligned_idx(self, idx: int) -> int:
        return idx

    @property
    def eos_tracker(self) -> EOSTracker:
        return self._eos_tracker

    @property
    def max_length(self) -> int | None:
        return None

    def reset(self) -> None:
        pass

    def compute_num_available_steps(self, max_seq_len: int) -> int:
        return 0

    @property
    def min_tokens(self) -> int:
        return 0

    @property
    def log_probabilities(self) -> int:
        return 0

    @property
    def log_probabilities_echo(self) -> bool:
        return False

    def get_min_token_logit_mask(
        self, num_steps: int
    ) -> list[npt.NDArray[np.int32]]:
        return []

    def advance_token_buffer(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        pass

    def advance_fsm(self, token: int) -> bool:
        return False

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        pass

    def update_with_future_token(self) -> None:
        pass

    def realize_future_token(
        self, new_token: int, log_probabilities: LogProbabilities | None = None
    ) -> None:
        pass

    @property
    def matcher(self) -> Any | None:
        return None

    @property
    def json_schema(self) -> str | None:
        return None

    @property
    def grammar(self) -> str | None:
        return None

    def set_matcher(self, matcher: Any) -> None:
        pass

    @property
    def sampling_params(self) -> SamplingParams:
        return SamplingParams()

    @property
    def is_initial_prompt(self) -> bool:
        return True

    def to_generation_output(self) -> TextGenerationOutput:
        raise NotImplementedError

    @property
    def spec_decoding_state(self) -> SpecDecodingState:
        return SpecDecodingState()

    @property
    def is_done(self) -> bool:
        return self.status.is_done


def _ref_count(cache: VisionEncoderCache[FakeContext], image_hash: int) -> int:
    """Helper to get ref_count, asserting entry exists."""
    entry = cache.lookup(image_hash)
    assert entry is not None, f"Expected cache entry for {image_hash:#x}"
    return entry.ref_count


def _make_cache() -> VisionEncoderCache[FakeContext]:
    """Create a cache for testing."""
    return VisionEncoderCache()


def _make_cache_sized(max_entries: int) -> VisionEncoderCache[FakeContext]:
    """Create a size-bounded cache for testing."""
    return VisionEncoderCache(max_entries=max_entries)


def test_insert_and_lookup() -> None:
    cache = _make_cache()
    buf = _make_buffer(10)
    cache.insert(0xABC, [buf], 10)
    entry = cache.lookup(0xABC)
    assert entry is not None
    assert entry.num_tokens == 10
    assert len(entry.embeddings) == 1


def test_lookup_miss() -> None:
    cache = _make_cache()
    assert cache.lookup(0xDEAD) is None


def test_insert_idempotent() -> None:
    cache = _make_cache()
    buf1 = _make_buffer(10)
    buf2 = _make_buffer(20)
    entry1 = cache.insert(0xABC, [buf1], 10)
    entry2 = cache.insert(0xABC, [buf2], 20)
    assert entry1 is entry2
    assert entry1.num_tokens == 10


def test_lookup_refreshes_lru() -> None:
    cache = _make_cache_sized(2)
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.insert(0x2, [_make_buffer(1)], 1)
    cache.lookup(0x1)  # refresh
    cache.insert(0x3, [_make_buffer(1)], 1)
    assert cache.lookup(0x1) is not None
    assert cache.lookup(0x2) is None  # evicted
    assert cache.lookup(0x3) is not None


def test_evicts_oldest_unreferenced() -> None:
    cache = _make_cache_sized(2)
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.insert(0x2, [_make_buffer(1)], 1)
    cache.insert(0x3, [_make_buffer(1)], 1)
    assert cache.lookup(0x1) is None
    assert cache.lookup(0x2) is not None


def test_eviction_skips_referenced_entries() -> None:
    cache = _make_cache_sized(2)
    req = RequestID("r1")
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.acquire(req, 0x1)
    cache.insert(0x2, [_make_buffer(1)], 1)
    cache.insert(0x3, [_make_buffer(1)], 1)
    assert cache.lookup(0x1) is not None  # protected
    assert cache.lookup(0x2) is None  # evicted
    assert cache.lookup(0x3) is not None


def test_no_eviction_when_all_referenced() -> None:
    cache = _make_cache_sized(2)
    req = RequestID("r1")
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.acquire(req, 0x1)
    cache.insert(0x2, [_make_buffer(1)], 1)
    cache.acquire(req, 0x2)
    cache.insert(0x3, [_make_buffer(1)], 1)
    assert cache.lookup(0x1) is not None
    assert cache.lookup(0x2) is not None
    assert cache.lookup(0x3) is not None


def test_acquire_increments_ref() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.acquire(RequestID("r1"), 0x1)
    assert _ref_count(cache, 0x1) == 1


def test_acquire_idempotent_per_request() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    req = RequestID("r1")
    cache.acquire(req, 0x1)
    cache.acquire(req, 0x1)
    assert _ref_count(cache, 0x1) == 1


def test_multiple_requests_increment_separately() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.acquire(RequestID("r1"), 0x1)
    cache.acquire(RequestID("r2"), 0x1)
    assert _ref_count(cache, 0x1) == 2


def test_release_decrements_ref() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    req = RequestID("r1")
    cache.acquire(req, 0x1)
    cache.release_request(req)
    assert _ref_count(cache, 0x1) == 0


def test_release_unknown_request_is_noop() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    cache.release_request(RequestID("unknown"))
    assert _ref_count(cache, 0x1) == 0


def test_release_does_not_go_negative() -> None:
    cache = _make_cache()
    cache.insert(0x1, [_make_buffer(1)], 1)
    req = RequestID("r1")
    cache.acquire(req, 0x1)
    cache.release_request(req)
    cache.release_request(req)  # double release
    assert _ref_count(cache, 0x1) == 0


def test_get_uncached_all_cached() -> None:
    cache = _make_cache()
    cache.insert(0xA, [_make_buffer(5)], 5)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=0xA)],
    )
    misses = cache.get_uncached_contexts([ctx])
    assert len(misses) == 0
    assert _ref_count(cache, 0xA) == 1


def test_get_uncached_returns_miss() -> None:
    cache = _make_cache()
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=0xB)],
    )
    misses = cache.get_uncached_contexts([ctx])
    assert len(misses) == 1
    assert misses[0] is ctx


def test_get_uncached_partial_miss() -> None:
    """If one image is cached but another isn't, context is a miss.

    The cached image should have its ref acquired immediately, and
    only the uncached hash should appear in the returned set.
    """
    cache = _make_cache()
    cache.insert(0xA, [_make_buffer(5)], 5)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 5, image_hash=0xA),
            _make_image_meta(5, 10, image_hash=0xB),
        ],
    )
    misses = cache.get_uncached_contexts([ctx])
    assert len(misses) == 1
    # The cached image should already have its ref acquired.
    assert _ref_count(cache, 0xA) == 1


def test_prepare_partial_hit_only_encodes_uncached() -> None:
    """With a partial hit, only the uncached image is encoded and stored."""
    cache = _make_cache()
    hidden = 4

    # Pre-cache image A.
    buf_a = Buffer.from_numpy(np.ones((2, hidden), dtype=np.float32) * 1.0)
    cache.insert(0xA, [buf_a], 2)

    # Context has cached image A and uncached image B.
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 2, image_hash=0xA),
            _make_image_meta(2, 5, image_hash=0xB),
        ],
        image_token_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        processed_length=0,
        active_length=8,
    )
    uncached = cache.get_uncached_contexts([ctx])
    assert len(uncached) == 1

    # Simulate encoding ONLY image B (3 tokens).
    vision_embeds = [
        Buffer.from_numpy(np.ones((3, hidden), dtype=np.float32) * 2.0)
    ]
    result, _indices = cache.prepare_vision_outputs(
        context_batch=[ctx],
        uncached_contexts=uncached,
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    # Should assemble: image A (2 rows of 1.0) then image B (3 rows of 2.0).
    arr = result[0].to_numpy()
    assert arr.shape == (5, hidden)
    np.testing.assert_allclose(arr[:2], 1.0)
    np.testing.assert_allclose(arr[2:], 2.0)

    # Both images should have refs acquired.
    assert _ref_count(cache, 0xA) == 1
    assert _ref_count(cache, 0xB) == 1


def test_prepare_partial_hit_multi_context() -> None:
    """Partial hit in one context, full miss in another."""
    cache = _make_cache()
    hidden = 4

    # Pre-cache image A.
    buf_a = Buffer.from_numpy(np.ones((2, hidden), dtype=np.float32) * 1.0)
    cache.insert(0xA, [buf_a], 2)

    # ctx1: partial hit (A cached, B uncached).
    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 2, image_hash=0xA),
            _make_image_meta(2, 5, image_hash=0xB),
        ],
        image_token_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        processed_length=0,
        active_length=6,
    )
    # ctx2: full miss (C uncached).
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 4, image_hash=0xC)],
        image_token_indices=np.array([0, 1, 2, 3], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    uncached = cache.get_uncached_contexts([ctx1, ctx2])
    assert len(uncached) == 2
    # Image A in ctx1 should already have its ref.
    assert _ref_count(cache, 0xA) == 1

    # Encode only B (3 tokens) and C (4 tokens).
    vision_embeds = [
        Buffer.from_numpy(np.ones((7, hidden), dtype=np.float32) * 2.0)
    ]
    result, _indices = cache.prepare_vision_outputs(
        context_batch=[ctx1, ctx2],
        uncached_contexts=uncached,
        vision_embeds=vision_embeds,
        per_image_token_counts=[3, 4],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    # Assembled: A(2) + B(3) + C(4) = 9 rows.
    arr = result[0].to_numpy()
    assert arr.shape == (9, hidden)
    np.testing.assert_allclose(arr[:2], 1.0)  # image A from cache
    np.testing.assert_allclose(arr[2:], 2.0)  # images B, C from encoder


def test_get_uncached_skips_non_vision() -> None:
    cache = _make_cache()
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[],
        needs_vision=False,
    )
    misses = cache.get_uncached_contexts([ctx])
    assert len(misses) == 0


def test_none_hash_raises() -> None:
    """Missing image_hash raises ValueError when cache is enabled."""
    cache = _make_cache()
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=None)],
    )
    with pytest.raises(ValueError):
        cache.get_uncached_contexts([ctx])


def test_none_hash_allowed_when_disabled() -> None:
    """Missing image_hash is fine when cache is disabled."""
    cache = _make_cache_sized(0)
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 5, image_hash=None)],
    )
    misses = cache.get_uncached_contexts([ctx])
    assert len(misses) == 1


def test__cache_and_split_stores_per_image() -> None:
    cache = _make_cache()
    hidden = 4
    total = _make_buffer(8, hidden)
    cache._cache_and_split(
        vision_outputs=[total],
        per_image_token_counts=[3, 5],
        image_hashes=[0xA, 0xB],
        request_ids=[RequestID("r1"), RequestID("r1")],
    )
    entry_a = cache.lookup(0xA)
    entry_b = cache.lookup(0xB)
    assert entry_a is not None and entry_a.num_tokens == 3
    assert entry_b is not None and entry_b.num_tokens == 5
    assert entry_a.embeddings[0].to_numpy().shape == (3, hidden)
    assert entry_b.embeddings[0].to_numpy().shape == (5, hidden)
    np.testing.assert_array_equal(
        entry_a.embeddings[0].to_numpy(), total.to_numpy()[:3]
    )
    np.testing.assert_array_equal(
        entry_b.embeddings[0].to_numpy(), total.to_numpy()[3:8]
    )


def test__cache_and_split_acquires_refs() -> None:
    cache = _make_cache()
    total = _make_buffer(5, 4)
    cache._cache_and_split(
        vision_outputs=[total],
        per_image_token_counts=[5],
        image_hashes=[0xA],
        request_ids=[RequestID("r1")],
    )
    assert _ref_count(cache, 0xA) == 1


def test__cache_and_split_none_hash_not_cached() -> None:
    cache = _make_cache()
    total = _make_buffer(5, 4)
    cache._cache_and_split(
        vision_outputs=[total],
        per_image_token_counts=[5],
        image_hashes=[None],  # type: ignore[list-item]
        request_ids=[RequestID("r1")],
    )
    assert len(cache._cache) == 0


def test_assemble_concatenates_in_order() -> None:
    cache = _make_cache()
    hidden = 4
    buf_a = _make_buffer(3, hidden)
    buf_b = _make_buffer(5, hidden)
    cache.insert(0xA, [buf_a], 3)
    cache.insert(0xB, [buf_b], 5)

    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[
            _make_image_meta(0, 3, image_hash=0xA),
            _make_image_meta(3, 8, image_hash=0xB),
        ],
    )
    empty = [_make_buffer(0, hidden)]
    result = cache._assemble_embeddings(
        [ctx], n_devices=1, empty_embeddings=empty
    )
    arr = result[0].to_numpy()
    assert arr.shape == (8, hidden)
    np.testing.assert_array_equal(arr[:3], buf_a.to_numpy())
    np.testing.assert_array_equal(arr[3:], buf_b.to_numpy())


def test_assemble_returns_empty_when_no_vision() -> None:
    cache = _make_cache()
    ctx = FakeContext(request_id=RequestID("r1"), images=[], needs_vision=False)
    empty = [_make_buffer(0, 4)]
    result = cache._assemble_embeddings(
        [ctx], n_devices=1, empty_embeddings=empty
    )
    assert result is empty


def test_assemble_multi_context_ordering() -> None:
    """Embeddings concatenated in context order."""
    cache = _make_cache()
    hidden = 4
    buf_a = Buffer.from_numpy(np.ones((2, hidden), dtype=np.float32) * 1.0)
    buf_b = Buffer.from_numpy(np.ones((3, hidden), dtype=np.float32) * 2.0)
    cache.insert(0xA, [buf_a], 2)
    cache.insert(0xB, [buf_b], 3)

    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
    )
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 3, image_hash=0xB)],
    )
    empty = [_make_buffer(0, hidden)]
    result = cache._assemble_embeddings(
        [ctx1, ctx2], n_devices=1, empty_embeddings=empty
    )
    arr = result[0].to_numpy()
    assert arr.shape == (5, hidden)
    np.testing.assert_allclose(arr[:2], 1.0)
    np.testing.assert_allclose(arr[2:], 2.0)


def test_cross_request_dedup() -> None:
    """Two requests with the same image share one cache entry."""
    cache = _make_cache()
    cache.insert(0xABC, [_make_buffer(5)], 5)
    r1, r2 = RequestID("r1"), RequestID("r2")
    ctx1 = FakeContext(
        request_id=r1,
        images=[_make_image_meta(0, 5, image_hash=0xABC)],
    )
    ctx2 = FakeContext(
        request_id=r2,
        images=[_make_image_meta(0, 5, image_hash=0xABC)],
    )
    misses = cache.get_uncached_contexts([ctx1, ctx2])
    assert len(misses) == 0
    assert _ref_count(cache, 0xABC) == 2

    cache.release_request(r1)
    assert _ref_count(cache, 0xABC) == 1
    cache.release_request(r2)
    assert _ref_count(cache, 0xABC) == 0


def test_end_to_end_chunked_prefill() -> None:
    """Simulates the full 2-chunk prefill workflow."""
    cache = _make_cache()
    hidden = 4
    req = RequestID("request-1")

    # Chunk 1: cache miss → encode → store
    ctx = FakeContext(
        request_id=req,
        images=[_make_image_meta(100, 400, image_hash=0xABC)],
    )
    misses = cache.get_uncached_contexts([ctx])
    assert len(misses) == 1

    vision_output = _make_buffer(300, hidden)
    cache._cache_and_split(
        vision_outputs=[vision_output],
        per_image_token_counts=[300],
        image_hashes=[0xABC],
        request_ids=[req],
    )

    empty = [_make_buffer(0, hidden)]
    embeds1 = cache._assemble_embeddings(
        [ctx], n_devices=1, empty_embeddings=empty
    )
    assert embeds1[0].to_numpy().shape == (300, hidden)

    # Chunk 2: cache hit → no encoding
    misses = cache.get_uncached_contexts([ctx])
    assert len(misses) == 0

    embeds2 = cache._assemble_embeddings(
        [ctx], n_devices=1, empty_embeddings=empty
    )
    np.testing.assert_array_equal(embeds2[0].to_numpy(), embeds1[0].to_numpy())

    cache.release_request(req)
    assert _ref_count(cache, 0xABC) == 0
    assert cache.lookup(0xABC) is not None


def test_prepare_all_uncached_fast_path() -> None:
    """When every vision context is uncached, returns encoder output directly."""
    cache = _make_cache()
    hidden = 4
    req = RequestID("r1")
    ctx = FakeContext(
        request_id=req,
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    uncached = cache.get_uncached_contexts([ctx])
    assert len(uncached) == 1

    vision_embeds = [_make_buffer(3, hidden)]
    result, indices = cache.prepare_vision_outputs(
        context_batch=[ctx],
        uncached_contexts=uncached,
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    assert result is vision_embeds
    assert _ref_count(cache, 0xA) == 1
    np.testing.assert_array_equal(indices, [0, 1, 2])


def test_prepare_mixed_hits() -> None:
    """When some contexts are cached and others are not, assembles from cache."""
    cache = _make_cache()
    hidden = 4

    buf_a = Buffer.from_numpy(np.ones((2, hidden), dtype=np.float32) * 1.0)
    cache.insert(0xA, [buf_a], 2)

    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
        image_token_indices=np.array([0, 1], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 3, image_hash=0xB)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    uncached = cache.get_uncached_contexts([ctx1, ctx2])
    # ctx1 is a hit (all images cached), ctx2 is a miss.
    assert len(uncached) == 1
    assert uncached[0].request_id == RequestID("r2")

    vision_embeds = [
        Buffer.from_numpy(np.ones((3, hidden), dtype=np.float32) * 2.0)
    ]
    result, _indices = cache.prepare_vision_outputs(
        context_batch=[ctx1, ctx2],
        uncached_contexts=uncached,
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    # Should assemble: image A (2 rows of 1.0) then image B (3 rows of 2.0).
    arr = result[0].to_numpy()
    assert arr.shape == (5, hidden)
    np.testing.assert_allclose(arr[:2], 1.0)
    np.testing.assert_allclose(arr[2:], 2.0)


def test_prepare_all_cached() -> None:
    """When every context is already cached, returns assembled embeddings."""
    cache = _make_cache()
    hidden = 4
    buf = Buffer.from_numpy(np.ones((4, hidden), dtype=np.float32) * 3.0)
    cache.insert(0xC, [buf], 4)

    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 4, image_hash=0xC)],
        image_token_indices=np.array([0, 1, 2, 3], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    uncached = cache.get_uncached_contexts([ctx])
    assert len(uncached) == 0

    empty = [_make_buffer(0, hidden)]
    result, _indices = cache.prepare_vision_outputs(
        context_batch=[ctx],
        uncached_contexts=uncached,
        vision_embeds=empty,
        per_image_token_counts=[],
        n_devices=1,
        empty_embeddings=empty,
    )
    arr = result[0].to_numpy()
    assert arr.shape == (4, hidden)
    np.testing.assert_allclose(arr, 3.0)


def test_disabled_cache_never_hits() -> None:
    """With max_entries=0, every vision context is always uncached."""
    cache = _make_cache_sized(0)
    assert not cache.enabled

    hidden = 4
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )

    # First call: miss as expected.
    uncached = cache.get_uncached_contexts([ctx])
    assert len(uncached) == 1

    # Simulate encoding and caching.
    vision_embeds = [_make_buffer(3, hidden)]
    cache.prepare_vision_outputs(
        context_batch=[ctx],
        uncached_contexts=uncached,
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )

    # Second call with same image: still a miss — nothing was stored.
    uncached2 = cache.get_uncached_contexts([ctx])
    assert len(uncached2) == 1
    assert cache.lookup(0xA) is None


def test_disabled_cache_insert_returns_transient_entry() -> None:
    """Insert on a disabled cache returns a valid entry but doesn't store it."""
    cache = _make_cache_sized(0)
    buf = _make_buffer(5, 4)
    entry = cache.insert(0xBEEF, [buf], 5)
    assert entry.num_tokens == 5
    assert entry.embeddings == [buf]
    # Not stored in the cache.
    assert cache.lookup(0xBEEF) is None


def test_merge_indices_single_context_no_offset() -> None:
    """Indices start from 0 when processed_length is 0."""
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(2, 5, image_hash=0xA)],
        image_token_indices=np.array([2, 3, 4], dtype=np.int32),
        processed_length=0,
        active_length=10,
    )
    indices = compute_multimodal_merge_indices([ctx])
    np.testing.assert_array_equal(indices, [2, 3, 4])


def test_merge_indices_accounts_for_processed_length() -> None:
    """Indices in already-processed tokens become OOB sentinels."""
    # Prompt: [0..9, IMG, IMG, IMG, IMG, 14..19]  (20 tokens total)
    # Image at positions 10-13.  processed_length=12 means tokens 0-11 done.
    # Active window is tokens 12-19 (active_length=8).
    # Indices 10 and 11 are in processed region → OOB.
    # Indices 12 → offset 0, 13 → offset 1 within the active window.
    oob = np.iinfo(np.int32).min
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(10, 14, image_hash=0xA)],
        image_token_indices=np.array([10, 11, 12, 13], dtype=np.int32),
        processed_length=12,
        active_length=8,
    )
    indices = compute_multimodal_merge_indices([ctx])
    np.testing.assert_array_equal(indices, [oob, oob, 0, 1])


def test_merge_indices_batch_offsets() -> None:
    """Active-token offsets accumulate across contexts in a batch."""
    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
        image_token_indices=np.array([0, 1], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 3, image_hash=0xB)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=7,
    )
    indices = compute_multimodal_merge_indices([ctx1, ctx2])
    # ctx1 indices: [0, 1]  (offset 0)
    # ctx2 indices: [0, 1, 2] + 5 (ctx1.active_length) = [5, 6, 7]
    np.testing.assert_array_equal(indices, [0, 1, 5, 6, 7])


def test_merge_indices_skips_non_vision_contexts() -> None:
    """Non-vision contexts contribute to offset but produce no indices."""
    ctx_text = FakeContext(
        request_id=RequestID("r1"),
        needs_vision=False,
        active_length=10,
    )
    ctx_vision = FakeContext(
        request_id=RequestID("r2"),
        images=[_make_image_meta(0, 2, image_hash=0xA)],
        image_token_indices=np.array([0, 1], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    indices = compute_multimodal_merge_indices([ctx_text, ctx_vision])
    # text context contributes 10 tokens of offset.
    np.testing.assert_array_equal(indices, [10, 11])


def test_merge_indices_empty_batch() -> None:
    """Empty batch returns empty array."""
    indices = compute_multimodal_merge_indices([])
    assert indices.shape == (0,)
    assert indices.dtype == np.int32


def test_merge_indices_beyond_active_are_oob() -> None:
    """Indices beyond active_length must be OOB, not passed through."""
    oob = np.iinfo(np.int32).min
    # Image spans positions 2-7, but active window is only 0-4 (active_length=5).
    # Indices 5, 6, 7 are beyond active and must be OOB.
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(2, 8, image_hash=0xA)],
        image_token_indices=np.array([2, 3, 4, 5, 6, 7], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    indices = compute_multimodal_merge_indices([ctx])
    np.testing.assert_array_equal(indices, [2, 3, 4, oob, oob, oob])


def test_merge_indices_beyond_active_no_cross_contamination() -> None:
    """Beyond-active indices must not land in another request's token range."""
    oob = np.iinfo(np.int32).min
    # ctx1: vision request with image spanning beyond its active window.
    ctx1 = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(2, 8, image_hash=0xA)],
        image_token_indices=np.array([2, 3, 4, 5, 6, 7], dtype=np.int32),
        processed_length=0,
        active_length=5,
    )
    # ctx2: text-only decode request contributing 3 tokens.
    ctx2 = FakeContext(
        request_id=RequestID("r2"),
        needs_vision=False,
        active_length=3,
    )
    indices = compute_multimodal_merge_indices([ctx1, ctx2])
    # Only indices 2,3,4 are valid. 5,6,7 must be OOB — NOT 5+0=5, 6+0=6, 7+0=7
    # which would land in ctx2's token range [5, 8).
    np.testing.assert_array_equal(indices, [2, 3, 4, oob, oob, oob])
    # Verify no index falls in ctx2's range.
    valid = indices[indices != oob]
    assert all(v < 5 for v in valid)


def test_prepare_vision_outputs_returns_embeddings_and_indices() -> None:
    """prepare_vision_outputs returns both embeddings and scatter indices."""
    cache = _make_cache()
    hidden = 4
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(0, 3, image_hash=0xA)],
        image_token_indices=np.array([0, 1, 2], dtype=np.int32),
        processed_length=0,
        active_length=10,
    )
    uncached = cache.get_uncached_contexts([ctx])
    assert len(uncached) == 1

    vision_embeds = [_make_buffer(3, hidden)]
    embeddings, indices = cache.prepare_vision_outputs(
        context_batch=[ctx],
        uncached_contexts=uncached,
        vision_embeds=vision_embeds,
        per_image_token_counts=[3],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    assert embeddings is vision_embeds
    np.testing.assert_array_equal(indices, [0, 1, 2])


def test_prepare_vision_outputs_chunked_prefill() -> None:
    """Indices from prior chunks become OOB in prepare_vision_outputs."""
    cache = _make_cache()
    hidden = 4
    oob = np.iinfo(np.int32).min

    # 6 image tokens at positions 4-9, processed_length=6 (first 6 done),
    # active window is positions 6-15 (active_length=10).
    ctx = FakeContext(
        request_id=RequestID("r1"),
        images=[_make_image_meta(4, 10, image_hash=0xA)],
        image_token_indices=np.array([4, 5, 6, 7, 8, 9], dtype=np.int32),
        processed_length=6,
        active_length=10,
    )
    uncached = cache.get_uncached_contexts([ctx])
    vision_embeds = [_make_buffer(6, hidden)]
    _embeddings, indices = cache.prepare_vision_outputs(
        context_batch=[ctx],
        uncached_contexts=uncached,
        vision_embeds=vision_embeds,
        per_image_token_counts=[6],
        n_devices=1,
        empty_embeddings=[_make_buffer(0, hidden)],
    )
    # positions 4,5 < processed_length=6 → OOB
    # positions 6,7,8,9 → offsets 0,1,2,3
    np.testing.assert_array_equal(indices, [oob, oob, 0, 1, 2, 3])
