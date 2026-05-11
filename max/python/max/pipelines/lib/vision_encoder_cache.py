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

"""Reference-counted LRU cache for vision encoder outputs.

Stores per-image encoder embeddings so the vision encoder runs once per
unique image, regardless of how many chunks or requests reference it.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.interfaces.pipeline_variants.text_generation import (
    VLMContextType,
    VLMTextGenerationContext,
)
from max.interfaces.request import RequestID
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices
from max.profiler import traced


def concat_device_buffers(bufs: list[Buffer]) -> Buffer:
    """Concatenate 2D Buffers along dim 0 on device.

    Each buffer must have shape ``[n_rows_i, hidden]`` on the same device
    and with the same dtype. Allocates a single output buffer
    ``[sum(n_rows_i), hidden]`` and copies each input slice into it via
    ``inplace_copy_from``.

    Used both internally by the vision encoder cache (per-image splits
    re-assembled into a batch-shaped output) and by VLM model code that
    runs the vision encoder in multiple chunks and needs to concat the
    per-chunk outputs back into a single per-device tensor before handing
    off to ``prepare_vision_outputs``.
    """
    assert len(bufs) > 0, "concat_device_buffers requires at least one buffer"
    first = bufs[0]
    hidden = int(first.shape[1])
    dtype = first.dtype
    device = first.device
    for b in bufs[1:]:
        assert b.dtype == dtype, (
            f"concat_device_buffers: dtype mismatch ({b.dtype} vs {dtype})"
        )
        assert b.device == device, (
            f"concat_device_buffers: device mismatch ({b.device} vs {device})"
        )
        assert int(b.shape[1]) == hidden, (
            f"concat_device_buffers: dim-1 mismatch "
            f"({int(b.shape[1])} vs {hidden})"
        )
    total_rows = sum(int(b.shape[0]) for b in bufs)
    out = Buffer(
        shape=[total_rows, hidden],
        dtype=dtype,
        device=device,
    )
    offset = 0
    for b in bufs:
        n = int(b.shape[0])
        out[offset : offset + n, :].inplace_copy_from(b)
        offset += n
    return out


@dataclass
class VisionEncoderCacheEntry:
    """Cached vision encoder output for a single image."""

    embeddings: list[Buffer]
    """Per-device embeddings, each shape [num_tokens, hidden_size]."""

    num_tokens: int
    """Number of merged image tokens this entry covers."""

    ref_count: int = 0
    """Number of active requests referencing this entry."""


class VisionEncoderCache(Generic[VLMContextType]):
    """Reference-counted LRU cache for vision encoder outputs.

    Stores per-image encoder embeddings so the vision encoder runs once
    per unique image, regardless of how many chunks or requests
    reference it.

    Typical usage::

        uncached = self._ve_cache.get_uncached_contexts(context_batch)
        if uncached:
            embeds = self._encode(uncached)   # skip cached images via lookup()
            counts = [... per uncached image ...]
        else:
            embeds, counts = empty_embeddings, []

        embeddings, indices = self._ve_cache.prepare_vision_outputs(
            context_batch, uncached, embeds, counts,
            n_devices=..., empty_embeddings=...,
        )
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._cache: OrderedDict[int, VisionEncoderCacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._request_refs: defaultdict[RequestID, set[int]] = defaultdict(set)

    @traced
    def lookup(self, image_hash: int) -> VisionEncoderCacheEntry | None:
        """Look up a cached entry by image hash, refreshing LRU order."""
        entry = self._cache.get(image_hash)
        if entry is not None:
            self._cache.move_to_end(image_hash)
        return entry

    @property
    def enabled(self) -> bool:
        """Whether caching is enabled (max_entries > 0)."""
        return self._max_entries > 0

    @traced
    def insert(
        self,
        image_hash: int,
        embeddings: list[Buffer],
        num_tokens: int,
    ) -> VisionEncoderCacheEntry:
        """Insert a new cache entry. Returns existing entry if already cached.

        When the cache is disabled (``max_entries=0``), creates a
        transient entry without storing it.
        """
        if image_hash in self._cache:
            self._cache.move_to_end(image_hash)
            return self._cache[image_hash]
        entry = VisionEncoderCacheEntry(
            embeddings=embeddings,
            num_tokens=num_tokens,
        )
        if not self.enabled:
            return entry
        while len(self._cache) >= self._max_entries:
            if not self._evict_lru():
                break
        self._cache[image_hash] = entry
        return entry

    @traced
    def acquire(self, request_id: RequestID, image_hash: int) -> None:
        """Increment ref count for a (request, image) pair."""
        refs = self._request_refs[request_id]
        if image_hash in refs:
            return  # already acquired for this request
        entry = self._cache.get(image_hash)
        if entry is not None:
            entry.ref_count += 1
        refs.add(image_hash)

    @traced
    def release_request(self, request_id: RequestID) -> None:
        """Release all cache refs held by a request."""
        for h in self._request_refs.pop(request_id, set()):
            entry = self._cache.get(h)
            if entry is not None:
                entry.ref_count = max(0, entry.ref_count - 1)

    def _evict_lru(self) -> bool:
        """Evict the least-recently-used entry with ref_count == 0."""
        for key in list(self._cache.keys()):
            if self._cache[key].ref_count == 0:
                del self._cache[key]
                return True
        return False

    @staticmethod
    def _ensure_image_hashes(
        ctx: VLMTextGenerationContext,
    ) -> None:
        """Assert that all images have pre-computed hashes.

        The tokenizer must compute image_hash when vision caching is
        enabled.
        """
        for img in ctx.images:
            if img.image_hash is None:
                raise ValueError(
                    "image_hash must be set by the tokenizer when "
                    "vision caching is enabled"
                )

    @traced
    def get_uncached_contexts(
        self,
        context_batch: Sequence[VLMContextType],
    ) -> list[VLMContextType]:
        """Return contexts that have at least one uncached image.

        Contexts where every image is already cached get their refs
        acquired and are excluded.  For partial hits (some cached, some
        not), refs for the cached images are acquired immediately and
        the context is returned.

        Callers can check ``self.lookup(img.image_hash)`` to distinguish
        cached from uncached images within the returned contexts.

        Raises ``ValueError`` if any image is missing its hash.
        """
        uncached_contexts: list[VLMContextType] = []

        for ctx in context_batch:
            if not getattr(ctx, "needs_vision_encoding", False):
                continue

            if not self.enabled:
                uncached_contexts.append(ctx)
                continue

            self._ensure_image_hashes(ctx)

            cached_in_ctx: list[int] = []
            has_uncached = False

            for img in ctx.images:
                assert img.image_hash is not None
                if self.lookup(img.image_hash) is not None:
                    cached_in_ctx.append(img.image_hash)
                else:
                    has_uncached = True

            if not has_uncached:
                for h in cached_in_ctx:
                    self.acquire(ctx.request_id, h)
            else:
                for h in cached_in_ctx:
                    self.acquire(ctx.request_id, h)
                uncached_contexts.append(ctx)

        return uncached_contexts

    @traced
    def _cache_and_split(
        self,
        vision_outputs: list[Buffer],
        per_image_token_counts: list[int],
        image_hashes: list[int],
        request_ids: list[RequestID],
    ) -> None:
        """Split concatenated encoder output per-image and store each in cache.

        Args:
            vision_outputs: Per-device tensors, each [total_tokens, hidden].
            per_image_token_counts: Number of tokens per image.
            image_hashes: Content hash per image.
            request_ids: Request ID per image.
        """
        offset = 0
        for count, img_hash, req_id in zip(
            per_image_token_counts, image_hashes, request_ids, strict=True
        ):
            per_device = [
                dev_tensor[offset : offset + count, :]
                for dev_tensor in vision_outputs
            ]
            offset += count
            if img_hash is not None:
                self.insert(img_hash, per_device, count)
                self.acquire(req_id, img_hash)

    @traced
    def prepare_vision_outputs(
        self,
        context_batch: Sequence[VLMContextType],
        uncached_contexts: Sequence[VLMContextType],
        vision_embeds: list[Buffer],
        per_image_token_counts: list[int],
        n_devices: int,
        empty_embeddings: list[Buffer],
    ) -> tuple[list[Buffer], npt.NDArray[np.int32]]:
        """Store encoder output, assemble embeddings, and compute scatter indices.

        Only images not already in the cache are expected in
        *vision_embeds*.  Images that were already cached (partial hits)
        are skipped automatically.

        Args:
            context_batch: Full batch of contexts (cached + uncached).
            uncached_contexts: Subset from ``get_uncached_contexts``.
            vision_embeds: Per-device encoder output for uncached images.
            per_image_token_counts: Tokens per uncached image, matching
                the concatenation order of *vision_embeds*.
            n_devices: Number of devices.
            empty_embeddings: Empty per-device buffers for text-only batches.

        Returns:
            ``(embeddings, indices)`` — per-device buffers and a 1-D
            int32 scatter-index array.
        """
        if not self.enabled:
            embeddings = (
                vision_embeds if uncached_contexts else empty_embeddings
            )
            indices = compute_multimodal_merge_indices(context_batch)
            return embeddings, indices

        if not per_image_token_counts:
            # All images cached or text-only — assemble from cache,
            # no sync or slicing needed.
            embeddings = self._assemble_embeddings(
                context_batch, n_devices, empty_embeddings
            )
            indices = compute_multimodal_merge_indices(context_batch)
            return embeddings, indices

        # Record an event and synchronize so the vision encoder output
        # is visible before we slice or cache it.
        for buf in vision_embeds:
            if not buf.is_host:
                buf.device.default_stream.record_event().synchronize()

        hashes: list[int] = []
        req_ids: list[RequestID] = []
        all_uncached = True
        for ctx in uncached_contexts:
            for img in ctx.images:
                assert img.image_hash is not None
                if self.lookup(img.image_hash) is None:
                    hashes.append(img.image_hash)
                    req_ids.append(ctx.request_id)
                else:
                    all_uncached = False

        self._cache_and_split(
            vision_embeds, per_image_token_counts, hashes, req_ids
        )

        # every vision context was a miss and every image
        # was uncached, so the encoder output is already in order.
        n_vision = sum(
            1
            for ctx in context_batch
            if getattr(ctx, "needs_vision_encoding", False)
        )
        if len(uncached_contexts) == n_vision and all_uncached:
            embeddings = vision_embeds
        else:
            embeddings = self._assemble_embeddings(
                context_batch, n_devices, empty_embeddings
            )

        indices = compute_multimodal_merge_indices(context_batch)
        return embeddings, indices

    @traced
    def _assemble_embeddings(
        self,
        context_batch: Sequence[VLMContextType],
        n_devices: int,
        empty_embeddings: list[Buffer],
    ) -> list[Buffer]:
        """Build final image_embeddings tensor from cache.

        Must be called after _cache_and_split() so all images are cached.
        Concatenates in the same order as image_token_indices:
        all images from ctx[0], then ctx[1], etc.

        Returns:
            Per-device buffers, each [total_image_tokens, hidden_size].
        """
        all_device_bufs: list[list[Buffer]] = [[] for _ in range(n_devices)]

        for ctx in context_batch:
            if not getattr(ctx, "needs_vision_encoding", False):
                continue
            for img in ctx.images:
                assert img.image_hash is not None
                entry = self.lookup(img.image_hash)
                assert entry is not None, (
                    f"Image {img.image_hash} not in cache — "
                    "_cache_and_split must be called first"
                )
                for d in range(n_devices):
                    all_device_bufs[d].append(entry.embeddings[d])

        if not any(len(dl) > 0 for dl in all_device_bufs):
            return empty_embeddings

        # single image return directly, no copy.
        if all(len(dl) == 1 for dl in all_device_bufs):
            return [dl[0] for dl in all_device_bufs]

        # allocate on device and copy slices in.
        return [concat_device_buffers(dl) for dl in all_device_bufs]
