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
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    VLMContextType,
)
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices
from max.pipelines.request import RequestID
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


def _owned_row_slice(src: Buffer, start: int, count: int) -> Buffer:
    """Copy ``src[start:start + count, :]`` into a freshly allocated Buffer.

    An owned copy (not a view) so a cache entry does not pin the variable-size
    vision-encoder output buffer, avoiding GPU allocator fragmentation.
    """
    slot = Buffer.zeros(
        shape=[count, int(src.shape[1])],
        dtype=src.dtype,
        device=src.device,
    )
    slot.inplace_copy_from(src[start : start + count, :])
    return slot


@dataclass
class VisionEncodeResult:
    """A model's vision-encoder output for the uncached images of a batch."""

    embeddings: list[Buffer]
    """Per-device encoder output, each ``[total_tokens, hidden]``.

    Rows are ordered context-major, image-minor.
    """

    per_image_token_counts: list[int] | None = None
    """Explicit per-image token counts, in row order.

    ``None`` lets the driver derive them from placeholder spans. A model whose
    span differs from its emitted row count must set this.
    """


PackedVisionInputsT = TypeVar("PackedVisionInputsT")


@runtime_checkable
class SupportsVisionEncoding(Protocol[PackedVisionInputsT]):
    """A pipeline model that encodes images in two declared steps.

    Caching is the driver's job: the model packs and encodes, the cache
    stores and assembles. ``PackedVisionInputsT`` is the model's packed-input
    type, carried from prep to encode.
    """

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> PackedVisionInputsT | None:
        """Pack the batch's selected cache-miss pixels to device, during prep.

        Optional: returning ``None`` defers packing to :meth:`vision_execute`.
        Runs in the prep phase so the host-to-device copy overlaps the prior
        batch. ``selection`` is the ``(context, miss-images)`` pairs from
        :meth:`VisionEncoderCache.select`.
        """
        ...

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: PackedVisionInputsT | None,
    ) -> VisionEncodeResult:
        """Run the vision encoder over the batch's selected cache-miss images.

        Uses ``packed`` when :meth:`pack_vision_inputs` packed, otherwise packs
        from ``selection`` inline. Returns per-device embeddings, context-major
        then image-minor; caching is the cache's job.
        """
        ...

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row embedding buffers for text-only/cached batches.

        The cache assembles from these when no image is encoded this step;
        the model owns the hidden size and dtype.
        """
        ...


def derive_counts_from_spans(
    selection: Sequence[tuple[VLMContextType, Sequence[ImageMetadata]]],
) -> list[int]:
    """Per-image token counts from the cache's selection, in row order.

    Walks the ``(context, miss-images)`` pairs from
    :meth:`VisionEncoderCache.select` — so the counts equal the encoder's
    emitted rows by construction, in every mode (including a disabled cache
    under chunked prefill, where ``ctx.images`` would include already-processed
    images the encoder did not emit rows for).

    Args:
        selection: The ``(context, miss-images)`` pairs to encode this step.

    Returns:
        One token count per encoded image, in row order.
    """
    return [
        img.end_idx - img.start_idx
        for _ctx, miss_images in selection
        for img in miss_images
    ]


def validate_vision_encode_counts(
    per_image_token_counts: Sequence[int],
    embeddings: Sequence[Buffer],
) -> None:
    """Raise if the per-image counts don't sum to the encoder's row count.

    Guards the per-image cache split against a model whose placeholder span
    doesn't match its emitted rows.

    Args:
        per_image_token_counts: Per-image token counts, in row order.
        embeddings: Per-device encoder output; row count is from device 0.

    Raises:
        ValueError: On a count/row mismatch.
    """
    if not embeddings:
        return
    total_rows = int(embeddings[0].shape[0])
    total_counts = int(sum(per_image_token_counts))
    if total_counts != total_rows:
        raise ValueError(
            f"Vision encoder emitted {total_rows} row(s) but per-image token "
            f"counts sum to {total_counts}. The encoder must emit exactly one "
            "row per placeholder token; a model whose placeholder span "
            "includes non-scatter-target tokens must return explicit "
            "per_image_token_counts."
        )


@dataclass
class VisionEncoderCacheEntry:
    """Cached vision encoder output for a single image."""

    embeddings: list[Buffer]
    """Per-device embeddings, each shape [num_tokens, hidden_size]."""

    num_tokens: int
    """Number of merged image tokens this entry covers."""

    ref_count: int = 0
    """Number of active requests referencing this entry."""


@dataclass
class VisionEncoderMetrics:
    """Per-iteration vision encoder statistics for one batch.

    Populated by :class:`VisionEncoderCache` during batch preparation and
    surfaced by the scheduler in its per-iteration log so that vision
    encoder cost is attributed separately from the language model forward
    pass.
    """

    num_images_total: int = 0
    """Images referenced by vision requests in this batch (hits + misses)."""

    num_images_encoded: int = 0
    """Images the vision encoder actually ran on this batch (cache misses)."""

    num_images_cached: int = 0
    """Images served from the vision encoder cache this batch (cache hits)."""

    num_patches_encoded: int = 0
    """Input image patches fed to the vision encoder this batch."""

    num_tokens_encoded: int = 0
    """Merged vision tokens produced by the encoder this batch."""

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of images served from cache (0.0 when no images)."""
        if self.num_images_total == 0:
            return 0.0
        return self.num_images_cached / self.num_images_total


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

    def __init__(self, max_entries: int = 256, n_devices: int = 1) -> None:
        self._cache: OrderedDict[int, VisionEncoderCacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._n_devices = n_devices
        self._request_refs: defaultdict[RequestID, set[int]] = defaultdict(set)

        # Per-batch vision encoder metrics, populated during batch
        # preparation and drained by the scheduler once per iteration via
        # ``pop_metrics``. ``None`` when the most recent batch did no vision
        # work (e.g. a text-only or decode step).
        self._batch_metrics: VisionEncoderMetrics | None = None

    @traced
    def lookup(self, image_hash: int) -> VisionEncoderCacheEntry | None:
        """Look up a cached entry by image hash, refreshing LRU order.

        A falsy hash (``0``, the sentinel for an image/video with no
        content hash) is treated as a miss, so callers don't need to guard
        the call themselves.
        """
        if not image_hash:
            return None
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
        ctx: TextAndVisionContext,
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

        metrics = VisionEncoderMetrics()

        for ctx in context_batch:
            if not getattr(ctx, "needs_vision_encoding", False):
                continue

            if not self.enabled:
                uncached_contexts.append(ctx)
                for img in ctx.images:
                    self._record_encoded_image(metrics, img)
                continue

            self._ensure_image_hashes(ctx)

            cached_in_ctx: list[int] = []
            has_uncached = False

            for img in ctx.images:
                assert img.image_hash is not None
                metrics.num_images_total += 1
                if self.lookup(img.image_hash) is not None:
                    cached_in_ctx.append(img.image_hash)
                    metrics.num_images_cached += 1
                else:
                    has_uncached = True
                    self._record_encoded_image(metrics, img, count_total=False)

            if not has_uncached:
                for h in cached_in_ctx:
                    self.acquire(ctx.request_id, h)
            else:
                for h in cached_in_ctx:
                    self.acquire(ctx.request_id, h)
                uncached_contexts.append(ctx)

        self._batch_metrics = metrics if metrics.num_images_total > 0 else None
        return uncached_contexts

    @staticmethod
    def _record_encoded_image(
        metrics: VisionEncoderMetrics,
        img: object,
        count_total: bool = True,
    ) -> None:
        """Tally one cache-miss image (encoder runs on it) into ``metrics``.

        ``count_total`` is False when the caller already incremented
        ``num_images_total`` (the enabled-cache path counts every image up
        front to distinguish hits from misses).
        """
        if count_total:
            metrics.num_images_total += 1
        metrics.num_images_encoded += 1
        pixel_values = getattr(img, "pixel_values", None)
        if pixel_values is not None and getattr(pixel_values, "shape", None):
            metrics.num_patches_encoded += int(pixel_values.shape[0])
        start_idx = getattr(img, "start_idx", None)
        end_idx = getattr(img, "end_idx", None)
        if start_idx is not None and end_idx is not None:
            metrics.num_tokens_encoded += int(end_idx) - int(start_idx)

    def pop_metrics(self) -> VisionEncoderMetrics | None:
        """Return the metrics for the most recent batch and reset them.

        Returns ``None`` when the most recent batch preparation did no
        vision encoding (text-only or decode step). Intended to be called
        once per scheduler iteration.
        """
        metrics = self._batch_metrics
        self._batch_metrics = None
        return metrics

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
            start = offset
            offset += count
            # acquire for it, but still advance past its tokens in the encoder
            # output (this method only populates the cache for future reuse;
            # the current forward uses the encoder output directly, so skipping
            # is output-neutral).
            if not img_hash:
                continue
            # Allocate owned copies rather than views so the cache entry does
            # not pin the (variable-size) vision-encoder output buffer.  This
            # prevents GPU allocator fragmentation caused by mismatched holes
            # left behind when the output buffer is freed.
            per_device = [
                _owned_row_slice(dev_tensor, start, count)
                for dev_tensor in vision_outputs
            ]
            self.insert(img_hash, per_device, count)
            self.acquire(req_id, img_hash)

    @traced
    def prepare_vision_outputs(
        self,
        context_batch: Sequence[VLMContextType],
        uncached_contexts: Sequence[VLMContextType],
        uncached_images: Sequence[Sequence[ImageMetadata]],
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
            uncached_images: The cache-miss images per ``uncached_contexts``
                entry (the single source of the encode selection), aligned with
                the concatenation order of *vision_embeds*.
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
        for ctx, miss_images in zip(
            uncached_contexts, uncached_images, strict=True
        ):
            for img in miss_images:
                assert img.image_hash is not None
                hashes.append(img.image_hash)
                req_ids.append(ctx.request_id)
            if len(miss_images) != len(ctx.images):
                all_uncached = False  # a partial hit in this context

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

    @traced
    def select(
        self, context_batch: Sequence[VLMContextType]
    ) -> list[tuple[VLMContextType, list[ImageMetadata]]]:
        """Select contexts to encode, each paired with its cache-miss images.

        Computes the cache-miss set once (over ``ctx.next_images``), acquires
        refs for already-cached images immediately (so a hit can't be evicted
        between selection and assembly), and returns each selected context
        paired with its miss images. Every downstream consumer reads that same
        returned selection: the model's pack/encode steps, the counts
        (:func:`derive_counts_from_spans`), and the store/split
        (``prepare_vision_outputs``).

        ``get_uncached_contexts`` scans ``ctx.images`` (all images) rather than
        ``next_images`` to decide which contexts to return. That is consistent
        because a fully-processed image (in ``ctx.images`` but not
        ``next_images``) is always cache-resident: a request holds a ref on
        every image it has encoded, so the entry can't be evicted while the
        request is live, and the ``ctx.images`` scan sees it as a hit.
        """
        uncached = self.get_uncached_contexts(context_batch)
        return [
            (
                ctx,
                [
                    img
                    for img in ctx.next_images
                    if img.image_hash is None
                    or self.lookup(img.image_hash) is None
                ],
            )
            for ctx in uncached
        ]

    @traced
    def cache_vision_embeddings(
        self,
        context_batch: Sequence[VLMContextType],
        selection: Sequence[tuple[VLMContextType, Sequence[ImageMetadata]]],
        encode_result: VisionEncodeResult,
        empty_embeddings: list[Buffer],
    ) -> tuple[list[Buffer], npt.NDArray[np.int32]]:
        """Resolve/validate token counts, then store and assemble embeddings.

        Uses ``encode_result.per_image_token_counts`` when set, else derives
        them from placeholder spans (skipping images already resident).

        Returns:
            ``(embeddings, scatter_indices)`` — per-device buffers and a 1-D
            int32 scatter-index array.
        """
        counts = encode_result.per_image_token_counts
        if counts is None:
            counts = derive_counts_from_spans(selection)
        validate_vision_encode_counts(counts, encode_result.embeddings)
        result = self.prepare_vision_outputs(
            context_batch=context_batch,
            uncached_contexts=[ctx for ctx, _ in selection],
            uncached_images=[list(miss) for _, miss in selection],
            vision_embeds=encode_result.embeddings,
            per_image_token_counts=counts,
            n_devices=self._n_devices,
            empty_embeddings=empty_embeddings,
        )
        return result
