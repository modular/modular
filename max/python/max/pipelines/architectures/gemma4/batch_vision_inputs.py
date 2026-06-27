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

"""Dataclasses and builders for batched vision / video model inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.graph.buffer_utils import cast_tensor_to
from max.pipelines.lib.vision_encoder_cache import (
    VisionEncoderCache,
    concat_device_buffers,
)
from max.pipelines.request import RequestID
from max.profiler import traced

from .context import Gemma4Context
from .vision_model.pooling import avg_pool_by_positions


@dataclass
class VisionRawInputs:
    """Raw vision-encoder inputs for a batch of uncached images or video frames.

    All buffer lists are per-device replicas (length = ``n_devices``),
    except ``max_seq_len`` which is a single CPU scalar.
    """

    patches_flat: list[Buffer]
    pixel_position_ids: list[Buffer]
    cu_seqlens: list[Buffer]
    pool_weights: list[Buffer]
    max_seq_len: Buffer


@dataclass
class ImageInputs:
    """Image-specific inputs attached to a model-input batch.

    Exactly one of ``raw`` or ``cached`` is populated:

    * ``raw`` — at least one image needs the vision encoder.  The
      ``cache_*`` fields carry metadata so ``execute`` can update the
      ``VisionEncoderCache`` after the forward pass.
    * ``cached`` — every image was already in the cache; pre-assembled
      embeddings and scatter indices are ready to use directly.
    """

    raw: VisionRawInputs | None = None

    cache_context_batch: Sequence[Gemma4Context] | None = None
    cache_uncached_contexts: Sequence[Gemma4Context] | None = None
    cache_per_image_token_counts: list[int] | None = None

    cached_embeddings: list[Buffer] | None = None
    cached_token_indices: list[Buffer] | None = None
    cached_token_indices_np: npt.NDArray[np.int32] | None = None


@dataclass
class VideoInputs:
    """Video-specific inputs attached to a model-input batch.

    Exactly one of ``raw`` or ``cached_embeddings`` is populated:

    * ``raw`` — at least one video needs encoding.  ``cache_hashes`` and
      ``cache_req_ids`` carry metadata so ``execute`` can store the result.
    * ``cached_embeddings`` — every video was already in the cache; embeddings
      are pre-assembled and ready to use directly.
    """

    raw: VisionRawInputs | None = None
    token_indices: list[Buffer] | None = None
    token_indices_np: npt.NDArray[np.int32] | None = None

    # Metadata for caching freshly-encoded videos in execute().
    cache_hashes: list[int] | None = None
    cache_req_ids: list[RequestID] | None = None
    cache_per_video_token_counts: list[int] | None = None

    # Cache-hit path: all videos already cached.
    cached_embeddings: list[Buffer] | None = None


def create_empty_embeddings(
    devices: list[Device], hidden_size: int, dtype: DType = DType.bfloat16
) -> list[Buffer]:
    """Create empty (zero-row) embedding buffers, one per device."""
    return [
        Buffer.zeros(shape=[0, hidden_size], dtype=dtype).to(dev)
        for dev in devices
    ]


def create_empty_indices(devices: list[Device]) -> list[Buffer]:
    """Create empty (zero-length) scatter-index buffers, one per device."""
    return [
        Buffer.zeros(shape=[0], dtype=DType.int32).to(dev) for dev in devices
    ]


def merge_per_device_buffers(
    a_bufs: list[Buffer],
    b_bufs: list[Buffer],
) -> list[Buffer]:
    """Concatenate two per-device buffer lists element-wise along axis 0.

    When either side is empty the other is returned directly. Otherwise the
    concat stays on-device: allocate the combined buffer and copy each half
    into a contiguous leading-axis slice, avoiding a GPU->host->GPU round-trip.
    """
    merged: list[Buffer] = []
    for a, b in zip(a_bufs, b_bufs, strict=True):
        a_rows = a.shape[0]
        b_rows = b.shape[0]
        if a_rows == 0 and b_rows == 0:
            merged.append(a)
        elif a_rows == 0:
            merged.append(b)
        elif b_rows == 0:
            merged.append(a)
        else:
            combined = Buffer(
                shape=(a_rows + b_rows, *a.shape[1:]),
                dtype=a.dtype,
                device=a.device,
            )
            # Slice the leading axis at the buffer's rank: this merges both
            # rank-2 embeddings and rank-1 indices, and MAX requires
            # index-count == rank.
            n_extra = len(a.shape) - 1
            front = (slice(None, a_rows), *(slice(None),) * n_extra)
            back = (slice(a_rows, None), *(slice(None),) * n_extra)
            combined[front].inplace_copy_from(a)
            combined[back].inplace_copy_from(b)
            merged.append(combined)
    return merged


def _pinned_to_devices(
    np_array: npt.NDArray[Any], dtype: DType, devices: list[Device]
) -> list[Buffer]:
    """Copy a numpy array to each device via a pinned host buffer."""
    dev0 = devices[0]
    host: Buffer
    if not dev0.is_host:
        host = DevicePinnedBuffer(
            dtype=dtype, shape=np_array.shape, device=dev0
        )
    else:
        host = Buffer(shape=np_array.shape, dtype=dtype, device=dev0)
    host.to_numpy()[:] = np_array
    device_bufs = [host.to(d) for d in devices]
    for d in device_bufs:
        d.inplace_copy_from(host)
    return device_bufs


@traced
def pack_vision_buffers(
    devices: list[Device],
    pooling_kernel_size: int,
    all_patches: list[npt.NDArray[np.floating[Any]]],
    all_pos_ids: list[npt.NDArray[np.integer[Any]]],
    patch_counts: list[int],
    soft_token_counts: list[int],
    dtype: DType,
) -> VisionRawInputs:
    """Build device-replicated ``VisionRawInputs`` from numpy arrays."""
    patches_flat_np = np.concatenate(all_patches, axis=0).astype(np.float32)
    pos_ids_np = np.concatenate(all_pos_ids, axis=0)

    n_items = len(all_patches)
    cu_seqlens_np = np.empty(n_items + 1, dtype=np.uint32)
    cu_seqlens_np[0] = 0
    np.cumsum(patch_counts, out=cu_seqlens_np[1:])

    max_seq_len_np = np.array(max(patch_counts), dtype=np.uint32)
    pool_weights_np = avg_pool_by_positions(
        all_pos_ids, soft_token_counts, pooling_kernel_size
    )

    # Use pinned host buffers for h2d copies.
    patches_flat_bufs = _pinned_to_devices(
        patches_flat_np, DType.float32, devices
    )
    patches_flat = [cast_tensor_to(buf, dtype) for buf in patches_flat_bufs]

    pool_weights_bufs = _pinned_to_devices(
        pool_weights_np.astype(np.float32), DType.float32, devices
    )

    return VisionRawInputs(
        patches_flat=patches_flat,
        pixel_position_ids=_pinned_to_devices(
            pos_ids_np.astype(np.int32), DType.int32, devices
        ),
        cu_seqlens=_pinned_to_devices(cu_seqlens_np, DType.uint32, devices),
        pool_weights=pool_weights_bufs,
        max_seq_len=Buffer.from_numpy(max_seq_len_np),
    )


@traced
def build_image_inputs(
    context_batch: Sequence[Gemma4Context],
    uncached: Sequence[Gemma4Context],
    devices: list[Device],
    pooling_kernel_size: int,
    ve_cache: VisionEncoderCache[Gemma4Context],
    empty_embeddings: list[Buffer],
    dtype: DType,
) -> ImageInputs | None:
    """Assemble ``ImageInputs`` — raw or cached — for a batch."""
    k = pooling_kernel_size

    if uncached:
        all_patches: list[npt.NDArray[np.floating[Any]]] = []
        all_pos_ids: list[npt.NDArray[np.integer[Any]]] = []
        patch_counts: list[int] = []
        soft_token_counts: list[int] = []

        for ctx in uncached:
            # Slice off already-encoded images so pixel_position_ids (the full
            # per-image list) realigns with next_images under chunked prefill.
            ctx_pos_ids = ctx.pixel_position_ids[ctx.image_idx :]
            for img_idx, img in enumerate(ctx.next_images):
                num_soft = img.end_idx - img.start_idx
                num_patches = num_soft * k * k
                if num_patches != len(img.pixel_values):
                    raise ValueError(
                        f"Expected {num_patches} patches, "
                        f"got {len(img.pixel_values)}"
                    )
                if (
                    img.image_hash is not None
                    and ve_cache.lookup(img.image_hash) is not None
                ):
                    continue
                all_patches.append(img.pixel_values)
                all_pos_ids.append(ctx_pos_ids[img_idx])
                patch_counts.append(num_patches)
                soft_token_counts.append(num_soft)

        per_image_token_counts = [
            img.end_idx - img.start_idx
            for ctx in uncached
            for img in ctx.next_images
            if img.image_hash is None or ve_cache.lookup(img.image_hash) is None
        ]

        raw = (
            pack_vision_buffers(
                devices,
                pooling_kernel_size,
                all_patches,
                all_pos_ids,
                patch_counts,
                soft_token_counts,
                dtype,
            )
            if all_patches
            else None
        )

        return ImageInputs(
            raw=raw,
            cache_context_batch=context_batch,
            cache_uncached_contexts=uncached,
            cache_per_image_token_counts=per_image_token_counts,
        )

    # All images are cached (or no images at all).
    cached_embeds, scatter_np = ve_cache.prepare_vision_outputs(
        context_batch=context_batch,
        uncached_contexts=uncached,
        vision_embeds=empty_embeddings,
        per_image_token_counts=[],
        n_devices=len(devices),
        empty_embeddings=empty_embeddings,
    )
    if scatter_np is not None and len(scatter_np) > 0:
        return ImageInputs(
            cached_embeddings=cached_embeds,
            cached_token_indices_np=scatter_np.astype(np.int32),
        )

    return None


@traced
def build_video_inputs(
    context_batch: Sequence[Gemma4Context],
    devices: list[Device],
    pooling_kernel_size: int,
    dtype: DType,
    ve_cache: VisionEncoderCache[Gemma4Context] | None = None,
    empty_embeddings: list[Buffer] | None = None,
) -> VideoInputs | None:
    """Assemble ``VideoInputs`` from pre-unpacked per-frame context data.

    When *ve_cache* is provided, videos already in the cache skip encoding.
    Scatter indices are always rebuilt per chunk — they depend on
    ``processed_length``, not on whether encoding was skipped.
    """
    video_ctxs = [ctx for ctx in context_batch if ctx.video_frame_patches]
    if not video_ctxs:
        return None

    # Build chunked-prefill-aware scatter indices for all video contexts.
    # Absolute token positions are mapped into the current chunk's active
    # window; positions outside the window get the int32-min sentinel so
    # scatter_nd_skip_oob_indices ignores them.
    oob_idx = np.iinfo(np.int32).min
    batch_offset = 0
    scatter_parts: list[npt.NDArray[np.int32]] = []
    for ctx in context_batch:
        processed_length = ctx.tokens.processed_length
        active_len = ctx.tokens.active_length
        for start, end in ctx.video_token_ranges:
            rel = np.arange(start, end, dtype=np.int64) - processed_length
            valid = (rel >= 0) & (rel < active_len)
            scatter_parts.append(
                np.where(
                    valid,
                    (rel + batch_offset).astype(np.int32),
                    np.int32(oob_idx),
                )
            )
        batch_offset += active_len
    scatter_np = np.concatenate(scatter_parts).astype(np.int32)

    # Cache-aware path: check each video against the cache.
    if ve_cache is not None and ve_cache.enabled:
        uncached_ctxs: list[Gemma4Context] = []
        for ctx in video_ctxs:
            for vh in ctx.video_hashes:
                if ve_cache.lookup(vh):
                    ve_cache.acquire(ctx.request_id, vh)
                else:
                    uncached_ctxs.append(ctx)
                    break  # miss on at least one video — encode this ctx

        if not uncached_ctxs:
            # All videos cached — assemble embeddings from cache entries.
            assert empty_embeddings is not None
            all_device_bufs: list[list[Buffer]] = [[] for _ in devices]
            for ctx in video_ctxs:
                for vh in ctx.video_hashes:
                    if entry := ve_cache.lookup(vh):
                        for d in range(len(devices)):
                            all_device_bufs[d].append(entry.embeddings[d])
            if any(bufs for bufs in all_device_bufs):
                cached_embs: list[Buffer] = (
                    [dl[0] for dl in all_device_bufs]
                    if all(len(dl) == 1 for dl in all_device_bufs)
                    else [concat_device_buffers(dl) for dl in all_device_bufs]
                )
            else:
                cached_embs = empty_embeddings
            return VideoInputs(
                cached_embeddings=cached_embs, token_indices_np=scatter_np
            )

        video_ctxs = uncached_ctxs  # encode only uncached contexts

    # Raw (encode) path: pack pixel buffers and collect cache metadata.
    all_frame_patches: list[npt.NDArray[np.floating[Any]]] = []
    all_frame_pos_ids: list[npt.NDArray[np.integer[Any]]] = []
    frame_patch_counts: list[int] = []
    frame_soft_token_counts: list[int] = []
    cache_hashes: list[int] = []
    cache_req_ids: list[RequestID] = []
    cache_per_video_token_counts: list[int] = []
    for ctx in video_ctxs:
        all_frame_patches.extend(ctx.video_frame_patches)
        all_frame_pos_ids.extend(ctx.video_frame_pos_ids)
        frame_patch_counts.extend(ctx.video_frame_patch_counts)
        frame_soft_token_counts.extend(ctx.video_frame_soft_token_counts)
        for i, (start, end) in enumerate(ctx.video_token_ranges):
            cache_hashes.append(
                ctx.video_hashes[i] if i < len(ctx.video_hashes) else 0
            )
            cache_req_ids.append(ctx.request_id)
            cache_per_video_token_counts.append(end - start)

    raw = pack_vision_buffers(
        devices,
        pooling_kernel_size,
        all_frame_patches,
        all_frame_pos_ids,
        frame_patch_counts,
        frame_soft_token_counts,
        dtype,
    )
    return VideoInputs(
        raw=raw,
        token_indices_np=scatter_np,
        cache_hashes=cache_hashes,
        cache_req_ids=cache_req_ids,
        cache_per_video_token_counts=cache_per_video_token_counts,
    )
