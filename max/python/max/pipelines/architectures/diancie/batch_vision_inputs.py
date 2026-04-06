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
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache
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
    """Video-specific inputs attached to a model-input batch."""

    raw: VisionRawInputs
    token_indices: list[Buffer] | None = None
    token_indices_np: npt.NDArray[np.int32] | None = None


def create_empty_embeddings(
    devices: list[Device], hidden_size: int
) -> list[Buffer]:
    """Create empty (zero-row) embedding buffers, one per device."""
    return [
        Buffer.zeros(shape=[0, hidden_size], dtype=DType.bfloat16).to(dev)
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
    """Concatenate two per-device buffer lists element-wise.

    When either side is empty the other is returned directly, avoiding
    unnecessary round-trips through NumPy.
    """
    merged: list[Buffer] = []
    for a, b in zip(a_bufs, b_bufs, strict=True):
        a_empty = a.shape[0] == 0
        b_empty = b.shape[0] == 0
        if a_empty and b_empty:
            merged.append(a)
        elif a_empty:
            merged.append(b)
        elif b_empty:
            merged.append(a)
        else:
            dev = a.device
            a_np = a.to(Device()).to_numpy()
            b_np = b.to(Device()).to_numpy()
            combined = np.concatenate([a_np, b_np], axis=0)
            merged.append(Buffer.from_numpy(combined).to(dev))
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
    patches_flat = [
        cast_tensor_to(buf, DType.bfloat16) for buf in patches_flat_bufs
    ]

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
) -> ImageInputs | None:
    """Assemble ``ImageInputs`` — raw or cached — for a batch."""
    k = pooling_kernel_size

    if uncached:
        all_patches: list[npt.NDArray[np.floating[Any]]] = []
        all_pos_ids: list[npt.NDArray[np.integer[Any]]] = []
        patch_counts: list[int] = []
        soft_token_counts: list[int] = []

        for ctx in uncached:
            ctx_pos_ids = ctx.pixel_position_ids
            if ctx.next_images and len(ctx_pos_ids) != len(ctx.next_images):
                raise ValueError(
                    f"Expected {len(ctx.next_images)} pixel_position_ids, "
                    f"got {len(ctx_pos_ids)}"
                )

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
) -> VideoInputs | None:
    """Assemble ``VideoInputs`` from pre-unpacked per-frame context data."""
    all_frame_patches: list[npt.NDArray[np.floating[Any]]] = []
    all_frame_pos_ids: list[npt.NDArray[np.integer[Any]]] = []
    frame_patch_counts: list[int] = []
    frame_soft_token_counts: list[int] = []

    batch_offset = 0
    scatter_parts: list[npt.NDArray[np.int32]] = []

    for ctx in context_batch:
        all_frame_patches.extend(ctx.video_frame_patches)
        all_frame_pos_ids.extend(ctx.video_frame_pos_ids)
        frame_patch_counts.extend(ctx.video_frame_patch_counts)
        frame_soft_token_counts.extend(ctx.video_frame_soft_token_counts)

        for start, end in ctx.video_token_ranges:
            scatter_parts.append(
                np.arange(
                    batch_offset + start,
                    batch_offset + end,
                    dtype=np.int32,
                )
            )
        batch_offset += len(ctx.tokens.active)

    if not all_frame_patches:
        return None

    raw = pack_vision_buffers(
        devices,
        pooling_kernel_size,
        all_frame_patches,
        all_frame_pos_ids,
        frame_patch_counts,
        frame_soft_token_counts,
    )
    scatter_np = np.concatenate(scatter_parts).astype(np.int32)
    return VideoInputs(raw=raw, token_indices_np=scatter_np)
