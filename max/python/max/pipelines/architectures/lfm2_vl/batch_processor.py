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
"""Ragged batching for Lfm2VlForConditionalGeneration (ModuleV3).

The framework packs the ragged text batch (``DecoderBatchProcessor``)
and its vision cache selects each step's not-yet-encoded images; this
processor packs the tower's inputs for those images: the processor's
unpadded patch rows, plus four host-built tables the graph cannot
derive in-graph — the bilinear reads of the learned position table,
the per-tile ``cu_seqlens``, the largest tile's patch count, and the
2x2 fold indices.

Everything the tower needs is a function of the per-tile grids the
stock tokenizer forwards from the processor (``extra_model_args
["spatial_shapes"]``, see ``tokenizer.py``), so the tables are built
here in numpy and are exact integers/floats, not a second
implementation to drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib.module_registration import DecoderBatchProcessor
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor.compilation import CompiledCallable

from .model_config import Lfm2VlConfig
from .tokenizer import TILE_GRIDS_KEY


def _bilinear_rows(
    grid: int, source: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Torch-bilinear (`align_corners=False`) reads of a 1-D axis.

    For each of `grid` outputs, the two source rows it reads and their
    weights: torch samples at (t + 0.5) * S / T - 0.5, clamped into
    [0, S - 1]. Plain bilinear — the antialiased downscale torch adds
    when scaling BELOW the source grid is not replicated (see the
    vision/tower.py TODO).
    """
    t = np.arange(grid, dtype=np.float64) + 0.5
    s = np.clip(t * source / grid - 0.5, 0.0, source - 1)
    i0 = np.floor(s).astype(np.int64)
    i1 = np.minimum(i0 + 1, source - 1)
    w1 = s - i0
    return i0, i1, w1


@dataclass
class Lfm2VlVisionRawInputs:
    """The tower's six buffers for one launch, its input tree's order."""

    pixel_values: Buffer
    pos_idx: Buffer
    pos_w: Buffer
    tile_offsets: Buffer
    max_tile_len: Buffer
    block_gather: Buffer


class Lfm2VlBatchProcessor(DecoderBatchProcessor):
    """Ragged text batching, plus the tower's per-launch tables.

    Single GPU, so the base class's text packing needs no device
    override (gemma4's pins the vision operands to device zero for its
    broadcast; there is nothing to pin here). Images go through the
    pipeline-owned ``VisionEncoderCache``, which selects each step's
    images and scatters the embeddings from the token spans the
    tokenizer recorded.
    """

    _vision_model: CompiledCallable[Any, Any] | None = None
    _empty_vision_embeddings: list[Buffer] | None = None

    @property
    def _arch_config(self) -> Lfm2VlConfig:
        """Reads the framework-bound finalized config rather than a cached copy."""
        assert isinstance(self.config, Lfm2VlConfig)
        return self.config

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledCallable[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Wires the compiled vision encoder."""
        del session
        self._vision_model = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> Lfm2VlVisionRawInputs:
        """Packs the pipeline-selected images' pixels and tables to device."""
        config = self._arch_config
        device0 = devices[0]
        source = config.vision_position_grid

        # Unpadded patch rows, tile after tile (the tokenizer already
        # unpacked each request's canvas in processor tile order).
        pixels: list[np.ndarray] = []
        pos_idx_rows: list[np.ndarray] = []
        pos_w_rows: list[np.ndarray] = []
        tile_lengths: list[int] = []
        fold_rows: list[np.ndarray] = []
        base = 0
        for ctx, images in selection:
            raw = ctx.extra_model_args.get(TILE_GRIDS_KEY)
            if raw is None:
                raise ValueError(
                    "a context carries images but no recorded tile "
                    "grids: the tokenizer must record the processor's "
                    "`spatial_shapes` (Lfm2VlTokenizer), or the tower "
                    "cannot place its patches"
                )
            grid = np.asarray(raw, dtype=np.int64).reshape(-1, 2)
            covered = 0
            for image in images:
                rows = np.asarray(image.pixel_values, dtype=np.float32)
                if rows.ndim != 2:
                    rows = rows.reshape(-1, rows.shape[-1])
                if rows.shape[1] != config.vision_patch_dim:
                    raise ValueError(
                        f"patch rows of width {rows.shape[1]}, expected "
                        f"{config.vision_patch_dim} (16x16x3)"
                    )
                pixels.append(rows)
                covered += int(rows.shape[0])
            if covered != int(grid.prod(axis=1).sum()):
                raise ValueError(
                    f"{covered} patch rows for grids covering "
                    f"{int(grid.prod(axis=1).sum())} patches: the "
                    "processor's pixel layout changed"
                )
            for ph, pw in (tuple(map(int, row)) for row in grid):
                if (
                    ph % config.downsample_factor
                    or pw % config.downsample_factor
                ):
                    raise ValueError(
                        f"a {ph}x{pw} patch grid is not even in both axes: "
                        "the processor's 32-pixel resize did not run, and "
                        f"the {config.downsample_factor}x"
                        f"{config.downsample_factor} fold would eat a "
                        "partial block"
                    )
                if ph < source or pw < source:
                    # Below the learned source x source position grid,
                    # the reference downsamples that grid ANTIALIASED
                    # (HF siglip2 upcasts to float32 for it); the plain
                    # two-tap bilinear here would read different
                    # positional embeddings. Refuse until the
                    # antialiased resample is implemented.
                    raise ValueError(
                        f"a {ph}x{pw} patch grid is smaller than the "
                        f"{source}x{source} learned position grid: the "
                        "antialiased downscale the reference applies "
                        "there is not implemented, so the tower would "
                        "place patches against the wrong positional "
                        "embeddings"
                    )
                ys0, ys1, wy = _bilinear_rows(ph, source)
                xs0, xs1, wx = _bilinear_rows(pw, source)
                i00 = ys0[:, None] * source + xs0[None, :]
                i10 = ys1[:, None] * source + xs0[None, :]
                i01 = ys0[:, None] * source + xs1[None, :]
                i11 = ys1[:, None] * source + xs1[None, :]
                pos_idx_rows.append(
                    np.stack([i00, i10, i01, i11], axis=-1).reshape(ph * pw, 4)
                )
                w00 = (1 - wy)[:, None] * (1 - wx)[None, :]
                w10 = wy[:, None] * (1 - wx)[None, :]
                w01 = (1 - wy)[:, None] * wx[None, :]
                w11 = wy[:, None] * wx[None, :]
                pos_w_rows.append(
                    np.stack([w00, w10, w01, w11], axis=-1).reshape(ph * pw, 4)
                )
                # The 2x2 fold: block (bh, bw) gathers its four patches
                # in (jh, jw) order — top-left, top-right, bottom-left,
                # bottom-right (see vision/projector.py).
                index = np.arange(ph * pw).reshape(ph, pw)
                blocks = np.stack(
                    [
                        index[0::2, 0::2].reshape(-1),
                        index[0::2, 1::2].reshape(-1),
                        index[1::2, 0::2].reshape(-1),
                        index[1::2, 1::2].reshape(-1),
                    ],
                    axis=-1,
                )
                fold_rows.append(blocks + base)
                tile_lengths.append(ph * pw)
                base += ph * pw

        if base != sum(int(p.shape[0]) for p in pixels):
            raise ValueError(
                f"{sum(int(p.shape[0]) for p in pixels)} patch rows for "
                f"grids covering {base} patches: the tokenizer's "
                "unpick dropped or duplicated rows"
            )
        pixel_values = np.concatenate(pixels, axis=0)
        pos_idx = np.concatenate(pos_idx_rows, axis=0).astype(np.uint32)
        pos_w = np.concatenate(pos_w_rows, axis=0).astype(np.float32)
        block_gather = (
            np.concatenate(fold_rows, axis=0).astype(np.uint32)
            if fold_rows
            else np.zeros((0, 4), dtype=np.uint32)
        )
        tile_offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(tile_lengths)]
        ).astype(np.uint32)
        max_tile_len = np.array(
            [max(tile_lengths) if tile_lengths else 0], dtype=np.uint32
        )
        return Lfm2VlVisionRawInputs(
            pixel_values=Buffer.from_numpy(pixel_values).to(device0),
            pos_idx=Buffer.from_numpy(pos_idx).to(device0),
            pos_w=Buffer.from_numpy(pos_w).to(device0),
            tile_offsets=Buffer.from_numpy(tile_offsets).to(device0),
            # Stays on the HOST: the ragged kernel validates max_seq_len
            # as a CPU uint32 scalar (see max.nn.kernels).
            max_tile_len=Buffer.from_numpy(max_tile_len),
            block_gather=Buffer.from_numpy(block_gather).to(device0),
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: Lfm2VlVisionRawInputs | None,
    ) -> VisionEncodeResult:
        """Runs the tower on the packed pixels."""
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        if self._vision_model is None:
            raise ValueError(
                "No vision encoder is loaded; image inputs are not supported."
            )
        # The tower runs on device 0, and its embeddings are copied to
        # the other devices. Its inputs are its named tree's leaves, in
        # order.
        (device0_embeddings,) = self._vision_model.execute_raw(
            packed.pixel_values,
            packed.pos_idx,
            packed.pos_w,
            packed.tile_offsets,
            packed.max_tile_len,
            packed.block_gather,
        )
        return VisionEncodeResult(
            embeddings=[device0_embeddings.to(device) for device in devices]
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for text-only / decode steps.

        This is hit on every text-only step, so it must not allocate per
        call. The dtype MUST be the arch's model dtype: the language
        graph declares the vision merge at ``config.dtype`` (float32
        under the parity gate's fp32 serve), and an empty batch must
        type-match a full one.
        """
        if self._empty_vision_embeddings is None:
            host = Buffer.zeros(
                shape=[0, self._arch_config.hidden_size],
                dtype=self._arch_config.dtype,
            )
            self._empty_vision_embeddings = [host.to(d) for d in devices]
        return self._empty_vision_embeddings
