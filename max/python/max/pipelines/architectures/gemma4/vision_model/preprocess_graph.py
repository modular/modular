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
"""Gemma 4 image preprocessing as a CPU graph a serving engine runs.

The graph does for one image what ``Gemma4ImageProcessor`` and
``pack_vision_buffers`` do: it composites the image on white, resizes it
with Pillow's antialiased bicubic filter, accumulating in float32 and
rounding once to uint8, rescales and patchifies it, and computes the
tower's patch positions, segment bounds and pooling gather table.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.nn.legacy.graph_groups import (
    VisionHostMax,
    VisionRowIndex,
    VisionRows,
    VisionSegments,
)
from max.nn.legacy.image_preprocess import (
    composite_on_white,
    resize_bicubic_to,
)
from max.nn.legacy.layer import Module

from ..image_processor import Gemma4ImageProcessor


class Gemma4ImagePreprocess(Module):
    """Turns one RGBA8 image into the vision tower's inputs.

    Takes
    :func:`~max.nn.legacy.image_preprocess.image_preprocess_input_types`,
    the size being the processor's aspect-ratio-preserving target, and
    returns the tower's one-device input tree for the image, on the CPU.
    """

    def __init__(self, processor: Gemma4ImageProcessor, dtype: DType) -> None:
        super().__init__()
        self.processor = processor
        self.dtype = dtype

    def __call__(
        self, image: TensorValue, size: TensorValue
    ) -> OrderedDict[str, object]:
        """Returns the tower's input tree for one image."""
        processor = self.processor
        patch = processor.patch_size
        k = processor.pooling_kernel_size
        cpu = DeviceRef.CPU()

        levels = resize_bicubic_to(
            composite_on_white(image), size, multiple=patch * k
        )[0]
        # The graph's float32 division can land an ulp from numpy's, so each
        # channel's 256 levels are rescaled and normalized in numpy, as the
        # processor does, and looked up.
        table = np.broadcast_to(np.arange(256, dtype=np.float32), (3, 256))
        if processor.do_rescale:
            table = table * processor.rescale_factor
        if processor.do_normalize:
            table = (
                table - processor.image_mean[:, None]
            ) / processor.image_std[:, None]
        channel_base = ops.constant(
            np.arange(0, 3 * 256, 256, dtype=np.int32).reshape(3, 1, 1),
            DType.int32,
            cpu,
        )
        pixels = ops.gather(
            ops.constant(table.reshape(-1), DType.float32, cpu),
            levels.cast(DType.int32) + channel_base,
            axis=0,
        )

        channels, height, width = pixels.shape
        blocks_h = height // (patch * k)
        blocks_w = width // (patch * k)
        patches_h, patches_w = blocks_h * k, blocks_w * k
        seq = patches_h * patches_w
        # Rows run over the patch grid row-major; a patch's values are
        # pixel-row-major with the channels innermost.
        patches_flat = (
            pixels.reshape([channels, patches_h, patch, patches_w, patch])
            .permute([1, 3, 2, 4, 0])
            .reshape([seq, patch * patch * channels])
            .cast(self.dtype)
        )

        def axis(side: Dim, shape: list[Dim | int]) -> TensorValue:
            positions = ops.range(0, side, 1, dtype=DType.int32, device=cpu)
            return ops.broadcast_to(
                positions.reshape(shape), [patches_h, patches_w]
            )

        pixel_position_ids = ops.stack(
            [axis(patches_w, [1, patches_w]), axis(patches_h, [patches_h, 1])],
            axis=-1,
        ).reshape([seq, 2])
        # Each pooled token averages the k x k patches of its block, listed
        # row-major. The bins are full, so no slot needs the sentinel.
        pool_gather_index = (
            ops.range(0, seq, 1, dtype=DType.int32, device=cpu)
            .reshape([blocks_h, k, blocks_w, k])
            .permute([0, 2, 1, 3])
            .reshape([blocks_h * blocks_w, k * k])
        )
        return OrderedDict(
            patches=VisionRows(
                patches_flat=(patches_flat,),
                pixel_position_ids=(pixel_position_ids,),
            ),
            segments=VisionSegments(
                cu_seqlens=(ops.shape_to_tensor([0, seq]).cast(DType.uint32),)
            ),
            pooling=VisionRowIndex(pool_gather_index=(pool_gather_index,)),
            segment_bound=VisionHostMax(
                max_seq_len=ops.shape_to_tensor([seq])
                .reshape([])
                .cast(DType.uint32)
            ),
        )
