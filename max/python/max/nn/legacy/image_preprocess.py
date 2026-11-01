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
"""Building blocks of a vision model's image preprocessing graph.

A ``vision_preprocess`` graph turns one decoded image into its vision
encoder's inputs, on the CPU. It takes the image as RGBA8, which is how a
serving engine decodes every image, and the size the host resizes it to: the
host sizes each image itself, since the prompt's token count depends on the
size before any graph runs.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, TensorValue, ops

__all__ = [
    "composite_on_white",
    "image_preprocess_input_types",
    "resize_bicubic_to",
    "video_preprocess_input_types",
]


def image_preprocess_input_types() -> list[TensorType]:
    """Returns an image preprocessing graph's inputs.

    They are the decoded image, ``uint8 [1, image_height, image_width, 4]``
    RGBA, and the height and width to resize it to, ``int64 [2]``. The alpha
    is the image's own only for an image Pillow opens as ``RGBA``; for every
    other image, palette transparency and ``LA`` included, it is 255. A
    graph composites on it or drops it, as its model's processor does.
    """
    cpu = DeviceRef.CPU()
    return [
        TensorType(
            DType.uint8, [1, "image_height", "image_width", 4], device=cpu
        ),
        TensorType(DType.int64, [2], device=cpu),
    ]


def video_preprocess_input_types(
    frames: int | str, sizes: int
) -> list[TensorType]:
    """Returns a video preprocessing graph's inputs.

    The inputs are for one encoder attention segment of a clip.

    They are the segment's decoded frames, ``uint8 [frames, frame_height,
    frame_width, 4]`` RGBA with alpha 255, and the heights and widths to
    resize them to, in order, ``int64 [sizes, 2]``.

    Args:
        frames: The frames per segment, or a symbolic dim's name when a
            segment's frame count varies.
        sizes: How many resizes the graph applies in turn.
    """
    cpu = DeviceRef.CPU()
    return [
        TensorType(
            DType.uint8, [frames, "frame_height", "frame_width", 4], device=cpu
        ),
        TensorType(DType.int64, [sizes, 2], device=cpu),
    ]


def composite_on_white(image: TensorValue) -> TensorValue:
    """Pastes an RGBA8 image onto white through its alpha, as Pillow does.

    Pillow rounds ``(color * alpha + 255 * (255 - alpha)) / 255`` to the
    nearest integer. The quotient is never a tie, so float32 rounds it the
    same way.

    Args:
        image: ``uint8 [1, height, width, 4]`` RGBA.

    Returns:
        The RGB image, ``uint8 [1, height, width, 3]``.
    """
    pixels = image.cast(DType.float32)
    color = pixels[:, :, :, :3]
    alpha = pixels[:, :, :, 3:]
    return ops.round((color * alpha + 255.0 * (255.0 - alpha)) / 255.0).cast(
        DType.uint8
    )


def resize_bicubic_to(
    images: TensorValue, size: TensorValue, multiple: int
) -> TensorValue:
    """Resizes uint8 RGB images with the antialiased bicubic filter of Pillow.

    A Hugging Face processor resizes a uint8 image to uint8, with Pillow or
    torchvision, and only then rescales and normalizes it. This resize
    accumulates in float32, as PyTorch's antialiased bicubic does, and
    rounds half to even and clamps to [0, 255] once, at the end. It does not
    round and clamp between its passes as Pillow's uint8 resize does, so it
    is within a level or two of Pillow's on smooth content, and further
    around hard edges near black or white, whose overshoot Pillow clamps.

    Args:
        images: ``uint8 [frames, height, width, 3]`` RGB.
        size: ``int64 [2]``, the height and width to resize to. Each must be
            a multiple of ``multiple``, which the graph checks when it runs.
        multiple: What the resized sides are multiples of, such as the patch
            size times the merge size.

    Returns:
        ``uint8 [frames, 3, height, width]``, channels first.
    """
    frames = images.shape[0]
    resized = ops.resize_bicubic(
        images.permute([0, 3, 1, 2]),
        ops.concat([ops.shape_to_tensor([frames, 3]), size]),
        antialias=True,
        out_dims=[frames, 3, "resized_height", "resized_width"],
    )
    height, width = (
        Dim(side) // multiple * multiple
        for side in ("resized_height", "resized_width")
    )
    return resized.rebind(
        [frames, 3, height, width],
        message=f"the resized sides must be multiples of {multiple}",
    )
