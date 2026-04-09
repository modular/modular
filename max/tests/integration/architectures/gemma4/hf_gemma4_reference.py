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

"""HuggingFace Gemma4 processor/image-processor reference for testing.

Copied from ``transformers.models.gemma4`` (installed under a venv) with:
- Relative imports converted to absolute ``transformers.*`` imports
- ``import torch`` removed; the single ``torch.from_numpy`` call replaced
  with a plain numpy list so the module is torch-free
- Non-essential decorators (``@auto_docstring``, ``@filter_out_non_signature_kwargs``)
  removed for portability across transformers versions

After import, these classes are registered with ``AutoImageProcessor`` and
``AutoProcessor`` so that ``*.from_pretrained("gg-hf-gg/gemma-4-31b-it")``
works without a transformers version that ships Gemma4 natively.
"""

# ruff: noqa

from __future__ import annotations

import itertools
import math
import re

import numpy as np
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    PretrainedConfig,
)
from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    make_nested_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config — register so AutoConfig can resolve model_type "gemma4"
# (model_config.py may have already done this; the try/except is harmless)
# ---------------------------------------------------------------------------


class _Gemma4HFConfig(PretrainedConfig):
    model_type = "gemma4"

    def __init__(self, vision_config=None, text_config=None, **kwargs):
        vision_config = vision_config or {}
        text_config = text_config or {}
        self.vision_config = PretrainedConfig(**vision_config)
        self.text_config = PretrainedConfig(**text_config)
        super().__init__(**kwargs)


try:
    AutoConfig.register("gemma4", _Gemma4HFConfig)
except ValueError:
    pass  # already registered

# ---------------------------------------------------------------------------
# Image processor (from image_processing_gemma4.py, torch removed)
# ---------------------------------------------------------------------------

_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


class Gemma4ImageProcessor(BaseImageProcessor):
    """HuggingFace Gemma4 image processor (reference copy, no torch)."""

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: int | float = 1 / 255,
        do_normalize: bool = False,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool | None = True,
        patch_size: int = 16,
        max_soft_tokens: int = 280,
        pooling_kernel_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size, default_to_square=True)
        image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, "
                f"got {max_soft_tokens}."
            )
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size

    def aspect_ratio_preserving_resize(
        self,
        image: np.ndarray,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
        resample: PILImageResampling,
        input_data_format: ChannelDimension | str | None = None,
    ) -> np.ndarray:
        height, width = get_image_size(image, channel_dim=input_data_format)
        total_px = height * width
        target_px = max_patches * (patch_size**2)
        factor = math.sqrt(target_px / total_px)
        ideal_height = factor * height
        ideal_width = factor * width
        side_mult = pooling_kernel_size * patch_size

        target_height = int(math.floor(ideal_height / side_mult)) * side_mult
        target_width = int(math.floor(ideal_width / side_mult)) * side_mult

        if target_height == 0 and target_width == 0:
            raise ValueError("Attempting to resize to a 0 x 0 image.")

        max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
        if target_height == 0:
            target_height = side_mult
            target_width = min(
                int(math.floor(width / height)) * side_mult,
                max_side_length,
            )
        elif target_width == 0:
            target_width = side_mult
            target_height = min(
                int(math.floor(height / width)) * side_mult,
                max_side_length,
            )

        if target_height * target_width > target_px:
            raise ValueError(
                f"Resizing [{height}x{width}] to "
                f"[{target_height}x{target_width}] but this exceeds "
                f"{max_patches} patches with patch_size {patch_size}"
            )

        if target_height == height and target_width == width:
            return image

        return resize(
            image,
            size=(target_height, target_width),
            resample=resample,
            input_data_format=input_data_format,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = None,
        size: dict[str, int] | None = None,
        resample: PILImageResampling | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension | None = ChannelDimension.FIRST,
        input_data_format: str | ChannelDimension | None = None,
        do_convert_rgb: bool | None = None,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor
            if rescale_factor is not None
            else self.rescale_factor
        )
        do_normalize = (
            do_normalize if do_normalize is not None else self.do_normalize
        )
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = (
            do_convert_rgb
            if do_convert_rgb is not None
            else self.do_convert_rgb
        )
        patch_size = patch_size if patch_size is not None else self.patch_size
        max_soft_tokens = (
            max_soft_tokens
            if max_soft_tokens is not None
            else self.max_soft_tokens
        )
        pooling_kernel_size = (
            pooling_kernel_size
            if pooling_kernel_size is not None
            else self.pooling_kernel_size
        )

        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, "
                f"got {max_soft_tokens}."
            )

        max_patches = max_soft_tokens * pooling_kernel_size**2

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be PIL.Image.Image, "
                "numpy.ndarray, or torch.Tensor"
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled "
                "images. If the input images have pixel values between 0 and "
                "1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        processed_images = []
        for image in images:
            if do_resize:
                image = self.aspect_ratio_preserving_resize(
                    image=image,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                    resample=resample,
                    input_data_format=input_data_format,
                )
            if do_rescale:
                image = self.rescale(
                    image=image,
                    scale=rescale_factor,
                    input_data_format=input_data_format,
                )
            if do_normalize:
                image = self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )
            image = to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format
            )
            processed_images.append(image)

        num_soft_tokens_per_image = []
        for img in processed_images:
            h, w = img.shape[-2], img.shape[-1]
            n_patches = (h // patch_size) * (w // patch_size)
            num_soft_tokens_per_image.append(
                n_patches // (pooling_kernel_size**2)
            )

        # Different-shaped images can't be stacked into a single array.
        # Skip tensor conversion for pixel_values when shapes are ragged.
        shapes = {img.shape for img in processed_images}
        skip_keys = {"pixel_values"} if len(shapes) > 1 else None

        data = {"pixel_values": processed_images}
        result = BatchFeature(
            data=data,
            tensor_type=return_tensors,
            skip_tensor_conversion=skip_keys,
        )
        result["num_soft_tokens_per_image"] = num_soft_tokens_per_image
        return result


# ---------------------------------------------------------------------------
# Processor (from processing_gemma4.py)
# ---------------------------------------------------------------------------

AudioInput = np.ndarray | list[float] | list[np.ndarray] | list[list[float]]


class Gemma4ProcessorKwargs(ProcessingKwargs):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "return_mm_token_type_ids": True,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
        },
        "audio_kwargs": {},
    }


class Gemma4Processor(ProcessorMixin):
    attributes = ["feature_extractor", "image_processor", "tokenizer"]
    feature_extractor_class = "Gemma4AudioFeatureExtractor"
    image_processor_class = "Gemma4ImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        feature_extractor=None,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_seq_length: int = 280,
        audio_seq_length: int = 750,
        audio_ms_per_token: int = 40,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.image_token = tokenizer.image_token
        image_tokens_expanded = "".join(
            [tokenizer.image_token] * image_seq_length
        )
        self.full_image_sequence = (
            f"{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}"
        )

        self.audio_seq_length = audio_seq_length
        self.audio_ms_per_token = audio_ms_per_token
        self.audio_token_id = getattr(tokenizer, "audio_token_id", None)
        self.audio_token = getattr(tokenizer, "audio_token", None)
        self.boa_token = getattr(tokenizer, "boa_token", None)
        self.eoa_token = getattr(tokenizer, "eoa_token", None)
        self.full_audio_sequence: str | None = None
        if self.audio_token and self.boa_token and self.eoa_token:
            audio_tokens_expanded = "".join(
                [self.audio_token] * audio_seq_length
            )
            self.full_audio_sequence = (
                f"{self.boa_token}{audio_tokens_expanded}{self.eoa_token}"
            )

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            feature_extractor=feature_extractor,
            **kwargs,
        )

    def __call__(
        self,
        images: ImageInput | None = None,
        text: (
            TextInput
            | PreTokenizedInput
            | list[TextInput]
            | list[PreTokenizedInput]
        ) = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[Gemma4ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None and audio is None:
            raise ValueError(
                "Provide at least one of `text`, `images`, or `audio`."
            )

        kwargs.pop("enable_thinking", None)

        output_kwargs = self._merge_kwargs(
            Gemma4ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError(
                "Invalid input text. Please provide a string, "
                "or a list of strings"
            )

        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(
                images, **output_kwargs["images_kwargs"]
            )

            num_soft_tokens = image_inputs.pop(
                "num_soft_tokens_per_image", None
            )

            if not text:
                text = [
                    " ".join([self.image_token] * len(imgs))
                    for imgs in batched_images
                ]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images "
                    f"({len(batched_images)}) and text ({len(text)})."
                )

            if num_soft_tokens is not None:
                replacements = [
                    f"{self.boi_token}{self.image_token * n}{self.eoi_token}"
                    for n in num_soft_tokens
                ]
                replacements_iter = iter(replacements)
            else:
                replacements_iter = itertools.repeat(self.full_image_sequence)

            pattern = re.escape(self.image_token)
            text = [
                re.sub(pattern, lambda _: next(replacements_iter), prompt)
                for prompt in text
            ]

        # Audio handling omitted for brevity (deferred)

        return_tensors = output_kwargs["text_kwargs"].pop(
            "return_tensors", None
        )
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", False
        )
        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            if self.audio_token_id is not None:
                mm_token_type_ids[array_ids == self.audio_token_id] = 2
            text_inputs["token_type_ids"] = mm_token_type_ids.tolist()

        skip_keys = set()
        if "pixel_values" in image_inputs and isinstance(
            image_inputs["pixel_values"], list
        ):
            if len(image_inputs["pixel_values"]) > 0 and hasattr(
                image_inputs["pixel_values"][0], "shape"
            ):
                shapes = {pv.shape for pv in image_inputs["pixel_values"]}
                if len(shapes) > 1:
                    skip_keys.add("pixel_values")

        return BatchFeature(
            data={**text_inputs, **image_inputs},
            tensor_type=return_tensors,
            skip_tensor_conversion=skip_keys or None,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + [
            "token_type_ids"
        ]
        image_processor_input_names = self.image_processor.model_input_names
        image_processor_input_names = [
            n for n in image_processor_input_names if n != "num_crops"
        ]
        return list(tokenizer_input_names + image_processor_input_names)


# ---------------------------------------------------------------------------
# Feature extractor stub (audio deferred — just enough for from_pretrained)
# ---------------------------------------------------------------------------

from transformers import AutoFeatureExtractor
from transformers.feature_extraction_sequence_utils import (
    SequenceFeatureExtractor,
)


class Gemma4AudioFeatureExtractor(SequenceFeatureExtractor):
    """Minimal stub so ``AutoProcessor.from_pretrained`` can load the processor.

    Audio support is deferred; this class only exists to satisfy the
    ``feature_extractor`` attribute declared by ``Gemma4Processor``.
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Register with Auto* classes
# ---------------------------------------------------------------------------

AutoFeatureExtractor.register(_Gemma4HFConfig, Gemma4AudioFeatureExtractor)
AutoImageProcessor.register(_Gemma4HFConfig, Gemma4ImageProcessor)
AutoProcessor.register(_Gemma4HFConfig, Gemma4Processor)
