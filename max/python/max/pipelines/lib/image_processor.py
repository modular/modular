# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import logging

import numpy as np
import PIL.Image
from max.driver import Buffer as DTensor
from max.dtype import DType
from max.experimental import Tensor
from max.experimental import functional as F
from max.experimental.compile_utils import max_compile
from max.graph import DeviceRef, TensorType, TensorValue, ops
from PIL import Image

logger = logging.getLogger(__name__)


PipelineImageInput = (
    PIL.Image.Image
    | np.ndarray
    | Tensor
    | list[PIL.Image.Image]
    | list[np.ndarray]
    | list[Tensor]
)


class VaeImageProcessor:
    config_name = "config.json"

    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 4,
        resample: str = "lanczos",
        reducing_gap: int | None = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
        device: DeviceRef = DeviceRef.GPU(),
        dtype: DType = DType.bfloat16,
    ):
        """Initialize the VaeImageProcessor.

        Args:
            do_resize (bool, optional): Whether to resize images. Defaults to True.
            vae_scale_factor (int, optional): The VAE scale factor. Defaults to 8.
            vae_latent_channels (int, optional): The number of latent channels for the VAE. Defaults to 4.
            resample (str, optional): The resampling mode for resizing. Defaults to "lanczos".
            reducing_gap (int, optional): A reduction gap parameter for resampling. Defaults to None.
            do_normalize (bool, optional): Whether to normalize images to [-1, 1]. Defaults to True.
            do_binarize (bool, optional): Whether to binarize images. Defaults to False.
            do_convert_rgb (bool, optional): Whether to convert images to RGB. Defaults to False.
            do_convert_grayscale (bool, optional): Whether to convert images to grayscale. Defaults to False.
            device (DeviceRef, optional): The device to use for the image processor. Defaults to DeviceRef.GPU().
            dtype (DType, optional): The data type to use for the image processor. Defaults to DType.bfloat16.

        Raises:
            ValueError: If both do_convert_rgb and do_convert_grayscale are set to True.
        """
        super().__init__()
        if do_convert_rgb and do_convert_grayscale:
            raise ValueError(
                "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
                " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
                " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
            )

        self.do_normalize = do_normalize
        self.device = device
        self.dtype = dtype
        self._denormalize_conditionally = max_compile(
            self._denormalize_conditionally,
            input_types=self._denormalize_conditionally_input_types(),
        )

    @staticmethod
    def denormalize(images: np.ndarray | Tensor) -> np.ndarray | Tensor:
        r"""Denormalize an image array to [0,1].

        Args:
            images (`np.ndarray` or `Tensor`):
                The image array to denormalize.

        Returns:
            `np.ndarray` or `Tensor`:
                The denormalized image array.
        """
        if isinstance(images, (Tensor, TensorValue)):
            images = images * 0.5 + 0.5
            images = F.min(
                images,
                Tensor.constant(1.0, dtype=images.dtype, device=images.device),
            )
            images = F.max(
                images,
                Tensor.constant(0.0, dtype=images.dtype, device=images.device),
            )
            return images
        return np.clip(images * 0.5 + 0.5, 0, 1)

    def _denormalize_conditionally(
        self,
        images: np.ndarray | Tensor,
    ) -> np.ndarray:
        r"""Denormalize a batch of images based on a condition list.

        Args:
            images (`np.ndarray` or `Tensor`):
                The input image tensor.
        """
        images = self.denormalize(images) if self.do_normalize else images
        images = ops.cast(images, DType.float32)
        return images

    @staticmethod
    def max_to_numpy(images: Tensor) -> np.ndarray:
        r"""Convert a Max tensor to a NumPy image.

        Args:
            images (`Tensor`):
                The Max tensor to convert to NumPy format.

        Returns:
            `np.ndarray`:
                A NumPy array representation of the images.
        """
        images = DTensor.to_numpy(images)
        images = np.transpose(images, (0, 2, 3, 1))
        return images

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> list[PIL.Image.Image]:
        r"""Convert a numpy image or a batch of images to a PIL image.

        Args:
            images (`np.ndarray`):
                The image array to convert to PIL format.

        Returns:
            `list[PIL.Image.Image]`:
                A list of PIL images.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L") for image in images
            ]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def _denormalize_conditionally_input_types(self) -> list[TensorType]:
        return [
            TensorType(
                shape=("batch_size", "num_channels", "height", "width"),
                device=self.device,
                dtype=self.dtype,
            ),
        ]

    def postprocess(
        self,
        image: Tensor,
        output_type: str = "pil",
        do_denormalize: list[bool] | None = None,
    ) -> PIL.Image.Image | np.ndarray | Tensor:
        """Postprocess the image output from tensor to `output_type`.

        Args:
            image (`Tensor`):
                The image input, should be a Max tensor with shape `B x C x H x W`.
            output_type (`str`, *optional*, defaults to `pil`):
                The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.
            do_denormalize (`list[bool]`, *optional*, defaults to `None`):
                Whether to denormalize the image to [0,1]. If `None`, will use the value of `do_normalize` in the
                `VaeImageProcessor` config.

        Returns:
            `PIL.Image.Image`, `np.ndarray` or `Tensor`:
                The postprocessed image.
        """
        if not isinstance(image, Tensor) and not isinstance(image, TensorValue):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support Max tensor"
            )
        if output_type not in ["latent", "max", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `max`, `latent`"
            )
            logger.warning(deprecation_message)
            output_type = "np"

        if output_type == "latent":
            return image

        image = self._denormalize_conditionally(image)

        if output_type == "max":
            return image[0]

        image = self.max_to_numpy(image[0])

        if output_type == "np":
            return image

        if output_type == "pil":
            return self.numpy_to_pil(image)
