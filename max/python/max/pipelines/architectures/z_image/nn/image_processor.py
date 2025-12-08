from typing import List
from PIL import Image
from max.experimental.tensor import Tensor
from max.driver import CPU
import numpy as np


class VaeImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def denormalize(images: Tensor) -> Tensor:
        r"""
        Denormalize an image array to [0,1].

        Args:
            images (`Tensor`):
                The image array to denormalize.

        Returns:
            `Tensor`:
                The denormalized image array.
        """
        return (images * 0.5 + 0.5).clip(min=0.0, max=1.0)

    def tensor_to_pil(self, images: Tensor) -> List[Image.Image]:
        """
        Convert a batch of tensors to a list of PIL Images.

        Args:
            images (`Tensor`):
                The image tensor with shape `B x C x H x W`.

        Returns:
            `List[Image.Image]`:
                The list of PIL images.
        """
        # Ensure tensor is on CPU for conversion
        images = images.to(CPU())

        # Convert to numpy (this is the bridge to PIL)
        # We expect B x C x H x W, need to convert to B x H x W x C for PIL
        # Convert to numpy first to avoid potential symbolic tensor issues with permute
        images_np = np.from_dlpack(images)
        images_np = images_np.transpose(0, 2, 3, 1)

        # Convert to uint8 [0, 255]
        images_np = (images_np * 255).round().astype("uint8")

        if images_np.ndim == 3:
            images_np = images_np[None, ...]

        pil_images = [Image.fromarray(image) for image in images_np]
        return pil_images

    def postprocess(
        self,
        image: Tensor,
        output_type: str = "pil",
    ) -> List[Image.Image] | Tensor:
        """
        Postprocess the image output from tensor to `output_type`.

        Args:
            image (`Tensor`):
                The image input, should be a tensor with shape `B x C x H x W`.
            output_type (`str`, *optional*, defaults to `pil`):
                The output type of the image, can be one of `pil`, `latent`.

        Returns:
            `List[Image.Image]` or `Tensor`:
                The postprocessed image.
        """

        if output_type == "latent":
            return image

        image = self.denormalize(image)

        return self.tensor_to_pil(image)
