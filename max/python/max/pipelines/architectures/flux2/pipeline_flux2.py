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

from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal

import numpy as np
import PIL.Image
from max import functional as F
from max import random
from max.dtype import DType
from max.graph import DeviceRef
from max.interfaces import PixelGenerationOutput, TokenBuffer
from max.pipelines import PixelContext
from max.pipelines.lib.diffusion_schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from max.pipelines.lib.image_processor import VaeImageProcessor
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    PixelModelInputs,
)
from max.tensor import Tensor
from tqdm import tqdm

from ..autoencoders import AutoencoderKLFlux2Model
from ..mistral3 import Mistral3TextEncoderModel
from ..mistral3.tokenizer import Mistral3Tokenizer
from .model import Flux2Model
from .system_messages import SYSTEM_MESSAGE


def format_input(
    prompts: list[str],
    system_message: str = SYSTEM_MESSAGE,
    images: list[PIL.Image.Image] | list[list[PIL.Image.Image]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Format a batch of text prompts into the conversation format expected by apply_chat_template.

    Optionally, add images to the input.

    Adapted from:
    https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/text_encoder.py#L68

    Args:
        prompts: List of text prompts.
        system_message: System message to use (default: SYSTEM_MESSAGE).
        images: Optional list of images to add to the input.

    Returns:
        List of conversations, where each conversation is a list of message dicts.
    """
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    if images is None or len(images) == 0:
        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]
    else:
        assert len(images) == len(
            prompts
        ), "Number of images must match number of prompts"
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
            ]
            for _ in cleaned_txt
        ]

        for i, (el, img_list) in enumerate(zip(messages, images)):
            if img_list is not None:
                el.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_obj}
                            for image_obj in img_list
                        ],
                    }
                )
            el.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": cleaned_txt[i]}],
                }
            )

        return messages


@dataclass(kw_only=True)
class Flux2ModelInputs(PixelModelInputs):
    """
    Flux2-specific PixelModelInputs.

    Defaults:
    - width: 1024
    - height: 1024
    - guidance_scale: 4.0
    - num_inference_steps: 50
    - num_images_per_prompt: 1
    - input_image: None (optional input image for image-to-image generation)

    """

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    input_image: Any | None = None
    """Optional input image for image-to-image generation (PIL.Image.Image).
    
    This field is used for Flux2 image-to-image generation where an input image
    is provided as a condition for the generation process.
    """


@dataclass
class Flux2PipelineOutput:
    """Output class for Flux2 image generation pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray` or `Tensor`)
            List of denoised PIL images of length `batch_size` or numpy array or Max tensor of shape `(batch_size,
            height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion
            pipeline. Max tensors can represent either the denoised images or the intermediate latents ready to be
            passed to the decoder.
    """

    images: list[PIL.Image.Image] | np.ndarray | Tensor


class Flux2Pipeline(DiffusionPipeline):
    config_name = "model_index.json"

    components = {
        "scheduler": FlowMatchEulerDiscreteScheduler,
        "vae": AutoencoderKLFlux2Model,
        "text_encoder": Mistral3TextEncoderModel,
        "tokenizer": Mistral3Tokenizer,
        "transformer": Flux2Model,
    }

    def init_remaining_components(self) -> None:
        if not getattr(self, "scheduler", None):
            self.scheduler = FlowMatchEulerDiscreteScheduler()
        image_processor_class = self.components.get(
            "image_processor", VaeImageProcessor
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        image_processor = image_processor_class(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.image_processor = image_processor

    def prepare_inputs(self, context: PixelContext) -> Flux2ModelInputs:
        return Flux2ModelInputs.from_context(context)

    def _pil_image_to_tensor(
        self,
        image: PIL.Image.Image,
    ) -> Tensor:

        img_array = (np.array(image, dtype=np.float32) / 127.5) - 1.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.ascontiguousarray(img_array)
        img_tensor = Tensor.from_dlpack(img_array).to(self.vae.devices[0]).cast(self.vae.config.dtype)

        return img_tensor

    def _get_mistral_3_small_prompt_embeds(
        self,
        text_input_ids: np.ndarray,
        hidden_states_layers: list[int] | None = None,
        max_sequence_length: int | None = None,
    ) -> Tensor:

        if hidden_states_layers is None:
            hidden_states_layers = [10, 20, 30]
        
        if max_sequence_length is None:
            max_sequence_length = text_input_ids.shape[-1]
        
        if text_input_ids.ndim == 1:
            text_input_ids = np.expand_dims(text_input_ids, axis=0)
        
        text_input_ids = text_input_ids.astype(np.int64)
        hidden_states_tuple = self.text_encoder(text_input_ids)
        
        if not isinstance(hidden_states_tuple, tuple):
            raise ValueError(
                f"Expected tuple of hidden states, got {type(hidden_states_tuple)}"
            )
        
        layer_tensors = []
        for k in hidden_states_layers:
            layer_idx = k - 1
            if layer_idx >= len(hidden_states_tuple):
                raise ValueError(
                    f"Layer index {k} (0-based: {layer_idx}) is out of range. "
                    f"Total layers: {len(hidden_states_tuple)}"
                )

            hs = hidden_states_tuple[layer_idx]

            if not isinstance(hs, Tensor):
                hs = Tensor.from_dlpack(hs)

            if hs.rank == 2:
                real_seq_len = hs.shape[0].dim
                hidden_dim = hs.shape[1].dim
                if real_seq_len < max_sequence_length:
                    padding_size = max_sequence_length - real_seq_len
                    padding = Tensor.zeros(
                        [padding_size, hidden_dim],
                        dtype=hs.dtype,
                        device=hs.device,
                    )
                    hs = F.concat([hs, padding], axis=0)
                elif real_seq_len > max_sequence_length:
                    hs = hs[:max_sequence_length]

                hs = F.reshape(hs, [1, max_sequence_length, hidden_dim])
            elif hs.rank == 3:
                batch_size = hs.shape[0].dim
                real_seq_len = hs.shape[1].dim
                hidden_dim = hs.shape[2].dim
                if real_seq_len < max_sequence_length:
                    padding_size = max_sequence_length - real_seq_len
                    padding = Tensor.zeros(
                        [batch_size, padding_size, hidden_dim],
                        dtype=hs.dtype,
                        device=hs.device,
                    )
                    hs = F.concat([hs, padding], axis=1)
                elif real_seq_len > max_sequence_length:
                    hs = hs[:, :max_sequence_length, :]

            layer_tensors.append(hs)

        stacked = F.stack(layer_tensors, axis=1)
        stacked = F.permute(stacked, [0, 2, 1, 3])

        batch_size = stacked.shape[0].dim
        seq_len = stacked.shape[1].dim
        num_layers = stacked.shape[2].dim
        hidden_dim = stacked.shape[3].dim
        prompt_embeds = F.reshape(
            stacked, [batch_size, seq_len, num_layers * hidden_dim]
        )
        
        return prompt_embeds

    def _prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
    ) -> tuple[Tensor, Tensor]:
        text_input_ids = tokens.array
        if text_input_ids.ndim == 1:
            text_input_ids = np.expand_dims(text_input_ids, axis=0)
        text_input_ids = text_input_ids.astype(np.int64)

        max_sequence_length = text_input_ids.shape[-1]

        prompt_embeds = self._get_mistral_3_small_prompt_embeds(
            text_input_ids=text_input_ids,
            hidden_states_layers=None,
            max_sequence_length=max_sequence_length,
        )

        bs_embed, seq_len, _ = prompt_embeds.shape

        prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.reshape(
            (bs_embed.dim * num_images_per_prompt, seq_len, -1)
        )

        batch_size = bs_embed.dim * num_images_per_prompt
        text_seq_len = seq_len.dim if hasattr(seq_len, "dim") else seq_len
        text_ids = self._prepare_text_ids(
            batch_size=batch_size,
            seq_len=text_seq_len,
            device=self.text_encoder.devices[0],
        )

        return prompt_embeds, text_ids

    def _prepare_latents_for_denoising(
        self,
        model_inputs: Flux2ModelInputs,
        dtype: DType,
    ) -> tuple[Tensor, Tensor, Tensor]:
        latents: Tensor = (
            Tensor.from_dlpack(model_inputs.latents)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )

        latents = self._patchify_latents(latents)
        latents = self._pack_latents(latents)

        image_seq_len = latents.shape[1].dim
        patch_h = patch_w = int(image_seq_len**0.5)

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size=latents.shape[0].dim,
            height=patch_h,
            width=patch_w,
            device=self.transformer.devices[0],
            dtype=dtype,
        )

        guidance = Tensor.full(
            [latents.shape[0]],
            model_inputs.guidance_scale,
            device=self.transformer.devices[0],
            dtype=dtype,
        )

        return latents, latent_image_ids, guidance

    def _denoise_latents(
        self,
        latents: Tensor,
        prompt_embeds: Tensor,
        text_ids: Tensor,
        guidance: Tensor,
        timestep: Tensor,
        latent_image_ids: Tensor,
        compiled_model: Any,
        random_latents_seq_len: int,
    ) -> Tensor:

        latent_model_input = latents.cast(prompt_embeds.dtype)
        
        hidden_states_drv = latent_model_input.driver_tensor
        encoder_hidden_states_drv = prompt_embeds.driver_tensor
        timestep_drv = timestep.driver_tensor
        img_ids_drv = latent_image_ids.driver_tensor
        txt_ids_drv = text_ids.driver_tensor
        guidance_drv = guidance.driver_tensor

        noise_pred_drv = compiled_model(
            hidden_states_drv,
            encoder_hidden_states_drv,
            timestep_drv,
            img_ids_drv,
            txt_ids_drv,
            guidance_drv,
        )[0]

        noise_pred = Tensor.from_dlpack(noise_pred_drv)
        noise_pred = noise_pred[:, :random_latents_seq_len, :]

        return noise_pred

    def _decode_latents(
        self,
        latents: Tensor,
        latent_image_ids: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:

        if output_type == "latent":
            return latents

        patchified_height = height // (self.vae_scale_factor * 2)
        patchified_width = width // (self.vae_scale_factor * 2)

        latents_unpacked = self._unpack_latents_with_ids(
            latents, latent_image_ids, patchified_height, patchified_width
        )

        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

        num_channels = bn_mean.shape[0].dim
        bn_mean = F.reshape(bn_mean, (1, num_channels, 1, 1))
        bn_var = F.reshape(bn_var, (1, num_channels, 1, 1))
        bn_std = F.sqrt(bn_var + self.vae.config.batch_norm_eps)

        latents_unpacked = latents_unpacked * bn_std + bn_mean
        latents_unpacked = self._unpatchify_latents(latents_unpacked)

        image = self.vae.decode(latents_unpacked.driver_tensor)

        if not isinstance(image, Tensor):
            image = Tensor.from_dlpack(image)

        if output_type in ["np", "pil"]:
            image = self.image_processor.postprocess(image, output_type=output_type)

        return image

    @staticmethod
    def _prepare_text_ids(
        batch_size: int,
        seq_len: int,
        device: DeviceRef,
    ) -> Tensor:
        coords = np.stack(
            [
                np.zeros(seq_len, dtype=np.int64),
                np.zeros(seq_len, dtype=np.int64),
                np.zeros(seq_len, dtype=np.int64),
                np.arange(seq_len, dtype=np.int64),
            ],
            axis=-1,
        )

        text_ids = np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        text_ids = Tensor.from_dlpack(text_ids).to(device)
        return text_ids

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int,
        height: int,
        width: int,
        device: DeviceRef,
        dtype: DType,
    ) -> Tensor:
        t_coords, h_coords, w_coords, l_coords = np.meshgrid(
            np.array([0]),
            np.arange(height),
            np.arange(width),
            np.array([0]),
            indexing="ij",
        )

        latent_ids = np.stack([t_coords, h_coords, w_coords, l_coords], axis=-1)
        latent_ids = latent_ids.reshape(-1, 4)
        latent_ids = np.tile(latent_ids[np.newaxis, :, :], (batch_size, 1, 1))

        latent_image_ids = Tensor.from_dlpack(latent_ids.astype(np.int64)).to(
            device
        )

        return latent_image_ids

    @staticmethod
    def _prepare_image_ids(
        image_latents: list[Tensor],
        scale: int = 10,
        device: DeviceRef | None = None,
    ) -> Tensor:
        if not isinstance(image_latents, list):
            raise ValueError(
                f"Expected `image_latents` to be a list, got {type(image_latents)}."
            )

        if len(image_latents) == 0:
            raise ValueError("Expected at least one image latent in the list.")

        if device is None:
            device = image_latents[0].device

        image_latent_ids = []

        for i, latent in enumerate(image_latents):
            latent_squeezed = F.squeeze(latent, axis=0)
            _, height, width = latent_squeezed.shape
            height = height.dim
            width = width.dim

            t_coord = scale + scale * i
            t_coords = np.full((height, width), t_coord, dtype=np.int64)
            h_coords, w_coords = np.meshgrid(
                np.arange(height, dtype=np.int64),
                np.arange(width, dtype=np.int64),
                indexing="ij",
            )
            l_coords = np.zeros((height, width), dtype=np.int64)

            coords = np.stack([t_coords, h_coords, w_coords, l_coords], axis=-1)
            coords = coords.reshape(-1, 4)
            coords_tensor = Tensor.from_dlpack(coords).to(device)
            image_latent_ids.append(coords_tensor)

        image_latent_ids = F.concat(image_latent_ids, axis=0)
        image_latent_ids = F.unsqueeze(image_latent_ids, 0)

        return image_latent_ids

    @staticmethod
    def _pack_latents(latents: Tensor) -> Tensor:
        batch_size = latents.shape[0].dim
        num_channels = latents.shape[1].dim
        height = latents.shape[2].dim
        width = latents.shape[3].dim
        latents = F.reshape(latents, (batch_size, num_channels, height * width))
        latents = F.permute(latents, (0, 2, 1))
        return latents

    @staticmethod
    def _unpack_latents_with_ids(
        x: Tensor, x_ids: Tensor, height: int | None = None, width: int | None = None
    ) -> Tensor:
        batch_size = x.shape[0].dim
        seq_len = x.shape[1].dim
        ch = x.shape[2].dim

        h_ids = x_ids[:, :, 1].cast(DType.int64)
        w_ids = x_ids[:, :, 2].cast(DType.int64)

        if height is None or width is None:
            h = int(h_ids.max().item()) + 1
            w = int(w_ids.max().item()) + 1
        else:
            h = height
            w = width

        flat_ids = h_ids * w + w_ids

        x_list = []
        for b in range(batch_size):
            data_b = x[b]
            flat_ids_b = flat_ids[b]

            out = Tensor.zeros([h * w, ch], dtype=x.dtype, device=x.device)
            indices = F.reshape(flat_ids_b, [seq_len, 1]).cast(DType.int64)
            out = F.scatter_nd(out, data_b, indices)

            out = F.reshape(out, [h, w, ch])
            out = F.permute(out, (2, 0, 1))
            x_list.append(out)

        result = F.stack(x_list, axis=0)
        return result

    @staticmethod
    def retrieve_latents(
        encoder_output: "DiagonalGaussianDistribution",
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> Tensor:
        if hasattr(encoder_output, "mode") and sample_mode == "mode":
            return encoder_output.mode()
        elif hasattr(encoder_output, "sample") and sample_mode == "sample":
            return encoder_output.sample(generator=generator)
        else:
            raise AttributeError(
                f"Could not access latents from encoder_output. "
                f"Expected DiagonalGaussianDistribution with 'mode' or 'sample' method, "
                f"got {type(encoder_output)}"
            )

    def _encode_vae_image(
        self,
        image: Tensor,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> Tensor:
        if len(image.shape) != 4:
            raise ValueError(f"Expected image dims 4, got {len(image.shape)}.")

        encoder_output = self.vae.encode(image, return_dict=True)

        if isinstance(encoder_output, dict):
            encoder_output = encoder_output["latent_dist"]

        image_latents = self.retrieve_latents(
            encoder_output, generator=generator, sample_mode=sample_mode
        )
        image_latents = self._patchify_latents(image_latents)

        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

        num_channels = bn_mean.shape[0].dim
        bn_mean = F.reshape(bn_mean, (1, num_channels, 1, 1))
        bn_var = F.reshape(bn_var, (1, num_channels, 1, 1))
        bn_std = F.sqrt(bn_var + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - bn_mean) / bn_std

        return image_latents

    @staticmethod
    def _patchify_latents(latents: Tensor) -> Tensor:
        batch_size = latents.shape[0].dim
        num_channels_latents = latents.shape[1].dim
        height = latents.shape[2].dim
        width = latents.shape[3].dim

        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents, height // 2, 2, width // 2, 2),
        )
        latents = F.permute(latents, (0, 1, 3, 5, 2, 4))
        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents * 4, height // 2, width // 2),
        )
        return latents

    @staticmethod
    def _unpatchify_latents(latents: Tensor) -> Tensor:
        batch_size = latents.shape[0].dim
        num_channels_latents = latents.shape[1].dim
        height = latents.shape[2].dim
        width = latents.shape[3].dim

        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents // 4, 2, 2, height, width),
        )
        latents = F.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents // 4, height * 2, width * 2),
        )
        return latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: DType,
        device: DeviceRef,
        latents: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents * 4, height // 2, width // 2)

        if latents is not None:
            latents = (
                latents
                if isinstance(latents, Tensor)
                else Tensor.from_dlpack(latents)
            )
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device).cast(dtype), latent_image_ids

        latents = random.normal(shape, device=device, dtype=dtype)

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        latents = self._pack_latents(latents)

        return latents, latent_image_ids

    def prepare_image_latents(
        self,
        images: list[Tensor],
        batch_size: int,
        device: DeviceRef,
        dtype: DType,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> tuple[Tensor, Tensor]:
        image_latents = []
        for image in images:
            image = image.to(device).cast(dtype)
            latent = self._encode_vae_image(
                image=image, generator=generator, sample_mode=sample_mode
            )
            image_latents.append(latent)

        image_latent_ids = self._prepare_image_ids(image_latents, device=device)

        packed_latents = []
        for latent in image_latents:
            packed = self._pack_latents(latent)
            packed = F.squeeze(packed, axis=0)
            packed_latents.append(packed)

        image_latents = F.concat(packed_latents, axis=0)
        image_latents = F.unsqueeze(image_latents, 0)
        image_latents = F.tile(image_latents, (batch_size, 1, 1))

        image_latent_ids = F.tile(image_latent_ids, (batch_size, 1, 1))
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    def _scheduler_step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        sigmas: Tensor,
        step_index: int,
    ) -> Tensor:
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        latents = latents + dt * noise_pred
        latents = latents.cast(latents_dtype)
        return latents

    def execute(
        self,
        model_inputs: Flux2ModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Flux2PipelineOutput:
        """Execute the pipeline."""
        # 1. Encode prompts
        prompt_embeds, text_ids = self._prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )

        image_latents = None
        image_latent_ids = None
        if model_inputs.input_image is not None:
            image_tensor = self._pil_image_to_tensor(model_inputs.input_image)
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=[image_tensor],
                batch_size=prompt_embeds.shape[0].dim,
                device=self.vae.devices[0],
                dtype=self.vae.config.dtype,
            )

        # 2. Prepare latents for denoising
        dtype = prompt_embeds.dtype
        batch_size = prompt_embeds.shape[0].dim

        timesteps: np.ndarray = model_inputs.timesteps
        sigmas_array: np.ndarray = model_inputs.sigmas
        
        latents, latent_image_ids, guidance = self._prepare_latents_for_denoising(
            model_inputs=model_inputs,
            dtype=dtype,
        )
        
        if len(sigmas_array) == len(timesteps):
            sigmas_array = np.append(sigmas_array, np.float32(0.0))
        
        sigmas = Tensor.from_dlpack(sigmas_array).to(
            self.transformer.devices[0]
        )
        
        num_timesteps = timesteps.shape[0]
        timesteps_batched = np.broadcast_to(
            timesteps[:, None], (num_timesteps, batch_size)
        )
        timesteps_batched = Tensor.from_dlpack(timesteps_batched).to(
            self.transformer.devices[0]
        ).cast(dtype)

        random_latents_seq_len = latents.shape[1].dim
        random_latent_ids = latent_image_ids
        
        if image_latents is not None:
            latents = F.concat([latents, image_latents], axis=1)
            latent_image_ids = F.concat([latent_image_ids, image_latent_ids], axis=1)
        
        image_seq_len = latents.shape[1].dim
        
        text_seq_len = prompt_embeds.shape[1].dim
        compiled_model = self.transformer._ensure_compiled(
            batch_size=batch_size,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
        )
        # 3. Denoising
        for i in tqdm(range(num_timesteps), desc="Denoising"):
            self._current_timestep = i
            timestep = timesteps_batched[i]

            noise_pred = self._denoise_latents(
                latents=latents,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                guidance=guidance,
                timestep=timestep,
                latent_image_ids=latent_image_ids,
                compiled_model=compiled_model,
                random_latents_seq_len=random_latents_seq_len,
            )

            if image_latents is not None:
                random_latents = latents[:, :random_latents_seq_len, :]
            else:
                random_latents = latents

            random_latents = self._scheduler_step(random_latents, noise_pred, sigmas, i)

            if image_latents is not None:
                latents = F.concat([random_latents, image_latents], axis=1)
            else:
                latents = random_latents

            if callback_queue is not None:
                callback_latents = random_latents
                callback_latent_ids = random_latent_ids
                image = self._decode_latents(
                    callback_latents,
                    callback_latent_ids,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )
                callback_queue.put_nowait(image)

        # 4. Decode
        if image_latents is not None:
            decode_latents = latents[:, :random_latents_seq_len, :]
            decode_latent_ids = random_latent_ids
        else:
            decode_latents = latents
            decode_latent_ids = latent_image_ids
        
        outputs = self._decode_latents(
            decode_latents,
            decode_latent_ids,
            model_inputs.height,
            model_inputs.width,
            output_type=output_type,
        )

        return Flux2PipelineOutput(images=outputs)
