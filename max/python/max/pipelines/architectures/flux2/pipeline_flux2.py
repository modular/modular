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
            # optionally add the images per batch element.
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
            # add the text.
            el.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": cleaned_txt[i]}],
                }
            )

        return messages


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for Flux2 timestep scheduling.

    Taken from:
    https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/sampling.py#L251

    Args:
        image_seq_len: Length of image sequence (H*W after packing).
        num_steps: Number of inference steps.

    Returns:
        Empirical mu value for scheduler.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


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

    """

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1


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
        # Scheduler is not a ComponentModel (no weights), so it is not loaded in _load_sub_models; create it here.
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

    def _prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
        hidden_states_layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if hidden_states_layers is None:
            hidden_states_layers = [10, 20, 30]

        # unsqueeze
        if tokens.array.ndim == 1:
            tokens.array = np.expand_dims(tokens.array, axis=0)

        # Convert to numpy array (not Tensor) for text_encoder
        # Mistral3TextEncoderModel expects numpy array, not Tensor
        text_input_ids = tokens.array.astype(np.int64)

        # Encode with Mistral3 text encoder
        # Mistral3TextEncoderModel returns tuple of hidden states (all layers)
        hidden_states_tuple = self.text_encoder(text_input_ids)

        if not isinstance(hidden_states_tuple, tuple):
            raise ValueError(
                f"Expected tuple of hidden states, got {type(hidden_states_tuple)}"
            )

        # Extract specific layers (10, 20, 30) and stack them
        # Note: hidden_states_tuple is 0-indexed, but hidden_states_layers is 1-indexed
        layer_tensors = []
        max_sequence_length = tokens.array.shape[-1]
        for k in hidden_states_layers:
            layer_idx = k - 1  # Convert 1-based to 0-based
            if layer_idx >= len(hidden_states_tuple):
                raise ValueError(
                    f"Layer index {k} (0-based: {layer_idx}) is out of range. "
                    f"Total layers: {len(hidden_states_tuple)}"
                )

            hs = hidden_states_tuple[layer_idx]

            # Convert to Tensor if needed
            if not isinstance(hs, Tensor):
                hs = Tensor.from_dlpack(hs)

            # Handle sequence length padding/truncation
            if hs.rank == 2:
                # Shape: [seq_len, hidden_dim]
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

                # Reshape to [1, seq_len, hidden_dim]
                hs = F.reshape(hs, [1, max_sequence_length, hidden_dim])
            elif hs.rank == 3:
                # Shape: [batch, seq_len, hidden_dim]
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

        # Stack layers: [1, 3, seq_len, hidden_dim]
        stacked = F.stack(layer_tensors, axis=1)

        # Permute to [1, seq_len, 3, hidden_dim]
        stacked = F.permute(stacked, [0, 2, 1, 3])

        # Reshape to [1, seq_len, 3*hidden_dim]
        batch_size = stacked.shape[0].dim
        seq_len = stacked.shape[1].dim
        num_layers = stacked.shape[2].dim
        hidden_dim = stacked.shape[3].dim
        prompt_embeds = F.reshape(
            stacked, [batch_size, seq_len, num_layers * hidden_dim]
        )

        bs_embed, seq_len, _ = prompt_embeds.shape

        # Tile for multiple images per prompt
        prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.reshape(
            (bs_embed.dim * num_images_per_prompt, seq_len, -1)
        )

        # Prepare text position IDs (4D for Flux2)
        batch_size_final = bs_embed.dim * num_images_per_prompt
        text_ids = self._prepare_text_ids(
            batch_size=batch_size_final,
            seq_len=seq_len.dim if hasattr(seq_len, "dim") else seq_len,
            device=self.text_encoder.devices[0],
        )

        return prompt_embeds, text_ids

    def _decode_latents(
        self,
        latents: Tensor,
        latent_image_ids: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> PIL.Image.Image | np.ndarray | Tensor:
        if output_type == "latent":
            # For latent output, return the first image in batch
            return latents[0] if latents.shape[0].dim > 1 else latents

        # Unpack latents using position IDs (Flux2 specific)
        latents_unpacked = self._unpack_latents_with_ids(latents, latent_image_ids)

        # Apply BatchNorm inverse transform (Flux2 specific)
        # Flux2 uses BatchNorm statistics instead of scaling_factor/shift_factor
        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

        num_channels = bn_mean.shape[0].dim
        bn_mean = F.reshape(bn_mean, (1, num_channels, 1, 1))
        bn_var = F.reshape(bn_var, (1, num_channels, 1, 1))
        bn_std = F.sqrt(bn_var + self.vae.config.batch_norm_eps)

        latents_unpacked = latents_unpacked * bn_std + bn_mean

        # Unpatchify latents: (B, C, H, W) -> (B, C//4, H*2, W*2)
        latents_unpacked = self._unpatchify_latents(latents_unpacked)

        # VAE decode
        image = self.vae.decode(latents_unpacked.driver_tensor)

        # Convert to Tensor if decode returns driver.Tensor
        if not isinstance(image, Tensor):
            image = Tensor.from_dlpack(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Match Flux1: pixel_generation expects each image as (H, W, C) so np.stack yields (N, H, W, C).
        if output_type == "np" and isinstance(image, np.ndarray):
            image = self._to_hwc(image)

        return image

    @staticmethod
    def _to_hwc(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        while img.ndim > 3:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.float32, copy=False)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: DeviceRef | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Tensor | None = None,
        max_sequence_length: int = 512,
        lora_scale: float | None = None,
        hidden_states_layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if hidden_states_layers is None:
            hidden_states_layers = [10, 20, 30]

        if lora_scale is not None and isinstance(self, Flux2Pipeline):
            self._lora_scale = lora_scale

            if self.text_encoder is not None and hasattr(
                self.text_encoder, "set_lora_scale"
            ):
                self.text_encoder.set_lora_scale(lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            # Format prompt using Flux2 chat template
            messages_batch = format_input(
                prompts=prompt, system_message=SYSTEM_MESSAGE
            )

            # Use HuggingFace tokenizer's apply_chat_template
            # Access the delegate tokenizer from Mistral3Tokenizer
            hf_tokenizer = self.tokenizer.delegate
            inputs = hf_tokenizer.apply_chat_template(
                messages_batch,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
                return_length=False,
                return_overflowing_tokens=False,
            )

            # Extract real tokens only (using attention mask)
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs.get("attention_mask", None)
            attention_mask = (
                attention_mask[0]
                if attention_mask is not None
                else [1] * len(input_ids)
            )

            # Filter to keep only real tokens (where mask == 1)
            real_token_ids = [
                token_id
                for token_id, mask in zip(input_ids, attention_mask)
                if mask == 1
            ]
            text_input_ids = np.array([real_token_ids], dtype=np.int64)

            # Encode with Mistral3 text encoder
            # Mistral3TextEncoderModel returns tuple of hidden states (all layers)
            hidden_states_tuple = self.text_encoder(text_input_ids)

            if not isinstance(hidden_states_tuple, tuple):
                raise ValueError(
                    f"Expected tuple of hidden states, got {type(hidden_states_tuple)}"
                )

            # Extract specific layers (10, 20, 30) and stack them
            # Note: hidden_states_tuple is 0-indexed, but hidden_states_layers is 1-indexed
            layer_tensors = []
            for k in hidden_states_layers:
                layer_idx = k - 1  # Convert 1-based to 0-based
                if layer_idx >= len(hidden_states_tuple):
                    raise ValueError(
                        f"Layer index {k} (0-based: {layer_idx}) is out of range. "
                        f"Total layers: {len(hidden_states_tuple)}"
                    )

                hs = hidden_states_tuple[layer_idx]

                # Convert to Tensor if needed
                # model_outputs.hidden_states returns tuple of TensorValue (V2 Tensor)
                # Following max-diffusers pattern: Tensor.from_dlpack(hs)
                if not isinstance(hs, Tensor):
                    # TensorValue (V2 Tensor) - convert to Tensor
                    # TensorValue can be directly converted using from_dlpack
                    hs = Tensor.from_dlpack(hs)

                # Handle sequence length padding/truncation
                if hs.rank == 2:
                    # Shape: [seq_len, hidden_dim]
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

                    # Reshape to [1, seq_len, hidden_dim]
                    hs = F.reshape(hs, [1, max_sequence_length, hidden_dim])
                elif hs.rank == 3:
                    # Shape: [batch, seq_len, hidden_dim]
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

            # Stack layers: [1, 3, seq_len, hidden_dim]
            stacked = F.stack(layer_tensors, axis=1)

            # Permute to [1, seq_len, 3, hidden_dim]
            stacked = F.permute(stacked, [0, 2, 1, 3])

            # Reshape to [1, seq_len, 3*hidden_dim]
            batch_size = stacked.shape[0].dim
            seq_len = stacked.shape[1].dim
            num_layers = stacked.shape[2].dim
            hidden_dim = stacked.shape[3].dim
            prompt_embeds = F.reshape(
                stacked, [batch_size, seq_len, num_layers * hidden_dim]
            )

            # Ensure correct device and dtype
            prompt_embeds = prompt_embeds.to(device).cast(
                prompt_embeds.dtype if hasattr(prompt_embeds, "dtype") else DType.bfloat16
            )

        bs_embed, seq_len, _ = prompt_embeds.shape

        # Tile for multiple images per prompt
        prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.reshape(
            (bs_embed.dim * num_images_per_prompt, seq_len, -1)
        )

        # Prepare text position IDs (4D for Flux2)
        # Flux2 uses 4D position IDs: [batch_size, seq_len, 4]
        # Following max-diffusers pattern: (T=0, H=0, W=0, L=[0..seq_len-1])
        batch_size_final = bs_embed.dim * num_images_per_prompt
        text_ids = self._prepare_text_ids(
            batch_size=batch_size_final,
            seq_len=seq_len.dim if hasattr(seq_len, 'dim') else seq_len,
            device=device,
        )

        return prompt_embeds, text_ids

    @staticmethod
    def _prepare_text_ids(
        batch_size: int,
        seq_len: int,
        device: DeviceRef,
    ) -> Tensor:
        # Create 4D coordinates: (T=0, H=0, W=0, L=[0..seq_len-1])
        coords = np.stack(
            [
                np.zeros(seq_len, dtype=np.int64),  # T dimension
                np.zeros(seq_len, dtype=np.int64),  # H dimension
                np.zeros(seq_len, dtype=np.int64),  # W dimension
                np.arange(seq_len, dtype=np.int64),  # L dimension
            ],
            axis=-1,
        )  # (seq_len, 4)

        # Expand to batch (batch_size, seq_len, 4)
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
        # Create 4D coordinates using numpy (T=0, H, W, L=0)
        # Following max-diffusers pattern
        t_coords, h_coords, w_coords, l_coords = np.meshgrid(
            np.array([0]),  # T dimension
            np.arange(height),  # H dimension
            np.arange(width),  # W dimension
            np.array([0]),  # L dimension
            indexing="ij",
        )

        # Stack and reshape to (H*W, 4)
        latent_ids = np.stack([t_coords, h_coords, w_coords, l_coords], axis=-1)
        latent_ids = latent_ids.reshape(-1, 4)

        # Expand to batch: (batch_size, H*W, 4)
        latent_ids = np.tile(latent_ids[np.newaxis, :, :], (batch_size, 1, 1))

        # Convert to Tensor with int64 dtype
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

        # Get device from first latent if not provided
        if device is None:
            device = image_latents[0].device

        # Create time offset for each reference image
        # T-coordinate for i-th image: scale + scale * i
        image_latent_ids = []

        for i, latent in enumerate(image_latents):
            # Remove batch dimension: [1, C, H, W] -> [C, H, W]
            latent_squeezed = F.squeeze(latent, axis=0)
            _, height, width = latent_squeezed.shape
            height = height.dim
            width = width.dim

            # T-coordinate for this image
            t_coord = scale + scale * i

            # Create coordinates using numpy (similar to _prepare_latent_image_ids)
            t_coords = np.full((height, width), t_coord, dtype=np.int64)
            h_coords, w_coords = np.meshgrid(
                np.arange(height, dtype=np.int64),
                np.arange(width, dtype=np.int64),
                indexing="ij",
            )
            l_coords = np.zeros((height, width), dtype=np.int64)

            # Stack: (H, W, 4)
            coords = np.stack([t_coords, h_coords, w_coords, l_coords], axis=-1)

            # Reshape: (H*W, 4)
            coords = coords.reshape(-1, 4)

            # Convert to Tensor
            coords_tensor = Tensor.from_dlpack(coords).to(device)
            image_latent_ids.append(coords_tensor)

        # Concatenate all coordinates along the first dimension
        # Each tensor is (H*W, 4), so concatenating gives (N_total, 4)
        image_latent_ids = F.concat(image_latent_ids, axis=0)

        # Add batch dimension: (1, N_total, 4)
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
    def _unpack_latents_with_ids(x: Tensor, x_ids: Tensor) -> Tensor:
        batch_size = x.shape[0].dim
        seq_len = x.shape[1].dim
        ch = x.shape[2].dim

        # Get h_ids and w_ids from position tensor (columns 1 and 2)
        h_ids = x_ids[:, :, 1].cast(DType.int64)  # [B, seq_len]
        w_ids = x_ids[:, :, 2].cast(DType.int64)  # [B, seq_len]

        # Calculate H and W from max indices + 1
        h = int(h_ids.max().item()) + 1
        w = int(w_ids.max().item()) + 1

        flat_ids = h_ids * w + w_ids

        # Create output tensor and scatter data into place
        x_list = []
        for b in range(batch_size):
            data_b = x[b]  # [seq_len, C]
            flat_ids_b = flat_ids[b]  # [seq_len]

            # Initialize output with zeros
            out = Tensor.zeros([h * w, ch], dtype=x.dtype, device=x.device)

            # Scatter: out[flat_ids[i], :] = data[i, :] for each i
            indices = F.reshape(flat_ids_b, [seq_len, 1]).cast(DType.int64)
            out = F.scatter_nd(out, data_b, indices)

            # Reshape from (H * W, C) to (C, H, W)
            out = F.reshape(out, [h, w, ch])
            out = F.permute(out, (2, 0, 1))  # [C, H, W]
            x_list.append(out)

        # Stack batches
        result = F.stack(x_list, axis=0)  # [B, C, H, W]
        return result

    @staticmethod
    def retrieve_latents(
        encoder_output: "DiagonalGaussianDistribution",
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> Tensor:
        # In Max, vae.encode() returns DiagonalGaussianDistribution directly
        # (unlike diffusers which wraps it in AutoencoderKLOutput)
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

        # Encode image using VAE
        encoder_output = self.vae.encode(image, return_dict=True)

        # Extract DiagonalGaussianDistribution from dict if needed
        if isinstance(encoder_output, dict):
            encoder_output = encoder_output["latent_dist"]

        # Retrieve latent from distribution
        image_latents = self.retrieve_latents(
            encoder_output, generator=generator, sample_mode=sample_mode
        )
        # Patchify latents: (1, C, H, W) -> (1, C*4, H//2, W//2)
        image_latents = self._patchify_latents(image_latents)
        # BatchNorm normalization
        # Get BatchNorm statistics (already Tensor)
        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

        # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
        num_channels = bn_mean.shape[0].dim
        bn_mean = F.reshape(bn_mean, (1, num_channels, 1, 1))
        bn_var = F.reshape(bn_var, (1, num_channels, 1, 1))

        # Calculate standard deviation: sqrt(var + eps)
        bn_std = F.sqrt(bn_var + self.vae.config.batch_norm_eps)

        # Normalize: (latent - mean) / std
        image_latents = (image_latents - bn_mean) / bn_std

        return image_latents

    @staticmethod
    def _patchify_latents(latents: Tensor) -> Tensor:
        batch_size = latents.shape[0].dim
        num_channels_latents = latents.shape[1].dim
        height = latents.shape[2].dim
        width = latents.shape[3].dim

        # Reshape: (B, C, H//2, 2, W//2, 2)
        # Split spatial dimensions into patch dimensions
        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents, height // 2, 2, width // 2, 2),
        )
        # Permute: (0, 1, 3, 5, 2, 4)
        # Rearrange: (B, C, H//2, 2, W//2, 2) -> (B, C, 2, 2, H//2, W//2)
        # This groups the 2x2 patch elements together
        latents = F.permute(latents, (0, 1, 3, 5, 2, 4))
        # Reshape: (B, C*4, H//2, W//2)
        # Flatten the 2x2 patch into channel dimension
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
        # Reverse the patchify permute: (0, 1, 3, 5, 2, 4) -> (0, 1, 4, 2, 5, 3)
        # From (B, C//4, 2, 2, H, W) to (B, C//4, H, 2, W, 2)
        latents = F.permute(latents, (0, 1, 4, 2, 5, 3))
        # Reshape to (B, C//4, H*2, W*2)
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
        # VAE applies 8x compression on images but we must also account for packing
        # which requires latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        # Flux2 latent shape: (B, C*4, H//2, W//2) before packing
        # After packing: (B, (H//2)*(W//2), C*4)
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

        # Prepare latent IDs before packing
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        # Pack latents: (B, C, H, W) -> (B, H*W, C)
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
            # Move to device and cast dtype
            image = image.to(device).cast(dtype)
            # Encode using VAE
            latent = self._encode_vae_image(
                image=image, generator=generator, sample_mode=sample_mode
            )
            image_latents.append(latent)  # (1, C*4, H_latent, W_latent)

        # Generate position IDs for all images
        image_latent_ids = self._prepare_image_ids(image_latents, device=device)

        # Pack each latent and concatenate
        packed_latents = []
        for latent in image_latents:
            # latent: (1, C*4, H_latent, W_latent)
            packed = self._pack_latents(latent)  # (1, H*W, C*4)
            # Remove batch dimension: (H*W, C*4)
            packed = F.squeeze(packed, axis=0)
            packed_latents.append(packed)

        # Concatenate all packed latents along sequence dimension
        # Each packed latent is (H*W, C*4), so concatenating gives (N_total, C*4)
        image_latents = F.concat(packed_latents, axis=0)  # (N_total, C*4)

        # Add batch dimension: (1, N_total, C*4)
        image_latents = F.unsqueeze(image_latents, 0)

        # Repeat for batch_size: (batch_size, N_total, C*4)
        # Using F.tile similar to prompt_embeds
        image_latents = F.tile(image_latents, (batch_size, 1, 1))

        # Repeat image_latent_ids for batch_size: (batch_size, N_total, 4)
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

        # 2. Denoise
        dtype = prompt_embeds.dtype
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

        image_seq_len = latents.shape[1].dim
        num_inference_steps = model_inputs.num_inference_steps
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)
        base_sigmas = np.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
            dtype=np.float32,
        )
        self.scheduler.set_timesteps(sigmas=base_sigmas, mu=mu)
        sigmas = (
            Tensor.from_dlpack(np.ascontiguousarray(self.scheduler.sigmas))
            .to(self.transformer.devices[0])
        )
        batch_size = prompt_embeds.shape[0].dim

        timesteps: np.ndarray = self.scheduler.timesteps
        num_timesteps = timesteps.shape[0]
        timesteps_normalized = (timesteps / 1000.0).astype(np.float32)
        timesteps_batched = np.broadcast_to(
            timesteps_normalized[:, None], (num_timesteps, batch_size)
        )
        timesteps_batched = Tensor.from_dlpack(timesteps_batched).to(
            self.transformer.devices[0]
        )

        image_seq_len = latents.shape[1].dim
        text_seq_len = prompt_embeds.shape[1].dim
        compiled_model = self.transformer._ensure_compiled(
            batch_size=batch_size,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
        )

        encoder_hidden_states_drv = prompt_embeds.driver_tensor
        guidance_drv = guidance.driver_tensor
        txt_ids_drv = text_ids.driver_tensor
        img_ids_drv = latent_image_ids.driver_tensor

        for i in tqdm(range(num_timesteps), desc="Denoising"):
            self._current_timestep = i
            t = timesteps[i]
            timestep_np = np.full((batch_size,), t, dtype=np.float32) / 1000.0
            timestep_tensor = (
                Tensor.from_dlpack(timestep_np)
                .to(prompt_embeds.device)
                .cast(prompt_embeds.dtype)
            )
            timestep_drv = timestep_tensor.driver_tensor

            hidden_states_drv = latents.driver_tensor

            noise_pred_drv = compiled_model(
                hidden_states_drv,
                encoder_hidden_states_drv,
                timestep_drv,
                img_ids_drv,
                txt_ids_drv,
                guidance_drv,
            )[0]

            noise_pred = Tensor.from_dlpack(noise_pred_drv)

            # scheduler step
            latents = self._scheduler_step(latents, noise_pred, sigmas, i)

            if callback_queue is not None:
                image = self._decode_latents(
                    latents,
                    latent_image_ids,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )
                callback_queue.put_nowait(image)

        # 3. Decode
        batch_size = latents.shape[0].dim
        image_list = []
        for b in range(batch_size):
            latents_b = latents[b : b + 1]
            latent_image_ids_b = latent_image_ids[b : b + 1]

            image_b = self._decode_latents(
                latents_b,
                latent_image_ids_b,
                model_inputs.height,
                model_inputs.width,
                output_type=output_type,
            )
            image_list.append(image_b)

        return Flux2PipelineOutput(images=image_list)
