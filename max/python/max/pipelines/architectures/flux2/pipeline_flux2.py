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

import logging
from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal

import numpy as np
from max import functional as F
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.interfaces import TokenBuffer
from max.pipelines import PixelContext
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from max.tensor import Tensor
from PIL import Image
from tqdm import tqdm

from ..autoencoders import AutoencoderKLFlux2Model
from ..mistral3.text_encoder import Mistral3TextEncoderModel
from .model import Flux2TransformerModel

logger = logging.getLogger("max.pipelines")


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


def _time_shift_exponential(
    mu: float, sigma_param: float, t: np.ndarray
) -> np.ndarray:
    """Resolution-dependent timestep shift (diffusers FlowMatchEulerDiscreteScheduler)."""
    out = np.exp(mu) / (np.exp(mu) + (1.0 / t - 1.0) ** sigma_param)
    return out.astype(np.float32)


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
    input_image: Image.Image | None = None
    """Optional input image for image-to-image generation (PIL.Image.Image).
    
    This field is used for Flux2 image-to-image generation where an input image
    is provided as a condition for the generation process.
    """


@dataclass
class Flux2PipelineOutput:
    """Container for Flux2 pipeline results.

    Attributes:
        images:
            Either a list of decoded PIL images, a NumPy array, or a MAX Tensor.
            When a Tensor is returned, it may represent decoded image data or
            intermediate latents depending on the selected output mode.
    """

    images: np.ndarray | Tensor


class Flux2Pipeline(DiffusionPipeline):
    """Diffusion pipeline for Flux2 image generation.

    This pipeline wires together:
        - Mistral3 text encoder
        - Flux2 transformer denoiser
        - Flux2 VAE (with BatchNorm-based latent normalization)
    """

    vae: AutoencoderKLFlux2Model
    text_encoder: Mistral3TextEncoderModel
    transformer: Flux2TransformerModel

    components = {
        "vae": AutoencoderKLFlux2Model,
        "text_encoder": Mistral3TextEncoderModel,
        "transformer": Flux2TransformerModel,
    }

    def init_remaining_components(self) -> None:
        """Initialize derived attributes that depend on loaded components."""
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self._scheduler_step_model: Model = self._build_scheduler_step_model()
        self._time_step_model: Model = self._build_all_time_steps_model()
        self._unsqueeze_model: Model = self._build_unsqueeze_model()
        self._concat_model: Model = self._build_concat_model()
        self._latent_prep_models: dict[str, Model] = {}
        self._vae_prep_models: dict[str, Model] = {}
        self._prompt_embed_models: dict[str, Model] = {}
        self._text_ids_models: dict[str, Model] = {}

        self._cached_guidance: dict[str, Tensor] = {}
        self._cached_text_ids: dict[str, Tensor] = {}
        self._cached_sigmas: dict[str, Tensor] = {}

    def _ensure_latent_prep_model(self, height, width, dtype) -> Model:
        key = f"{height}_{width}_{dtype}"
        if key in self._latent_prep_models:
            return self._latent_prep_models[key]

        device = self.devices[0]
        input_dtype = DType.float32
        channels = self.vae.config.latent_channels

        input_types = [
            TensorType(
                input_dtype, shape=["batch_size", channels, "height", "width"], device=device
            ),
        ]

        with Graph("latent_prep", input_types=input_types) as graph:
            latents = graph.inputs[0]
            batch_size = latents.shape[0]
            latents = ops.cast(latents, dtype)
            latents = ops.rebind(
                latents,
                (batch_size, channels, (height // 2) * 2, (width // 2) * 2),
            )
            latents = ops.reshape(
                latents, (batch_size, channels, height // 2, 2, width // 2, 2)
            )
            latents = ops.permute(latents, (0, 1, 3, 5, 2, 4))
            latents = ops.reshape(latents, (batch_size, channels * 4, height // 2 * width // 2))
            latents = ops.permute(latents, (0, 2, 1))
            graph.output(latents)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._latent_prep_models[key] = model
        return model

    def _ensure_vae_prep_model(self, height, width) -> Model:
        """Compile fused graph: unpack -> batch norm inverse -> unpatchify."""
        patch_size_num = (
            self.vae.config.patch_size[0] * self.vae.config.patch_size[1]
        )  # 4

        key = f"{height}_{width}"
        if key in self._vae_prep_models:
            return self._vae_prep_models[key]

        logger.debug(f"Compiling vae prep model for {height}x{width}")
        device = self.devices[0]
        dtype = self.vae.config.dtype
        channels = self.vae.config.latent_channels * patch_size_num

        input_types = [
            TensorType(
                dtype, shape=["batch_size", "seq_len", channels], device=device
            ),
            TensorType(dtype, shape=[channels], device=device),
            TensorType(dtype, shape=[channels], device=device),
        ]

        with Graph("vae_prep", input_types=input_types) as graph:
            latents, bn_mean, bn_var = graph.inputs
            batch_size = latents.shape[0]
            latents = ops.permute(latents, (0, 2, 1))
            latents = ops.rebind(
                latents, (batch_size, channels, height * width)
            )
            latents = ops.reshape(
                latents, (batch_size, channels, height, width)
            )

            mean_reshaped = ops.reshape(bn_mean, (1, channels, 1, 1))
            var_reshaped = ops.reshape(bn_var, (1, channels, 1, 1))
            eps = ops.constant(
                self.vae.config.batch_norm_eps, dtype=dtype, device=device
            )
            std = ops.sqrt(ops.add(var_reshaped, eps))
            latents = ops.add(ops.mul(latents, std), mean_reshaped)

            latents = ops.reshape(
                latents, (batch_size, channels // 4, 2, 2, height, width)
            )
            latents = ops.permute(latents, (0, 1, 4, 2, 5, 3))
            latents = ops.reshape(
                latents, (batch_size, channels // 4, height * 2, width * 2)
            )
            graph.output(latents)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._vae_prep_models[key] = model
        return model

    def _ensure_prompt_embed_model(
        self,
        num_layers: int,
    )->Model:
        device = self.devices[0]
        dtype = self.transformer.config.dtype
        key = f"{num_layers}"
        if key in self._prompt_embed_models:
            return self._prompt_embed_models[key]

        input_types = [
            TensorType(dtype, shape=["batch_size", "seq_len", "hidden_dim"], device=device)
            for _ in range(num_layers)
        ]
        with Graph("prompt_embed_postproc", input_types=input_types) as graph:
            layer_inputs = [inp for inp in graph.inputs]
            batch_size = layer_inputs[0].shape[0]
            seq_len = layer_inputs[0].shape[1]
            hidden_dim = layer_inputs[0].shape[2]

            stacked = ops.stack(layer_inputs, axis=1)
            stacked = ops.permute(stacked, [0, 2, 1, 3])
            result = ops.reshape(
                stacked, [batch_size, seq_len, num_layers * hidden_dim]
            )
            graph.output(result)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._prompt_embed_models[key] = model
        return model

    def _ensure_text_ids_model(
        self,
        batch_size: int,
        seq_len: int,
    )->Model:
        device = self.devices[0]
        model_key = f"{batch_size}_{seq_len}"
        if model_key in self._text_ids_models:
            return self._text_ids_models[model_key]

        with Graph("text_ids_gen", input_types=[]) as graph:
            coords = np.zeros((seq_len, 4), dtype=np.int64)
            coords[:, 3] = np.arange(seq_len, dtype=np.int64)
            coords_const = ops.constant(coords, dtype=DType.int64, device=device)
            coords_batch = ops.reshape(coords_const, (1, seq_len, 4))
            text_ids = ops.tile(coords_batch, [batch_size, 1, 1])
            graph.output(text_ids)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        self._text_ids_models[model_key] = model
        return model

    def _build_all_time_steps_model(self) -> Model:
        logger.debug("Compiling all time steps model")

        device = self.devices[0]
        dtype = self.transformer.config.dtype
        with Graph(
            "time_step_all",
            input_types=[TensorType(DType.float32, ["num_sigmas"], device)],
        ) as graph:
            sigmas = graph.inputs[0]
            sigmas_curr = ops.slice_tensor(sigmas, [slice(0, -1)])
            all_t = ops.cast(sigmas_curr, dtype)
            sigmas_next = ops.slice_tensor(sigmas, [slice(1, None)])
            dt_f32 = ops.sub(sigmas_next, sigmas_curr)
            all_dt = ops.cast(dt_f32, dtype)
            graph.output(all_t, all_dt)

        session = InferenceSession([Accelerator()])
        return session.load(graph)

    def _build_scheduler_step_model(self) -> Model:
        logger.debug("Compiling scheduler step model")

        device = self.devices[0]
        dtype = self.transformer.config.dtype
        input_types = [
            TensorType(
                dtype, shape=["batch", "seq", "channels"], device=device
            ),
            TensorType(
                dtype, shape=["batch", "pred_seq", "channels"], device=device
            ),
            TensorType(dtype, shape=[1], device=device),
            TensorType(DType.int64, shape=[1], device=DeviceRef.CPU()),
        ]
        with Graph("scheduler_step", input_types=input_types) as graph:
            latents_in, noise_pred_in, dt_in, num_noise_tokens_in = graph.inputs
            latents_sliced = ops.slice_tensor(
                latents_in,
                [slice(None), (slice(0, num_noise_tokens_in), "num_tokens"), slice(None)],
            )
            noise_pred_sliced = ops.slice_tensor(
                noise_pred_in,
                [slice(None), (slice(0, num_noise_tokens_in), "num_tokens"), slice(None)],
            )
            result = latents_sliced + dt_in * noise_pred_sliced
            graph.output(result)

        session = InferenceSession([Accelerator()])
        return session.load(graph)

    def _build_unsqueeze_model(self)-> Model:
        dtype = self.text_encoder.config.dtype
        device = self.devices[0]
        input_type = TensorType(dtype, shape=["seq_len", "hidden_dim"], device=device)
        with Graph("unsqueeze_layer", input_types=[input_type]) as graph:
            tensor_in = graph.inputs[0]
            seq_len = tensor_in.shape[0]
            hidden_dim = tensor_in.shape[1]
            result = ops.reshape(tensor_in, [1, seq_len, hidden_dim])
            graph.output(result)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        return model

    def _build_concat_model(self)-> Model:
        dtype = self.transformer.config.dtype
        device = self.devices[0]
        input_types = [
            TensorType(
                dtype,
                shape=["batch_size", "seq_len_latent", "latent_dim"],
                device=device,
            ),
            TensorType(
                dtype,
                shape=["batch_size", "seq_len_image", "latent_dim"],
                device=device,
            ),
            TensorType(
                DType.int64,
                shape=["batch_size", "seq_len_latent", 4],
                device=device,
            ),
            TensorType(
                DType.int64,
                shape=["batch_size", "seq_len_image", 4],
                device=device,
            ),
        ]
        with Graph("concat_layer", input_types=input_types) as graph:
            latents = graph.inputs[0]
            img_latents = graph.inputs[1]
            latent_img_ids = graph.inputs[2]
            image_latent_ids = graph.inputs[3]
            result_latents = ops.concat([latents, img_latents], axis=1)
            result_latent_ids = ops.concat([latent_img_ids, image_latent_ids], axis=1)
            graph.output(result_latents, result_latent_ids)

        session = InferenceSession([Accelerator()])
        model = session.load(graph)
        return model

    def prepare_inputs(self, context: PixelContext) -> Flux2ModelInputs:
        """Convert a PixelContext into Flux2ModelInputs."""
        if context.input_image is not None and isinstance(
            context.input_image, np.ndarray
        ):
            context.input_image = Image.fromarray(
                context.input_image.astype(np.uint8)
            )
        return Flux2ModelInputs.from_context(context)

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

    @staticmethod
    def _prepare_image_ids_from_shapes(
        latent_shapes: list[tuple[int, int]],
        batch_size: int,
        device: DeviceRef,
        scale: int = 10,
    ) -> Tensor:
        if len(latent_shapes) == 0:
            raise ValueError("Expected at least one latent shape.")

        coords_all = []
        for i, (height, width) in enumerate(latent_shapes):
            t_coord = scale + scale * i
            t_coords = np.full((height, width), t_coord, dtype=np.int64)
            h_coords, w_coords = np.meshgrid(
                np.arange(height, dtype=np.int64),
                np.arange(width, dtype=np.int64),
                indexing="ij",
            )
            l_coords = np.zeros((height, width), dtype=np.int64)
            coords = np.stack([t_coords, h_coords, w_coords, l_coords], axis=-1)
            coords_all.append(coords.reshape(-1, 4))

        coords_concat = np.concatenate(coords_all, axis=0)
        batched = np.tile(coords_concat[np.newaxis, :, :], (batch_size, 1, 1))
        return Tensor.from_dlpack(np.ascontiguousarray(batched)).to(device)

    def _encode_vae_image(
        self,
        image: Tensor,
        dtype: DType,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> tuple[Tensor, tuple[int, int]]:
        if image.rank != 4:
            raise ValueError(f"Expected image dims 4, got {image.rank}.")

        encoder_output = self.vae.encode(image, return_dict=True)
        if isinstance(encoder_output, dict):
            encoder_output = encoder_output["latent_dist"]

        image_latents = self.retrieve_latents(
            encoder_output, generator=generator, sample_mode=sample_mode
        )
        latent_h = image_latents.shape[2].dim
        latent_w = image_latents.shape[3].dim
        latent_prep_model = self._ensure_latent_prep_model(latent_h, latent_w, dtype)
        packed_drv = latent_prep_model.execute(
            image_latents.cast(DType.float32).driver_tensor
        )[0]
        packed = Tensor.from_dlpack(packed_drv)

        bn_mean = self.vae.bn.running_mean.cast(dtype)
        bn_var = self.vae.bn.running_var.cast(dtype)
        num_channels = bn_mean.shape[0].dim
        bn_mean = F.reshape(bn_mean, (1, 1, num_channels))
        bn_std = F.sqrt(
            F.reshape(bn_var, (1, 1, num_channels)) + self.vae.config.batch_norm_eps
        )
        packed = (packed - bn_mean) / bn_std
        return packed, (latent_h // 2, latent_w // 2)

    def prepare_image_latents(
        self,
        images: list[Tensor],
        batch_size: int,
        device: DeviceRef,
        dtype: DType,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> tuple[Tensor, Tensor]:
        image_latents: list[Tensor] = []
        latent_shapes: list[tuple[int, int]] = []
        for image in images:
            image = image.to(device).cast(self.vae.config.dtype)
            latent, latent_shape = self._encode_vae_image(
                image=image,
                dtype=dtype,
                generator=generator,
                sample_mode=sample_mode,
            )
            image_latents.append(latent)
            latent_shapes.append(latent_shape)

        if len(image_latents) == 1:
            condition_latents = image_latents[0]
        else:
            condition_latents = F.concat(image_latents, axis=1)

        condition_latents = F.tile(condition_latents, (batch_size, 1, 1))
        image_latent_ids = self._prepare_image_ids_from_shapes(
            latent_shapes=latent_shapes,
            batch_size=batch_size,
            device=device,
        )
        return condition_latents, image_latent_ids

    def _prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
        hidden_states_layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Create prompt embeddings and text position IDs for the transformer.

        Flux2 uses multiple hidden-state layers from the text encoder. Selected
        layers are padded/trimmed to a common sequence length, stacked, and then
        flattened across the layer/hidden dimensions.

        Args:
            tokens: TokenBuffer produced by tokenization / chat templating.
            num_images_per_prompt: Number of image generations per prompt.
            hidden_states_layers: Optional indices of hidden-state layers to use.

        Returns:
            A tuple of:
                - prompt_embeds: Tensor of shape (B', S, L*D)
                - text_ids: Tensor[int64] of shape (B', S, 4)
        """
        layers = hidden_states_layers or [10, 20, 30]
        max_seq = int(tokens.array.shape[-1])

        text_input_ids = Tensor.constant(
            tokens.array, dtype=DType.int64, device=self.text_encoder.devices[0]
        )
        hs_all = self.text_encoder(text_input_ids)

        selected: list[Tensor] = []
        for i in layers:
            hs = hs_all[i]
            hs = hs if isinstance(hs, Tensor) else Tensor.from_dlpack(hs)

            # Ensure [B, S, D]
            if hs.rank == 2:
                (hs_out,) = self._unsqueeze_model.execute(hs.driver_tensor)
                hs = Tensor.from_dlpack(hs_out)

            _, seq_len, _ = map(int, hs.shape)
            if seq_len < max_seq:
                hs = F.pad(
                    hs, pad=((0, 0), (0, max_seq - seq_len), (0, 0))
                )  # [B, max_seq, D]
            elif seq_len > max_seq:
                hs = hs[:, :max_seq, :]

            selected.append(hs)

        num_layers = len(selected)
        prompt_embed_model = self._ensure_prompt_embed_model(num_layers)
        driver_inputs = [hs.driver_tensor for hs in selected]
        prompt_embeds = Tensor.from_dlpack(prompt_embed_model.execute(*driver_inputs)[0])

        if num_images_per_prompt != 1:
            prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
            prompt_embeds = F.reshape(
                prompt_embeds, [batch_size * num_images_per_prompt, seq_len, -1]
            )
        
        bs_embed = prompt_embeds.shape[0].dim
        seq_len = prompt_embeds.shape[1].dim
        text_ids_key = f"{bs_embed}"
        if text_ids_key in self._cached_text_ids:
            text_ids = self._cached_text_ids[text_ids_key]
        else:
            text_ids_model = self._ensure_text_ids_model(
                batch_size=bs_embed,
                seq_len=seq_len,
            )
            text_ids = Tensor.from_dlpack(text_ids_model.execute()[0])
            self._cached_text_ids[text_ids_key] = text_ids

        return prompt_embeds, text_ids

    def _decode_latents(
        self,
        latents: Tensor,
        latent_image_ids: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent"] = "np",
    ) -> np.ndarray | Tensor:
        """Decode Flux2 packed latents into an image array (or return latents).

        Args:
            latents: Packed latents, typically shaped (B, S, C).
            latent_image_ids: Position IDs used to unpack into (B, C, H, W).
            output_type: "latent" to return latents, otherwise decode to NumPy.

        Returns:
            If output_type == "latent", returns latents (first element if B > 1).
            Otherwise returns a float32 HWC NumPy array.
        """
        if output_type == "latent":
            # Unpack latents (B, Seq, C) -> (B, C, H, W)
            latents_unpacked = self._unpack_latents(
                latents, height, width, latents.device, latents.dtype
            )
            # If 'latent', we return the Tensor.
            return (
                latents_unpacked[0]
                if latents_unpacked.shape[0].dim > 1
                else latents_unpacked
            )

        # Dimensions for Unpack: (H/16, W/16) from original inputs (patchify=2, vae=8)
        h_latent = height // 16
        w_latent = width // 16
        # Shapes from input tensor
        batch_size = latents.shape[0].dim
        seq_len = latents.shape[1].dim
        num_channels = latents.shape[2].dim
        # BN stats are already eager tensors from load_model()
        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

        model = self._ensure_vae_prep_model(
            h_latent,
            w_latent,
        )
        latents_unpacked_drv = model.execute(
            latents.driver_tensor, bn_mean.driver_tensor, bn_var.driver_tensor
        )[0]
        decoded = self.vae.decode(latents_unpacked_drv)
        return self._image_to_flat_hwc(self._to_numpy(decoded))

    def _to_numpy(self, image: Tensor) -> np.ndarray:
        """Convert a MAX Tensor to a CPU NumPy array (float32)."""
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    @staticmethod
    def _image_to_flat_hwc(image: np.ndarray) -> np.ndarray:
        """Convert a tensor-like NumPy image to a flat HWC float32 array."""
        img = np.asarray(image)
        while img.ndim > 3:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.float32, copy=False)

    def _preprocess_latents(
        self, latents: np.ndarray, latent_image_ids: Tensor, dtype: DType
    ) -> tuple[Tensor, Tensor]:
        latents = Tensor.from_dlpack(latents).to(self.transformer.devices[0])
        latent_prep_model = self._ensure_latent_prep_model(latents.shape[2], latents.shape[3], dtype)
        latents_drv = latent_prep_model.execute(latents.driver_tensor)[0]
        latents = Tensor.from_dlpack(latents_drv)

        latent_image_ids = Tensor.from_dlpack(
            latent_image_ids.astype(np.int64)
        ).to(self.transformer.devices[0])
        return latents, latent_image_ids

    def _pil_image_to_tensor(
        self,
        image: Image.Image,
    ) -> Tensor:
        img_array = (np.array(image, dtype=np.float32) / 127.5) - 1.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.ascontiguousarray(img_array)
        img_tensor = (
            Tensor.from_dlpack(img_array)
            .to(self.vae.devices[0])
            .cast(self.vae.config.dtype)
        )

        return img_tensor

    def execute(
        self,
        model_inputs: Flux2ModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent"] = "np",
    ) -> Flux2PipelineOutput:
        """Run the Flux2 denoising loop and decode outputs.

        Args:
            model_inputs: Inputs containing tokens, latents, timesteps, sigmas, and IDs.
            callback_queue: Optional queue for streaming intermediate decoded outputs.
            output_type: Output mode ("np", "latent")

        Returns:
            Flux2PipelineOutput containing one output per batch element.
        """
        # 1) Encode prompts.
        prompt_embeds, text_ids = self._prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )
        batch_size = int(prompt_embeds.shape[0])
        dtype = prompt_embeds.dtype

        image_latents = None
        if model_inputs.input_image is not None:
            image_tensor = self._pil_image_to_tensor(model_inputs.input_image)
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=[image_tensor],
                batch_size=batch_size,
                device=self.transformer.devices[0],
                dtype=dtype,
            )

        # 2) Prepare latents and conditioning tensors.
        latents, latent_image_ids = self._preprocess_latents(
            model_inputs.latents, model_inputs.latent_image_ids, dtype
        )
        if image_latents is not None:
            if image_latents.shape[1].dim != latents.shape[1].dim:
                raise ValueError(
                    "Input image latent sequence length mismatch. "
                    f"Expected {latents.shape[1].dim}, got {image_latents.shape[1].dim}."
                )

        # 3) Prepare scheduler tensors.
        device = self.transformer.devices[0]
        guidance_key = f"{batch_size}_{model_inputs.guidance_scale}_{dtype}_{device}"
        if guidance_key in self._cached_guidance:
            guidance = self._cached_guidance[guidance_key]
        else:
            guidance = Tensor.full(
                [latents.shape[0]],
                model_inputs.guidance_scale,
                device=device,
                dtype=dtype,
            )
            self._cached_guidance[guidance_key] = guidance

        image_seq_len = latents.shape[1].dim
        num_inference_steps = model_inputs.num_inference_steps
        sigmas_key = f"{num_inference_steps}_{image_seq_len}"
        if sigmas_key in self._cached_sigmas:
            sigmas = self._cached_sigmas[sigmas_key]
        else:
            mu = compute_empirical_mu(image_seq_len, num_inference_steps)
            # 1.0 down to 0.0 (N+1 points); time-shift excludes 0 to avoid 1/t in formula
            base_sigmas = np.linspace(
                1.0,
                1.0 / num_inference_steps,
                num_inference_steps,
                dtype=np.float32,
            )
            base_sigmas = _time_shift_exponential(mu, 1.0, base_sigmas)
            base_sigmas = np.append(base_sigmas, np.float32(0.0))
            sigmas = Tensor.from_dlpack(np.ascontiguousarray(base_sigmas)).to(
                device
            )
            self._cached_sigmas[sigmas_key] = sigmas

        encoder_hidden_states_drv = prompt_embeds.driver_tensor
        guidance_drv = guidance.driver_tensor
        txt_ids_drv = text_ids.driver_tensor
        img_ids_drv = latent_image_ids.driver_tensor

        sigmas_drv = sigmas.driver_tensor
        latents_drv = latents.driver_tensor
        all_timesteps_drv, all_dts_drv = self._time_step_model.execute(
            sigmas_drv
        )

        if hasattr(all_timesteps_drv, "driver_tensor"):
            all_timesteps_drv = all_timesteps_drv.driver_tensor
        if hasattr(all_dts_drv, "driver_tensor"):
            all_dts_drv = all_dts_drv.driver_tensor

        def _unwrap_model(model):
            while hasattr(model, "__wrapped__"):
                model = model.__wrapped__
            return model

        raw_compiled_model = _unwrap_model(self.transformer.model)
        raw_scheduler_step_model = _unwrap_model(self._scheduler_step_model)
        num_noise_tokens = Tensor.from_dlpack(np.array([latents_drv.shape[1]], dtype=np.int64))

        # 4) Denoising loop.
        for i in tqdm(range(num_inference_steps), desc="Denoising"):
            self._current_timestep = i
            timestep_drv = all_timesteps_drv[i : i + 1]
            dt_drv = all_dts_drv[i : i + 1]

            if image_latents is not None:
                latents_model_input_drv, latent_model_ids_drv = self._concat_model.execute(
                    latents_drv, image_latents.driver_tensor, img_ids_drv, image_latent_ids.driver_tensor
                )
            else:
                latents_model_input_drv = latents_drv
                latent_model_ids_drv = img_ids_drv

            noise_pred_drv = raw_compiled_model.execute(
                latents_model_input_drv,
                encoder_hidden_states_drv,
                timestep_drv,
                latent_model_ids_drv,
                txt_ids_drv,
                guidance_drv,
            )[0]

            latents_drv = raw_scheduler_step_model.execute(
                latents_drv, noise_pred_drv, dt_drv, num_noise_tokens
            )[0]

            if hasattr(device, "synchronize"):
                device.synchronize()

            if callback_queue is not None:
                latents = Tensor.from_dlpack(latents_drv)
                image = self._decode_latents(
                    latents,
                    latent_image_ids,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )

        latents = Tensor.from_dlpack(latents_drv)

        # 4) Decode final outputs per batch element.
        batch_size = latents.shape[0].dim
        image_list = []
        latent_image_ids_drv = latent_image_ids.driver_tensor
        for b in range(batch_size):
            latents_drv_b = latents_drv[b : b + 1, :, :]
            latent_image_ids_drv_b = latent_image_ids_drv[b : b + 1, :, :]
            latents_b = Tensor.from_dlpack(latents_drv_b)
            latent_image_ids_b = Tensor.from_dlpack(latent_image_ids_drv_b)

            image_b = self._decode_latents(
                latents_b,
                latent_image_ids_b,
                model_inputs.height,
                model_inputs.width,
                output_type=output_type,
            )
            image_list.append(image_b)

        return Flux2PipelineOutput(images=image_list)
