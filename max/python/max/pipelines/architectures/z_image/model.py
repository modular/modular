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

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from .nn.image_processor import VaeImageProcessor
from .nn.autoencoder_kl import AutoencoderKL
from .nn.transformer_z_image import ZImageTransformer2DModel
#from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # TODO ?
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm

import logging
import time
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
from max._core.engine import Model
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Value
from max.graph.tensor_utils import cast_tensors_to
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.kv_cache import (
    NullKVCacheManager,
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.nn import Module, ReturnLogits, Signals
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, PagedCacheValues
from max.nn.parallel import ParallelArrayOps
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    ModelInputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from max.interfaces import GenerationStatus, RequestID
from max.interfaces.pipeline_variants.image_generation import (
    ImageGenerationMetadata,
    ImageGenerationOutput,
)
from max.profiler import Tracer, traced
from transformers import AutoConfig

from .context import ZImageTextAndVisionContext, VisionEncodingData
from .model_config import ZImageConfig
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from .z_image import ZImage
from .qwen3_encoder import Qwen3Encoder

from max.experimental import tensor, random, functional as F
from max.experimental.tensor import ops

logger = logging.getLogger("max.pipelines")

@dataclass(frozen=True)
class ZImagePipelineOutput:
    """
    Output class for Z-Image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `Tensor`)
            List of denoised PIL images of length `batch_size` or max tensor of shape `(batch_size, height, width,
            num_channels)`. PIL images or max tensor present the denoised images of the diffusion pipeline.
    """

    images: List[PIL.Image.Image] | Tensor


@dataclass(eq=False)
class ZImageInputs(ModelInputs):
    """A class representing inputs for the ZImage model.

    This class encapsulates the input tensors required for the ZImage model execution,
    including both text and vision inputs. Vision inputs are optional and can be None
    for text-only processing."""


    prompt: str | List[str] = None
    """ The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds` instead."""

    height: int | None = None
    """ The height in pixels of the generated image."""

    width: int | None = None
    """ The width in pixels of the generated image."""

    num_inference_steps: int = 50
    """ The number of denoising steps. More denoising steps usually lead to a higher quality image at the
    expense of slower inference."""

    sigmas: List[float] | None = None
    """ Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
    their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
    will be used."""

    cfg_normalization: bool = False
    """ Whether to apply configuration normalization."""

    cfg_truncation: float = 1.0
    """ The truncation value for configuration."""

    negative_prompt: str | List[str] | None = None
    """ The prompt or prompts not to guide the image generation. If not defined, one has to pass
    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
    less than `1`)."""

    num_images_per_prompt: int | None = 1
    """ The number of images to generate per prompt."""

    generator: F.Generator | List[F.Generator] | None = None
    """ One or a list of [max generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
    to make generation deterministic."""

    latents: Tensor | None = None
    """ Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
    tensor will be generated by sampling using the supplied random `generator`."""

    prompt_embeds: List[Tensor] | None = None
    """ Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
    provided, text embeddings will be generated from `prompt` input argument."""

    negative_prompt_embeds: List[Tensor] | None = None
    """ Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument."""

    output_type: str | None = "pil"
    """ The output format of the generate image. Choose between
    [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`."""
    joint_attention_kwargs: Dict[str, Any] | None = None
    """ A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
    `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)."""

    callback_on_step_end: Callable[[int, int, Dict], None] | None = None
    """ A function that calls at the end of each denoising steps during the inference. The function is called
    with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
    callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
    `callback_on_step_end_tensor_inputs`."""

    callback_on_step_end_tensor_inputs: tuple[str] = ("latents")
    """ Tuple of tensors to be included in the `callback_kwargs` dictionary."""

    max_sequence_length: int = 512
    """ The maximum sequence length for the input sequence."""

    #signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    cu_seqlens: list[Tensor] | None = None
    """Cumulative sequence lengths for full attention."""

    max_seqlen: list[Tensor] | None = None
    """Maximum sequence length for full attention for vision inputs."""


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler: SchedulerMixin,
    num_inference_steps: int | None = None,
    device: str | Device | None = None,
    timesteps: List[int] | None = None,
    sigmas: List[float] | None = None,
    **kwargs,
) -> Tuple[Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `Device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class ZImageModel(
    #AlwaysSignalBuffersMixin,
    PipelineModel[TextAndVisionContext],
):
    """A ZImage model for text-to-image generation."""

    scheduler: FlowMatchEulerDiscreteScheduler
    """The scheduler to be used for image generation."""

    vae: Model
    """The VAE to be used for image generation."""

    text_encoder: Qwen3Config
    """The text encoder to be used for image generation."""

    transformer: Model
    """The transformer to be used for image generation."""

    model_config: ZImageConfig | None
    """The ZImage model configuration."""

    _input_row_offsets_prealloc: list[Tensor]
    """Pre-allocated per-device tensors for input row offsets in multi-step execution."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ):
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.model_config = None
        self._session = session  # reuse for on-device casts

        self.vae, self.text_encoder, self.transformer = self.load_model(session)

        self._parallel_ops = ParallelArrayOps(max_workers=24)

        # self.vae_scale_factor = 2 ** (len(self.vae.block_out_channels) - 1)
        # Access config from vae instance or config object
        self.vae_scale_factor = 2 ** (len(self.model_config.vae_config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor()

        # Serving interface: image_pipeline is self
        self._active_requests: dict[str, Any] = {}

    @property
    def image_pipeline(self) -> "ZImageModel":
        """Returns self for ImageGeneratorPipeline compatibility."""
        return self

    def next_chunk(
        self, contexts: dict[RequestID, TextAndVisionContext]
    ) -> dict[RequestID, ImageGenerationOutput]:
        """Process the next chunk of inputs (serving interface)."""
        outputs = {}
        for req_id, ctx in contexts.items():
            # Extract tokens
            # tokens is numpy array, convert to tensor.
            # active_length gives the length of valid tokens in the context window.
            # We assume the context contains the full prompt tokens.
            input_ids = tensor.Tensor(ctx.all_tokens, dtype=DType.int32, device=self.devices[0]).unsqueeze(0)

            # Extract parameters from extra_model_args or use defaults
            # TextAndVisionContext extra_model_args values are numpy arrays.
            width = int(ctx.extra_model_args.get("width", np.array([1024])).item()) if "width" in ctx.extra_model_args else 1024
            height = int(ctx.extra_model_args.get("height", np.array([1024])).item()) if "height" in ctx.extra_model_args else 1024
            guidance_scale = float(ctx.extra_model_args.get("guidance_scale", np.array([5.0])).item()) if "guidance_scale" in ctx.extra_model_args else 5.0
            num_inference_steps = int(ctx.extra_model_args.get("num_inference_steps", np.array([50])).item()) if "num_inference_steps" in ctx.extra_model_args else 50

            # Run generation
            # TODO: Support batching across requests
            images = self.__call__(
                input_ids=input_ids,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="tensor", # Return tensor for serving response
                return_dict=False
            )

            # images is a tuple (tensor,) or (list[PIL],)
            image_data = images[0]

            # Create output
            # ImageGenerationOutput requires image_data as numpy float32.
            # If tensor, move to cpu and numpy.
            if isinstance(image_data, Tensor):
                image_data_np = image_data.to(Device.cpu()).to_numpy()
            else:
                # Handle PIL input if needed (though we requested tensor)
                image_data_np = np.array(image_data)

            # Metadata
            metadata = ImageGenerationMetadata(
                model_name=self.pipeline_config.model_config.served_model_name,
                request_id=req_id
            )

            outputs[req_id] = ImageGenerationOutput(
                final_status=GenerationStatus.FINISHED,
                steps_executed=num_inference_steps,
                image_data=image_data_np.flatten().astype(np.float32),
                metadata=metadata
            )
        return outputs

    def do_classifier_free_guidance(self, guidance_scale: float) -> bool:
        return guidance_scale > 1.0

    @classmethod
    def calculate_max_seq_len(cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig) -> int:
        """Calculate the optimal max sequence length for the model."""
        # For Z-Image, this might be fixed or configurable.
        # Returning a safe default for now, similar to other vision models.
        return 4096


    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration."""
        if huggingface_config is None:
             # If config is None (e.g. from PipelineModel init), return a safe default or 0
             # This might be used for batch size inference.
             return 0
        return ZImageConfig.get_num_layers(huggingface_config)

    def load_model(self, session: InferenceSession) -> tuple[AutoencoderKL, Model, ZImageTransformer2DModel]:
        if self.pipeline_config.max_batch_size is None:
            raise ValueError("Expected max_batch_size to be set")
        self._input_row_offsets_prealloc = F.range(self.pipeline_config.max_batch_size + 1, dtype=DType.uint32).to(self.devices)

        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError("ZImage currently only supports safetensors weights")

        if self.adapter:
            model_state_dict = self.adapter(dict(self.weights.items()))
        else:
            model_state_dict = {key: value.data() for key, value in self.weights.items()}

        vae_state_dict: dict[str, WeightData] = {}
        text_encoder_state_dict: dict[str, WeightData] = {}
        transformer_state_dict: dict[str, WeightData] = {}

        for key, value in model_state_dict.items():
            if key.startswith("vae."):
                vae_state_dict[key] = value
            elif key.startswith("text_encoder."):
                text_encoder_state_dict[key] = value
            elif key.startswith("transformer."):
                transformer_state_dict[key] = value
            else:
                raise ValueError(
                    f"Key: {key} is not part of the VAE, text encoder or transformer"
                )

        # Generate ZImage config from HuggingFace config
        zimage_config = ZImageConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            vae_state_dict=vae_state_dict,
            text_encoder_state_dict=text_encoder_state_dict,
            transformer_state_dict=transformer_state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )
        self.model_config = zimage_config

        if self.model_config is None:
            raise ValueError("Model config must be initialized")

        # Instantiate ZImage container to build sub-models
        self.model = ZImage(self.model_config)

        # Load weights for VAE (Eager)
        self.model.vae.load_state_dict(vae_state_dict)

        # Load weights for Transformer (Eager)
        self.model.transformer.load_state_dict(transformer_state_dict)

        # Compile Text Encoder (Graph)
        logger.info("Building and compiling text encoder model...")
        before = time.perf_counter()
        text_encoder_graph = self._build_text_encoder_graph()
        text_encoder_model = session.load(
            text_encoder_graph, weights_registry=text_encoder_state_dict
        )
        after = time.perf_counter()
        logger.info(f"Building and compiling text encoder model took {after - before:.6f} seconds")

        return self.model.vae, text_encoder_model, self.model.transformer

    def _build_text_encoder_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        session: InferenceSession | None = None,
    ) -> Graph:
        # Retrieve config
        state_dict = self._get_state_dict(weights, adapter)
        model_config = Qwen3Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )

        # Build Graph
        nn_model: Module
        nn_model = Qwen3Encoder(model_config)

        # Get Graph Inputs
        graph_inputs = nn_model.input_types(self.kv_manager)

        # Load weights.
        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            # Stops strict from raising error when sharing LM head weights
            # (as LM head is never technically loaded from the state dict)
            strict=(
                not getattr(
                    self.huggingface_config, "tie_word_embeddings", False
                )
            ),
        )

        self.state_dict = nn_model.state_dict()

        with Graph("qwen3", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            kv_collection = PagedCacheValues(
                kv_blocks=kv_cache_inputs[0].buffer,
                cache_lengths=kv_cache_inputs[1].tensor,
                lookup_table=kv_cache_inputs[2].tensor,
                max_lengths=kv_cache_inputs[3].tensor,
            )
            outputs = nn_model(
                tokens.tensor,
                kv_collection,
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph

    def encode_prompt(
        self,
        prompt: str | List[str],
        device: Device | None = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | List[str] | None = None,
        prompt_embeds: List[Tensor] | None = None,
        negative_prompt_embeds: Tensor | None = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            prompt_embeds,
            max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                     negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if len(prompt) != len(negative_prompt):
                 raise ValueError("The lists of prompt and negative prompt must have the same length")
            negative_prompt_embeds = self._encode_prompt(
                negative_prompt,
                device,
                negative_prompt_embeds,
                max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: str | List[str],
        device: Device | None = None,
        prompt_embeds: List[Tensor] | None = None,
        max_sequence_length: int = 512,
    ) -> List[Tensor]:
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

            if isinstance(prompt, str):
                prompt = [prompt]

            for i, prompt_item in enumerate(prompt):
                messages = [
                    {"role": "user", "content": prompt_item},
                ]
                prompt_item = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                prompt[i] = prompt_item

            text_inputs = self.tokenizer(
                prompt,
                #padding="max_length",
                #max_length=max_sequence_length,
                truncation=True,
                #return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids.to(device)
            prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: DType,
        device: Device,
        generator: Generator,
        latents: Tensor | None = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = random.normal(shape,
                                    device=device,
                                    #generator=generator,  # TODO: implement generator
                                    mean=0.0,
                                    std=1.0,
                                    dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self) -> Dict[str, Any] | None:
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def interrupt(self) -> Callable[[Tensor], bool] | None:
        return self._interrupt

    def progress_bar(self, total: int):
        return tqdm(total=total)

    def __call__(
        self,
        prompt: str | List[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: List[float] | None = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: str | List[str] | None = None,
        num_images_per_prompt: int | None = 1,
        generator: Generator | List[Generator] | None = None,
        latents: Tensor | None = None,
        prompt_embeds: List[Tensor] | None = None,
        negative_prompt_embeds: List[Tensor] | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, Dict], None] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str] = ("latents"),
        max_sequence_length: int = 512,
        #signal_buffers: list[Tensor] | None = None,
        cu_seqlens: list[Tensor] | None = None,
        max_seqlen: list[Tensor] | None = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            cfg_normalization (`bool`, *optional*, defaults to False):
                Whether to apply configuration normalization.
            cfg_truncation (`float`, *optional*, defaults to 1.0):
                The truncation value for configuration.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`F.Generator` or `List[F.Generator]`, *optional*):
                One or a list of [max generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`List[Tensor]`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`List[Tensor]`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.ZImagePipelineOutput`] instead of a plain
                tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`tuple`, *optional*):
                The tuple of tensor inputs for the `callback_on_step_end` function. The tensors specified in the tuple
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.z_image.ZImagePipelineOutput`] or `tuple`: [`~pipelines.z_image.ZImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """
        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        # If prompt_embeds is provided and prompt is None, skip encoding
        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            DType.float32,
            device,
            generator,
            latents,
        )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (int(latents.shape[2]) // 2) * (int(latents.shape[3]) // 2)

        # 5. Prepare timesteps
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(int(timesteps.shape[0]) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = int(timesteps.shape[0])

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(self._num_timesteps):
                t = timesteps[i]
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = F.broadcast_to(t, (int(latents.shape[0]),))
                timestep = (1000 - timestep) / 1000
                # Normalized time for time-aware config (0 at start, 1 at end)
                t_norm = timestep[0].item()

                # Handle cfg truncation
                current_guidance_scale = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                # Run CFG only if configured AND scale is non-zero
                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                if apply_cfg:
                    latents_typed = latents.cast(self.transformer.dtype)
                    latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    timestep_model_input = timestep.repeat(2)
                else:
                    latent_model_input = latents.cast(self.transformer.dtype)
                    prompt_embeds_model_input = prompt_embeds
                    timestep_model_input = timestep

                latent_model_input = F.unsqueeze(latent_model_input, 2)
                latent_model_input_list = [latent_model_input[i] for i in range(int(latent_model_input.shape[0]))]

                model_out_list = self.transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                )[0]

                if apply_cfg:
                    # Perform CFG
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].cast(DType.float32)
                        neg = neg_out[j].cast(DType.float32)

                        pred = pos + current_guidance_scale * (pos - neg)

                        # Renormalization
                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            #ori_pos_norm = torch.linalg.vector_norm(pos)
                            #new_pos_norm = torch.linalg.vector_norm(pred)
                            ori_pos_norm = F.sqrt(F.sum(pos * pos))
                            new_pos_norm = F.sqrt(F.sum(pred * pred))
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)

                        noise_pred.append(pred)

                    noise_pred = F.stack(noise_pred, axis=0)
                else:
                    noise_pred = F.stack([t.cast(DType.float32) for t in model_out_list], axis=0)

                noise_pred = F.squeeze(noise_pred, 2)
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.cast(DType.float32), t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == int(timesteps.shape[0]) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = latents.cast(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type)

        # Offload all models
        #self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ZImagePipelineOutput(images=image)

    def next_chunk(
        self, batch: dict[str, Any]
    ) -> dict[str, "ImageGenerationOutput"]:
        """Process a batch of image generation requests.

        This method implements the serving interface expected by ImageGeneratorPipeline.

        Args:
            batch: Dictionary mapping request IDs to context objects containing prompts.

        Returns:
            Dictionary mapping request IDs to ImageGenerationOutput objects.
        """
        from max.interfaces import ImageGenerationOutput

        outputs: dict[str, ImageGenerationOutput] = {}

        for request_id, context in batch.items():
            # Track active request
            self._active_requests[request_id] = context

            # Extract prompt from context
            prompt = getattr(context, "prompt", None) or getattr(context, "text", "")
            height = getattr(context, "height", 1024)
            width = getattr(context, "width", 1024)
            num_inference_steps = getattr(context, "num_inference_steps", 50)

            # Generate image using __call__
            result = self(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                output_type="pil",
                return_dict=True,
            )

            # Create output
            outputs[request_id] = ImageGenerationOutput(
                image_data=result.images[0] if isinstance(result.images, list) else result.images,
                steps_executed=num_inference_steps,
            )

            # Clean up
            if request_id in self._active_requests:
                del self._active_requests[request_id]

        return outputs

    def release(self, request_id: str) -> None:
        """Release resources for a completed request.

        Args:
            request_id: The ID of the request to release.
        """
        if request_id in self._active_requests:
            del self._active_requests[request_id]
