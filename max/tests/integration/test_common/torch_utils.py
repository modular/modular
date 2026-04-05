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
"""Utilities for running torch models for testing."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt
import requests
import torch
from diffusers import DiffusionPipeline
from max.support import fetch_bytes_from_s3
from PIL import Image
from test_common.numerics import log_softmax
from test_common.test_data import (
    MockPixelGenerationRequest,
    MockTextGenerationRequest,
)
from transformers import (
    LogitsProcessorList,
    MllamaProcessor,
    PixtralProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def _create_logits_store(
    generate_logprobs: bool = False,
) -> tuple[list[dict], Callable]:  # type: ignore[type-arg]
    """Create a logits storage function and container.

    The `saved_logits` is captured into the `store_logits` closure, which is
    injected into `model.generate` as a logits processor.
    This allows saving the logits, and optionally logprobs in addition.

    Args:
        generate_logprobs: If True, also compute and store logprobs in addition to logits.
    """
    saved_logits = []

    def store_logits(input_ids: torch.LongTensor, scores: torch.FloatTensor):  # noqa: ANN202
        _ = input_ids  # Unused.
        # Currently always passing in one batch at a time.
        scores_np = scores[0].cpu().detach().numpy()
        next_token = scores_np.argmax(axis=-1)

        # Always store logits
        entry = {
            "next_token": next_token,
            "next_token_logits": scores_np[next_token],
            "logits": scores_np,
        }

        if generate_logprobs:
            # Also compute and store logprobs in addition to logits
            scores_logprobs = log_softmax(scores_np)
            entry["next_token_logprobs"] = float(scores_logprobs[next_token])
            entry["logprobs"] = scores_logprobs

        saved_logits.append(entry)
        return scores

    return saved_logits, store_logits


def run_text_generation(  # noqa: ANN201
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer
    | PreTrainedTokenizerFast
    | MllamaProcessor
    | PixtralProcessor,
    device: torch.device,
    textgen_requests: Iterable[MockTextGenerationRequest],
    num_steps: int = 50,
    print_outputs: bool = False,
    use_cache: bool | None = None,
    generate_logprobs: bool = False,
):
    """Run text generation using standard data processor for both text and images."""

    def standard_request_processor(
        request: MockTextGenerationRequest,
    ) -> dict[str, torch.Tensor]:
        if len(request.images) > 0:
            processed_images = [
                Image.open(requests.get(image, stream=True).raw)
                for image in request.images
            ]
            assert len(processed_images) == 1
            return data_processor(
                images=processed_images[0],
                text=request.prompt,
                return_tensors="pt",
            ).to(device)
        else:
            encoded_prompt = data_processor.encode(
                request.prompt, return_tensors="pt"
            ).to(device)
            return {
                "input_ids": encoded_prompt,
                "attention_mask": torch.ones_like(encoded_prompt),
            }

    return run_text_generation_with_custom_image_processing(
        model=model,
        data_processor=data_processor,
        device=device,
        textgen_requests=textgen_requests,
        num_steps=num_steps,
        print_outputs=print_outputs,
        use_cache=use_cache,
        request_processor_fn=standard_request_processor,
        generate_logprobs=generate_logprobs,
    )


def run_text_generation_with_custom_image_processing(  # noqa: ANN201
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: torch.device,
    textgen_requests: Iterable[MockTextGenerationRequest],
    num_steps: int,
    print_outputs: bool,
    request_processor_fn: Callable[
        [MockTextGenerationRequest], dict[str, torch.Tensor]
    ],
    use_cache: bool | None = None,
    generate_logprobs: bool = False,
):
    """Run text generation with custom request processing for specialized models."""
    saved_logits, store_logits = _create_logits_store(
        generate_logprobs=generate_logprobs
    )
    results = []

    for request in textgen_requests:
        generate_kwargs = request_processor_fn(request)
        outputs = model.generate(
            **generate_kwargs,
            max_new_tokens=num_steps,
            do_sample=False,
            logits_processor=LogitsProcessorList([store_logits]),
            num_return_sequences=1,
            pad_token_id=getattr(data_processor, "eos_token_id", None),
            # Only pass use_cache if it's not None to avoid conflicts with
            # models such as InternVL that hardcode use_cache in their
            # generate_kwargs.
            **({"use_cache": use_cache} if use_cache is not None else {}),
        )

        if print_outputs:
            print(
                "Prompt:",
                f"{request.prompt[:100]}...{request.prompt[-100:]}"
                if len(request.prompt) > 200
                else request.prompt,
            )
            print(
                "Output:",
                data_processor.batch_decode(outputs, skip_special_tokens=True)[
                    0
                ],
            )

        results.append({"prompt": request.prompt, "values": saved_logits[:]})
        saved_logits.clear()

    return results


def run_embeddings_generation(  # noqa: ANN201
    model: PreTrainedModel,
    data_processor: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: torch.device,
    prompts: Iterable[str],
    pool_embeddings: bool = False,
):
    """Generates embeddings for the input prompts.

    Args:
        pool_embeddings: If True, applies last token pooling and L2 normalization
                        as per Qwen3-Embedding. If False, returns raw hidden states.
    """

    def last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the hidden state of the last non-padding token."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    results = []
    for prompt in prompts:
        encoded_input = data_processor(
            [prompt], padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        output = model(**encoded_input)

        if pool_embeddings:
            # Apply last token pooling to get single embedding per sequence
            embeddings = last_token_pool(
                output.last_hidden_state, encoded_input["attention_mask"]
            )
            # Apply L2 normalization
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu().detach().to(torch.float32).numpy()
            # Squeeze batch dimension to match MAX output shape: [batch_size=1, hidden_dim] -> [hidden_dim]
            if embeddings.shape[0] == 1:
                embeddings = embeddings.squeeze(0)
        else:
            # Return raw hidden states without pooling [batch_size, seq_len, hidden_dim]
            embeddings = (
                output.last_hidden_state.cpu()
                .detach()
                .to(torch.float32)
                .numpy()
            )

        results.append({"prompt": prompt, "embeddings": embeddings})
    return results


def _packed_randn_tensor(
    batch_size: int,
    num_channels_latents: int,
    latent_height: int,
    latent_width: int,
    seed: int | None,
) -> npt.NDArray[np.float32]:
    """
    This function is copied from max/pipelines/lib/pixel_tokenizer.py
    to generate same latents as MAX.
    """
    rng = np.random.RandomState(seed)
    latents = rng.standard_normal(
        (batch_size, num_channels_latents, latent_height, latent_width)
    ).astype(np.float32)
    # packed
    latents = latents.reshape(
        batch_size,
        num_channels_latents,
        latent_height // 2,
        2,
        latent_width // 2,
        2,
    )
    latents = latents.transpose(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size,
        (latent_height // 2) * (latent_width // 2),
        num_channels_latents * 4,
    )
    return latents


def _canonical_randn_tensor(
    batch_size: int,
    num_channels_latents: int,
    latent_height: int,
    latent_width: int,
    seed: int | None,
) -> npt.NDArray[np.float32]:
    """Generate latents in the same (B, C, H, W) format as MAX pipeline _prepare_latents.

    Using the same seed as MAX yields identical latent values, so diffusers and MAX
    can be given the same underlying noise when we patchify this for diffusers.
    """
    rng = np.random.RandomState(seed)
    return rng.standard_normal(
        (batch_size, num_channels_latents, latent_height, latent_width)
    ).astype(np.float32)


def _patchify_latents_numpy(
    latents: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Patchify (B, C, H, W) to (B, C*4, H//2, W//2). Same layout as MAX/diffusers _patchify_latents."""
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(
        batch_size,
        num_channels_latents,
        height // 2,
        2,
        width // 2,
        2,
    )
    latents = latents.transpose(0, 1, 3, 5, 2, 4)
    return latents.reshape(
        batch_size,
        num_channels_latents * 4,
        height // 2,
        width // 2,
    )


def _is_flux2_pipeline(pipeline: Any) -> bool:
    """Detect FLUX2 pipeline variants robustly.

    Class name checks alone can fail for wrapped/custom subclasses. We also
    inspect the diffusers config class marker when available.
    """
    class_name = pipeline.__class__.__name__
    if class_name == "Flux2Pipeline" or "Flux2" in class_name:
        return True

    config = getattr(pipeline, "config", None)
    if config is None:
        return False

    config_class_name = None
    if isinstance(config, dict):
        config_class_name = config.get("_class_name")
    elif hasattr(config, "get"):
        config_class_name = config.get("_class_name")
    else:
        config_class_name = getattr(config, "_class_name", None)

    return config_class_name == "Flux2Pipeline"


def _is_z_image_pipeline(pipeline: Any) -> bool:
    """Detect diffusers Z-Image pipelines (latents are BCHW, full transformer.in_channels)."""
    class_name = pipeline.__class__.__name__
    if class_name in ("ZImagePipeline", "ZImageImg2ImgPipeline"):
        return True

    config = getattr(pipeline, "config", None)
    if config is None:
        return False

    config_class_name = None
    if isinstance(config, dict):
        config_class_name = config.get("_class_name")
    elif hasattr(config, "get"):
        config_class_name = config.get("_class_name")
    else:
        config_class_name = getattr(config, "_class_name", None)

    return config_class_name in ("ZImagePipeline", "ZImageImg2ImgPipeline")


def _is_z_image_img2img_pipeline(pipeline: Any) -> bool:
    """True when the torch reference is diffusers ``ZImageImg2ImgPipeline``."""
    if pipeline.__class__.__name__ == "ZImageImg2ImgPipeline":
        return True
    config = getattr(pipeline, "config", None)
    if config is None:
        return False
    if isinstance(config, dict):
        return config.get("_class_name") == "ZImageImg2ImgPipeline"
    if hasattr(config, "get"):
        return config.get("_class_name") == "ZImageImg2ImgPipeline"
    return getattr(config, "_class_name", None) == "ZImageImg2ImgPipeline"


def _z_image_img2img_starting_latents(
    pipeline: Any,
    *,
    input_image: Image.Image,
    height: int,
    width: int,
    num_inference_steps: int,
    strength: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Match MAX ``prepare_img2img_latents``: VAE mode + numpy noise + scheduler.scale_noise.

    ``ZImageImg2ImgPipeline.prepare_latents`` skips image encoding when ``latents``
    is passed; we precompute the same noisy latents diffusers would build so the
    noise tensor matches ``_canonical_randn_tensor`` (and MAX).
    """
    from diffusers.pipelines.z_image.pipeline_z_image_img2img import (
        calculate_shift,
        retrieve_latents,
        retrieve_timesteps,
    )

    vae_scale = int(pipeline.vae_scale_factor) * 2
    if height % vae_scale != 0 or width % vae_scale != 0:
        raise ValueError(
            f"Height and width must be divisible by {vae_scale}; got {height}x{width}"
        )

    latent_height = 2 * (int(height) // (pipeline.vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (pipeline.vae_scale_factor * 2))
    num_channels_latents = int(pipeline.transformer.config.in_channels)
    image_seq_len = (latent_height // 2) * (latent_width // 2)

    init_image = pipeline.image_processor.preprocess(
        input_image,
        height=height,
        width=width,
    )
    init_image = init_image.to(device=device, dtype=torch.float32)

    encode_dtype = next(pipeline.vae.parameters()).dtype
    image_tensor = init_image.to(dtype=encode_dtype)
    encoder_output = pipeline.vae.encode(image_tensor)
    image_latents = retrieve_latents(
        encoder_output, generator=None, sample_mode="argmax"
    )
    shift = float(getattr(pipeline.vae.config, "shift_factor", 0.0) or 0.0)
    scale = float(pipeline.vae.config.scaling_factor)
    image_latents = (image_latents - shift) * scale
    image_latents = image_latents.to(device=device, dtype=torch.float32)

    sch_cfg = pipeline.scheduler.config
    mu = calculate_shift(
        image_seq_len,
        getattr(sch_cfg, "base_image_seq_len", 256),
        getattr(sch_cfg, "max_image_seq_len", 4096),
        getattr(sch_cfg, "base_shift", 0.5),
        getattr(sch_cfg, "max_shift", 1.15),
    )
    pipeline.scheduler.sigma_min = 0.0
    retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=None,
        mu=mu,
    )
    timesteps, _ = pipeline.get_timesteps(num_inference_steps, strength, device)
    if timesteps.shape[0] < 1:
        raise ValueError(
            "After strength adjustment, num_inference_steps is < 1 for Z-Image img2img."
        )
    batch_size = int(init_image.shape[0])
    latent_timestep = timesteps[:1].repeat(batch_size)

    latents_np = _canonical_randn_tensor(
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        seed,
    )
    noise = torch.from_numpy(latents_np).to(device=device, dtype=torch.float32)
    return pipeline.scheduler.scale_noise(image_latents, latent_timestep, noise)


def _load_input_image(image_uri: str) -> Image.Image:
    """Load an input image for image-to-image generation."""
    if image_uri.startswith("s3://"):
        image_bytes = fetch_bytes_from_s3(image_uri)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    if image_uri.startswith(("http://", "https://")):
        return Image.open(requests.get(image_uri, stream=True).raw).convert(
            "RGB"
        )
    return Image.open(image_uri).convert("RGB")


def run_image_generation(
    *,
    pipeline: DiffusionPipeline,
    device: torch.device,
    requests: list[MockPixelGenerationRequest],
    num_steps: int,
    print_outputs: bool = False,
) -> list[dict]:  # type: ignore[type-arg]
    """Run image generation using a diffusers pipeline.

    Args:
        pipeline: A diffusers pipeline (e.g., FluxPipeline)
        device: Device to run on
        requests: List of MockPixelGenerationRequest objects
        num_steps: Number of denoising steps (can override request values)
        print_outputs: Whether to print outputs

    Returns:
        List of dicts with prompt and generated images
    """

    results = []

    pipeline.to(device)  # type: ignore[attr-defined]
    is_flux2 = _is_flux2_pipeline(pipeline)
    is_z_image = _is_z_image_pipeline(pipeline)
    is_z_image_img2img = _is_z_image_img2img_pipeline(pipeline)

    for mock_request in requests:
        prompt = mock_request.prompt
        if print_outputs:
            print(f"Generating image for prompt: {prompt}")

        # Use parameters from mock_request, with num_steps override if different
        inference_steps = (
            num_steps
            if num_steps is not None
            else mock_request.num_inference_steps
        )
        height = (
            mock_request.height if mock_request.height is not None else 1024
        )
        width = mock_request.width if mock_request.width is not None else 1024
        guidance_scale = mock_request.guidance_scale
        seed = mock_request.seed if mock_request.seed is not None else 42
        input_image = (
            _load_input_image(mock_request.input_image)
            if mock_request.input_image is not None
            else None
        )
        strength = (
            float(mock_request.strength)
            if mock_request.strength is not None
            else 0.6
        )

        # Prepare latents using the same approach as MAX pipeline
        # This ensures deterministic and comparable outputs
        transformer_in_ch = pipeline.transformer.config.in_channels  # type: ignore[attr-defined]
        num_channels_latents = (
            transformer_in_ch if is_z_image else transformer_in_ch // 4
        )
        vae_scale_factor = pipeline.vae_scale_factor  # type: ignore[attr-defined]
        latent_height = 2 * (height // (vae_scale_factor * 2))
        latent_width = 2 * (width // (vae_scale_factor * 2))

        # Z-Image img2img: diffusers ignores the input image when ``latents`` is set.
        # Build the same flow-matching noisy latents as MAX (VAE mode + canonical noise).
        if is_z_image_img2img and input_image is not None:
            latents = _z_image_img2img_starting_latents(
                pipeline,
                input_image=input_image,
                height=height,
                width=width,
                num_inference_steps=inference_steps,
                strength=strength,
                seed=seed,
                device=device,
            )
        # Generate latents using numpy RandomState (same as MAX).
        # Flux2 expects patchified latents shaped (B, C*4, H//2, W//2).
        # Z-Image matches MAX pixel_tokenizer: (B, in_channels, H, W), no packing.
        elif is_flux2:
            latents_np = _canonical_randn_tensor(
                batch_size=1,
                num_channels_latents=num_channels_latents,
                latent_height=latent_height,
                latent_width=latent_width,
                seed=seed,
            )
            latents_np = _patchify_latents_numpy(latents_np)
            latents = torch.from_numpy(latents_np)
        elif is_z_image:
            latents_np = _canonical_randn_tensor(
                batch_size=1,
                num_channels_latents=num_channels_latents,
                latent_height=latent_height,
                latent_width=latent_width,
                seed=seed,
            )
            latents = torch.from_numpy(latents_np)
        else:
            latents = torch.from_numpy(
                _packed_randn_tensor(
                    batch_size=1,
                    num_channels_latents=num_channels_latents,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    seed=seed,
                )
            )

        if print_outputs:
            print(f"Latent shape: {latents.shape}, dtype: {latents.dtype}")

        # Generate image with pre-generated latents
        pipeline_kwargs = {
            "prompt": prompt,
            "latents": latents,
            "num_inference_steps": inference_steps,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
        }

        if input_image is not None:
            pipeline_kwargs["image"] = input_image
            if is_z_image_img2img:
                pipeline_kwargs["strength"] = strength

        # Add negative prompt if provided
        if mock_request.negative_prompt:
            pipeline_kwargs["negative_prompt"] = mock_request.negative_prompt

        output = pipeline(**pipeline_kwargs)  # type: ignore[operator]

        # Convert PIL image to numpy array
        image = output.images[0]
        image_np = np.array(
            image
        )  # Shape: (H, W, C), dtype typically uint8 [0, 255]

        # Normalize to [0, 1] to match MAX output after postprocess
        if image_np.dtype == np.uint8:
            image_np = image_np.astype(np.float32) / 255.0

        if print_outputs:
            print(
                f"Generated image shape: {image_np.shape}, dtype: {image_np.dtype}, range: [{image_np.min():.3f}, {image_np.max():.3f}]"
            )

        results.append(
            {
                "prompt": prompt,
                "images": image_np,
            }
        )

    return results
