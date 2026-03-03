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

from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal, cast

import max.experimental.functional as F
import numpy as np
import numpy.typing as npt
import torch
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.pipelines import PixelContext
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    PixelModelInputs,
)
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from max.profiler import Tracer
from tqdm import tqdm
from transformers import Gemma3ForConditionalGeneration

from ..autoencoders import (
    AutoencoderKLLTX2AudioModel,
    AutoencoderKLLTX2VideoModel,
)
from .model import (
    LTX2TextConnectorsModel,
    LTX2TransformerModel,
    LTX2VocoderModel,
)


@dataclass(kw_only=True)
class LTX2ModelInputs(PixelModelInputs):
    """
    LTX2-specific PixelModelInputs.
    """

    width: int = 768
    height: int = 512
    guidance_scale: float = 4.0
    num_inference_steps: int = 40
    num_visuals_per_prompt: int = 1
    num_frames: int = 121
    frame_rate: float = 24.0
    mask: npt.NDArray[np.bool_] | None = None
    """Attention mask for the text encoder."""
    extra_params: dict[str, npt.NDArray[Any]] | None = None
    """LTX2-specific preprocessed arrays."""

    @property
    def do_true_cfg(self) -> bool:
        return self.negative_tokens is not None


@dataclass
class LTX2PipelineOutput:
    """Output class for LTX2 video+audio generation pipelines.

    Args:
        frames: Generated video tensor of shape ``(batch, frames, height, width, channels)``
            with values in ``[0, 1]``.
        audio: Generated audio waveform tensor of shape ``(batch, channels, samples)``.
    """

    frames: np.ndarray | Tensor
    audios: np.ndarray | Tensor


class LTX2Pipeline(DiffusionPipeline):
    """A LTX2 pipeline for text-to-video-audio generation."""

    vae: AutoencoderKLLTX2VideoModel
    audio_vae: AutoencoderKLLTX2AudioModel
    # text_encoder: Gemma3TextEncoderModel
    transformer: LTX2TransformerModel
    connectors: LTX2TextConnectorsModel
    vocoder: LTX2VocoderModel

    components = {
        "vae": AutoencoderKLLTX2VideoModel,
        "audio_vae": AutoencoderKLLTX2AudioModel,
        # "text_encoder": Gemma3TextEncoderModel,
        "transformer": LTX2TransformerModel,
        "connectors": LTX2TextConnectorsModel,
        "vocoder": LTX2VocoderModel,
    }

    def init_remaining_components(self) -> None:
        """Initialize derived attributes that depend on loaded components."""

        self.audio_vae_mel_compression_ratio = 4

        model_id = str(self.pipeline_config.model.model_path)
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        device = self.transformer.devices[0]
        stats_dtype = DType.float32
        vae_mean = getattr(self.vae, "latents_mean", None)
        vae_std = getattr(self.vae, "latents_std", None)
        if vae_mean is not None and vae_std is not None:
            self._vae_latents_mean: Tensor | None = Tensor.constant(
                np.array(vae_mean, dtype=np.float32),
                dtype=stats_dtype,
                device=device,
            )
            self._vae_latents_std: Tensor | None = Tensor.constant(
                np.array(vae_std, dtype=np.float32),
                dtype=stats_dtype,
                device=device,
            )
        else:
            self._vae_latents_mean = None
            self._vae_latents_std = None

        audio_mean = getattr(self.audio_vae, "latents_mean", None)
        audio_std = getattr(self.audio_vae, "latents_std", None)
        if audio_mean is not None and audio_std is not None:
            self._audio_latents_mean: Tensor | None = Tensor.constant(
                np.array(audio_mean, dtype=np.float32),
                dtype=stats_dtype,
                device=device,
            )
            self._audio_latents_std: Tensor | None = Tensor.constant(
                np.array(audio_std, dtype=np.float32),
                dtype=stats_dtype,
                device=device,
            )
        else:
            self._audio_latents_mean = None
            self._audio_latents_std = None

        self.build_pack_latents()
        self.build_pack_audio_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step_video()
        self.build_scheduler_step_audio()
        self.build_decode_video_latents()
        self.build_decode_audio_latents()
        self.build_prepare_cfg_latents_timesteps()
        self.build_apply_cfg_guidance()

        self._cached_text_ids: dict[str, Tensor] = {}
        self._cached_sigmas: dict[str, Tensor] = {}

    def prepare_inputs(self, context: PixelContext) -> LTX2ModelInputs:  # type: ignore[override]
        return LTX2ModelInputs.from_context(context)

    # -----------------------------------------------------------------------
    # build_* methods: compile hot-path functions via max_compile (Flux2 style)
    # -----------------------------------------------------------------------

    def build_pack_latents(self) -> None:
        device = self.transformer.devices[0]
        _channels = self.transformer.config.in_channels
        _latent_num_frames = 16  # (121-1)//8+1
        _latent_height = 16
        _latent_width = 24
        input_types = [
            TensorType(
                DType.float32,
                shape=[
                    1,
                    _channels,
                    _latent_num_frames,
                    _latent_height,
                    _latent_width,
                ],
                device=device,
            ),
        ]
        self.__dict__["_pack_video_latents"] = max_compile(
            self._pack_video_latents,
            input_types=input_types,
        )

    def build_pack_audio_latents(self) -> None:
        device = self.transformer.devices[0]
        _latent_mel_bins = 64 // self.audio_vae_mel_compression_ratio  # 16
        _audio_channels = (
            self.transformer.config.audio_in_channels // _latent_mel_bins
        )  # 8
        _audio_num_frames = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                DType.float32,
                shape=[1, _audio_channels, _audio_num_frames, _latent_mel_bins],
                device=device,
            ),
        ]
        self.__dict__["_pack_audio_latents_packed"] = max_compile(
            self._pack_audio_latents_packed,
            input_types=input_types,
        )

    def build_prepare_scheduler(self) -> None:
        """Compile prepare_scheduler: pre-compute timesteps and per-step dts."""
        input_types = [
            TensorType(
                DType.float32,
                shape=["num_sigmas"],
                device=self.transformer.devices[0],
            ),
        ]
        self.__dict__["prepare_scheduler"] = max_compile(
            self.prepare_scheduler,
            input_types=input_types,
        )

    def build_scheduler_step_video(self) -> None:
        """Compile scheduler_step_video: Euler update for video latents.

        batch=1, seq=6144 (16*16*24), channels=128
        """
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        _channels = self.transformer.config.in_channels  # 128
        _video_seq_len = 6144  # 16 * 16 * 24
        input_types = [
            TensorType(
                dtype, shape=[1, _video_seq_len, _channels], device=device
            ),
            TensorType(
                DType.float32,
                shape=[1, _video_seq_len, _channels],
                device=device,
            ),
            TensorType(DType.float32, shape=[1], device=device),
            TensorType(DType.int64, shape=[], device=DeviceRef.CPU()),
        ]
        self.__dict__["scheduler_step_video"] = max_compile(
            self.scheduler_step,
            input_types=input_types,
        )

    def build_scheduler_step_audio(self) -> None:
        """Compile scheduler_step_audio: Euler update for audio latents.

        batch=1, seq=126 (round((121/24)*25)=126), channels=128
        """
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        _channels = self.transformer.config.audio_in_channels  # 128
        _audio_seq_len = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                dtype, shape=[1, _audio_seq_len, _channels], device=device
            ),
            TensorType(
                DType.float32,
                shape=[1, _audio_seq_len, _channels],
                device=device,
            ),
            TensorType(DType.float32, shape=[1], device=device),
            TensorType(DType.int64, shape=[], device=DeviceRef.CPU()),
        ]
        self.__dict__["scheduler_step_audio"] = max_compile(
            self.scheduler_step,
            input_types=input_types,
        )

    def build_decode_video_latents(self) -> None:
        """Compile _postprocess_video_latents if VAE latent statistics are available.

        Mirrors Flux2's build_decode_latents -> _postprocess_latents pattern.
        """
        if self._vae_latents_mean is None or self._vae_latents_std is None:
            return
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        num_channels = int(self._vae_latents_mean.shape[0])  # 128
        _latent_num_frames = 16  # (121-1)//8+1
        _latent_height = 16  # 512//32
        _latent_width = 24  # 768//32
        input_types = [
            TensorType(
                dtype,
                shape=[
                    1,
                    num_channels,
                    _latent_num_frames,
                    _latent_height,
                    _latent_width,
                ],
                device=device,
            ),
            TensorType(DType.float32, shape=[num_channels], device=device),
            TensorType(DType.float32, shape=[num_channels], device=device),
        ]
        self.__dict__["_postprocess_video_latents"] = max_compile(
            self._postprocess_video_latents,
            input_types=input_types,
        )

    def build_decode_audio_latents(self) -> None:
        """Compile _postprocess_audio_latents if audio VAE statistics are available.

        Mirrors Flux2's build_decode_latents -> _postprocess_latents pattern.
        """
        if self._audio_latents_mean is None or self._audio_latents_std is None:
            return
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        # latents_mean has shape [C*M = base_channels = 128], matching the packed
        # audio tensor last dim [B, L, C*M].  Denormalization happens on the packed
        # tensor before unpack — same order as diffusers.
        num_channels = int(self._audio_latents_mean.shape[0])  # 128 = C*M
        _audio_num_frames = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                dtype,
                shape=[1, _audio_num_frames, num_channels],
                device=device,
            ),
            TensorType(DType.float32, shape=[num_channels], device=device),
            TensorType(DType.float32, shape=[num_channels], device=device),
        ]
        self.__dict__["_postprocess_audio_latents"] = max_compile(
            self._postprocess_audio_latents,
            input_types=input_types,
        )

    def build_prepare_cfg_latents_timesteps(self) -> None:
        """Compile prepare_cfg_latents_timesteps: concat+cast for CFG latent doubling.

        Fuses two F.concat calls and two casts into a single compiled graph,
        eliminating per-step Python dispatch overhead.
          video: [1, 6144, 128] bfloat16 -> [2, 6144, 128] bfloat16
          audio: [1, 126, 128]   bfloat16 -> [2, 126, 128]   bfloat16
        """
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        _channels = self.transformer.config.in_channels  # 128
        _video_seq_len = 6144  # 16 * 16 * 24
        _audio_channels = self.transformer.config.audio_in_channels  # 128
        _audio_seq_len = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                dtype, shape=[1, _video_seq_len, _channels], device=device
            ),
            TensorType(
                dtype, shape=[1, _audio_seq_len, _audio_channels], device=device
            ),
            TensorType(dtype, shape=[1], device=device),
        ]
        self.__dict__["prepare_cfg_latents_timesteps"] = max_compile(
            self.prepare_cfg_latents_timesteps,
            input_types=input_types,
        )

    def build_apply_cfg_guidance(self) -> None:
        """Compile apply_cfg_guidance: CFG formula for video+audio noise preds.

        Fuses cast + split + guidance arithmetic into a single compiled graph:
          video in:  [2, 6144, 128] bfloat16 -> [1, 6144, 128] bfloat16
          audio in:  [2, 126, 128]   bfloat16 -> [1, 126, 128]   bfloat16
          guidance:  [1]             float32
        """
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        _channels = self.transformer.config.in_channels  # 128
        _video_seq_len = 6144  # 16 * 16 * 24
        _audio_channels = self.transformer.config.audio_in_channels  # 128
        _audio_seq_len = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                dtype, shape=[2, _video_seq_len, _channels], device=device
            ),
            TensorType(
                dtype, shape=[2, _audio_seq_len, _audio_channels], device=device
            ),
            TensorType(DType.float32, shape=[1], device=device),
        ]
        self.__dict__["apply_cfg_guidance"] = max_compile(
            self.apply_cfg_guidance,
            input_types=input_types,
        )

    def _encode_tokens(
        self,
        token_ids: np.ndarray,
        mask: np.ndarray,
        delete_encoder: bool = True,
    ) -> torch.Tensor:
        """Encode token_ids using transformers Gemma3ForConditionalGeneration.

        The token_ids come from PixelGenerationTokenizer which already handles
        tokenization. This method just runs them through the text encoder to
        get hidden states.

        Args:
            token_ids: Token IDs from PixelGenerationTokenizer (via model_inputs.token_ids).
            mask: Attention mask from PixelGenerationTokenizer (via model_inputs.mask).
            delete_encoder: Whether to delete the text encoder after encoding.

        Returns:
            Hidden states tensor from the text encoder, stacked across all layers.
        """
        input_ids = torch.from_dlpack(token_ids).to("cuda")
        attention_mask = torch.from_dlpack(mask).to("cuda")

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Stack all hidden states: [batch_size, seq_len, hidden_dim, num_layers]
            hidden_states = torch.stack(outputs.hidden_states, dim=-1)

        # Free GPU and CPU memory used by the text encoder after encoding.
        if delete_encoder:
            self.text_encoder.to("cpu")
            del self.text_encoder
            self.text_encoder = None
            torch.cuda.empty_cache()

        return hidden_states.to(torch.bfloat16)

    @staticmethod
    def _pack_text_embeds(
        text_hidden_states: torch.Tensor,
        sequence_lengths: Tensor,
        device: Device,
        padding_side: str = "left",
        scale_factor: int = 8,
        eps: float = 1e-6,
    ) -> Tensor:
        """
        Packs and normalizes text encoder hidden states, respecting padding. Normalization is performed per-batch and
        per-layer in a masked fashion (only over non-padded positions).

        Args:
            text_hidden_states (`Tensor` of shape `(batch_size, seq_len, hidden_dim, num_layers)`):
                Per-layer hidden_states from a text encoder (e.g. `Gemma3ForConditionalGeneration`).
            sequence_lengths (`Tensor of shape `(batch_size,)`):
                The number of valid (non-padded) tokens for each batch instance.
            device: (`Device`, *optional*):
                Device to place the resulting embeddings on
            padding_side: (`str`, *optional*, defaults to `"left"`):
                Whether the text tokenizer performs padding on the `"left"` or `"right"`.
            scale_factor (`int`, *optional*, defaults to `8`):
                Scaling factor to multiply the normalized hidden states by.
            eps (`float`, *optional*, defaults to `1e-6`):
                A small positive value for numerical stability when performing normalization.

        Returns:
            `Tensor` of shape `(batch_size, seq_len, hidden_dim * num_layers)`:
                Normed and flattened text encoder hidden states.
        """
        import torch

        # Convert MAX Tensors to PyTorch: MAX Tensor lacks masked_fill, amin,
        # amax, and multi-axis sum with keepdim.
        ths = text_hidden_states
        sl = torch.from_dlpack(sequence_lengths)
        torch_device = ths.device

        batch_size, seq_len, hidden_dim, num_layers = ths.shape
        original_dtype = ths.dtype

        # Create padding mask
        token_indices = torch.arange(seq_len, device=torch_device).unsqueeze(0)
        if padding_side == "right":
            # For right padding, valid tokens are from 0 to sequence_length-1
            mask = token_indices < sl[:, None]  # [batch_size, seq_len]
        elif padding_side == "left":
            # For left padding, valid tokens are from (T - sequence_length) to T-1
            start_indices = seq_len - sl[:, None]  # [batch_size, 1]
            mask = token_indices >= start_indices  # [B, T]
        else:
            raise ValueError(
                f"padding_side must be 'left' or 'right', got {padding_side}"
            )
        mask = mask[
            :, :, None, None
        ]  # [batch_size, seq_len] --> [batch_size, seq_len, 1, 1]

        # Compute masked mean over non-padding positions
        masked_ths = ths.masked_fill(~mask, 0.0)
        num_valid_positions = (sl * hidden_dim).view(batch_size, 1, 1, 1)
        masked_mean = masked_ths.sum(dim=(1, 2), keepdim=True) / (
            num_valid_positions + eps
        )

        # Compute min/max over non-padding positions
        x_min = ths.masked_fill(~mask, float("inf")).amin(
            dim=(1, 2), keepdim=True
        )
        x_max = ths.masked_fill(~mask, float("-inf")).amax(
            dim=(1, 2), keepdim=True
        )

        # Normalization
        normalized = (ths - masked_mean) / (x_max - x_min + eps)
        normalized = normalized * scale_factor

        # Pack the hidden states to a 3D tensor (batch_size, seq_len, hidden_dim * num_layers)
        normalized = normalized.flatten(2)
        mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
        normalized = normalized.masked_fill(~mask_flat, 0.0)
        normalized = normalized.to(dtype=original_dtype)
        return Tensor.from_dlpack(normalized.contiguous())

    @staticmethod
    def _left_align_text_embeds(
        prompt_embeds: Tensor,
        prompt_valid_length: Tensor,
    ) -> Tensor:
        """Shift valid tokens from the right end to the left end of the sequence.

        The tokenizer uses left-padding, so valid tokens sit at positions
        ``[L - valid_length, L)``.  The connector expects them at ``[0, valid_length)``
        (matching diffusers' ``flipped_mask`` logic). This method performs that shift
        entirely in PyTorch without a CPU round-trip.

        Args:
            prompt_embeds: ``[B, L, D]`` text embeddings with content at right end.
            prompt_valid_length: ``[B]`` int32 count of valid tokens per batch item.

        Returns:
            ``[B, L, D]`` tensor with content at ``[0, valid_length)`` and zeros
            at ``[valid_length, L)``.
        """
        import torch

        pe = torch.from_dlpack(prompt_embeds)
        vl = torch.from_dlpack(prompt_valid_length)  # [B] int32
        B, L, _D = pe.shape
        out = torch.zeros_like(pe)
        for b in range(B):
            n = int(vl[b].item())
            if n > 0:
                out[b, :n] = pe[b, L - n :]
        return Tensor.from_dlpack(out.contiguous())

    @staticmethod
    def _pack_latents(
        latents: Tensor, patch_size: int = 1, patch_size_t: int = 1
    ) -> Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, _num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            (
                batch_size,
                -1,
                post_patch_num_frames,
                patch_size_t,
                post_patch_height,
                patch_size,
                post_patch_width,
                patch_size,
            )
        )
        latents = latents.permute([0, 2, 4, 6, 1, 3, 5, 7])
        latents = latents.rebind(
            (
                batch_size,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                _num_channels,
                patch_size_t,
                patch_size,
                patch_size,
            )
        )
        # Flatten (F_post, H_post, W_post) -> S
        # Indices: 0(B), 1(F), 2(H), 3(W), 4(C), 5(pt), 6(p), 7(p)
        latents = latents.flatten(1, 3)
        # Flatten (C, pt, p, p) -> D
        # Indices after first flatten: 0(B), 1(S), 2(C), 3(pt), 4(p), 5(p)
        latents = latents.flatten(2, 5)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.shape[0]
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size

        latents = latents.reshape(
            (
                batch_size,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                -1,
                patch_size_t,
                patch_size,
                patch_size,
            )
        )
        _num_channels = latents.shape[4]
        latents = latents.permute((0, 4, 1, 5, 2, 6, 3, 7))
        latents = latents.rebind(
            (
                batch_size,
                _num_channels,
                post_patch_num_frames,
                patch_size_t,
                post_patch_height,
                patch_size,
                post_patch_width,
                patch_size,
            )
        )
        # Flatten (F_post, pt) -> F
        # Indices: 0(B), 1(C), 2(F), 3(pt), 4(H), 5(p), 6(W), 7(p)
        latents = latents.flatten(2, 3)
        # Flatten (H_post, p) -> H
        # Indices: 0(B), 1(C), 2(F), 3(H), 4(p), 5(W), 6(p)
        latents = latents.flatten(3, 4)
        # Flatten (W_post, p) -> W
        # Indices: 0(B), 1(C), 2(F), 3(H), 4(W), 5(p)
        latents = latents.flatten(4, 5)
        return latents

    @staticmethod
    def _pack_audio_latents(
        latents: Tensor,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> Tensor:
        # Audio latents shape: [B, C, L, M], where L is the latent audio length and M is the number of mel bins
        if patch_size is not None and patch_size_t is not None:
            # Packs the latents into a patch sequence of shape [B, L // p_t * M // p, C * p_t * p] (a ndim=3 tnesor).
            # dim=1 is the effective audio sequence length and dim=2 is the effective audio input feature size.
            batch_size, _num_channels, latent_length, latent_mel_bins = (
                latents.shape
            )
            post_patch_latent_length = latent_length // patch_size_t
            post_patch_mel_bins = latent_mel_bins // patch_size
            latents = latents.reshape(
                (
                    batch_size,
                    -1,
                    post_patch_latent_length,
                    patch_size_t,
                    post_patch_mel_bins,
                    patch_size,
                )
            )
            latents = latents.permute((0, 2, 4, 1, 3, 5))
            latents = latents.rebind(
                (
                    batch_size,
                    post_patch_latent_length,
                    post_patch_mel_bins,
                    _num_channels,
                    patch_size_t,
                    patch_size,
                )
            )
            # Flatten (L_post, M_post) -> S
            # Indices: 0(B), 1(L), 2(M), 3(C), 4(pt), 5(p)
            latents = latents.flatten(1, 2)
            # Flatten (C, pt, p) -> D
            # Indices: 0(B), 1(S), 2(C), 3(pt), 4(p)
            latents = latents.flatten(2, 4)
        else:
            # Packs the latents into a patch sequence of shape [B, L, C * M]. This implicitly assumes a (mel)
            # patch_size of M (all mel bins constitutes a single patch) and a patch_size_t of 1.
            latents = F.flatten(
                latents.transpose(1, 2), 2, 3
            )  # [B, C, L, M] --> [B, L, C * M]
        return latents

    @staticmethod
    def _unpack_audio_latents(
        latents: Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> Tensor:
        # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor of shape [B, C, L, M],
        # where L is the latent audio length and M is the number of mel bins.
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.shape[0]
            post_patch_latent_length = latent_length // patch_size_t
            post_patch_mel_bins = num_mel_bins // patch_size
            latents = latents.reshape(
                (
                    batch_size,
                    post_patch_latent_length,
                    post_patch_mel_bins,
                    -1,
                    patch_size_t,
                    patch_size,
                )
            )
            _num_channels = latents.shape[3]
            latents = latents.permute((0, 3, 1, 4, 2, 5))
            latents = latents.rebind(
                (
                    batch_size,
                    _num_channels,
                    post_patch_latent_length,
                    patch_size_t,
                    post_patch_mel_bins,
                    patch_size,
                )
            )
            # Flatten (L_post, pt) -> L
            # Indices: 0(B), 1(C), 2(L), 3(pt), 4(M), 5(p)
            latents = latents.flatten(2, 3)
            # Flatten (M_post, p) -> M
            # Indices: 0(B), 1(C), 2(L), 3(M), 4(p)
            latents = latents.flatten(3, 4)
        else:
            # Assume [B, S, D] = [B, L, C * M], which implies that patch_size = M and patch_size_t = 1.
            latents = latents.reshape(
                (latents.shape[0], latents.shape[1], -1, num_mel_bins)
            ).transpose(1, 2)
        return latents

    # -----------------------------------------------------------------------
    # Compiled instance methods (overridden in __dict__ by build_* at startup)
    # -----------------------------------------------------------------------

    def prepare_cfg_latents_timesteps(
        self, video_latents: Tensor, audio_latents: Tensor, timesteps: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Concat+cast video and audio latents for CFG [1,S,D]->[2,S,D].

        Compiled via build_prepare_cfg_latents_timesteps. Called once per denoising step
        when do_classifier_free_guidance is True.
        """
        dtype = self.transformer.config.dtype
        video = F.concat([video_latents, video_latents], axis=0).cast(dtype)
        audio = F.concat([audio_latents, audio_latents], axis=0).cast(dtype)
        timesteps = F.concat([timesteps, timesteps], axis=0)
        return video, audio, timesteps

    def apply_cfg_guidance(
        self,
        noise_pred_video: Tensor,
        noise_pred_audio: Tensor,
        guidance_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply classifier-free guidance formula to video and audio noise preds.

        Compiled via build_apply_cfg_guidance. Fuses cast + split + arithmetic
        for both modalities into a single compiled graph, called once per step.

        guidance: uncond + scale * (cond - uncond)
        """
        dtype = DType.float32
        scale = guidance_scale.cast(dtype)

        video = noise_pred_video.cast(dtype)
        v_uncond = F.slice_tensor(video, [slice(0, 1)])
        v_cond = F.slice_tensor(video, [slice(1, 2)])
        guided_video = v_uncond + scale * (v_cond - v_uncond)

        audio = noise_pred_audio.cast(dtype)
        a_uncond = F.slice_tensor(audio, [slice(0, 1)])
        a_cond = F.slice_tensor(audio, [slice(1, 2)])
        guided_audio = a_uncond + scale * (a_cond - a_uncond)

        return guided_video, guided_audio

    def _pack_video_latents(self, latents: Tensor) -> Tensor:
        """Cast and pack video latents [B,C,F,H,W] -> [B,S,C] (patch_size=1).

        With patch_size=patch_size_t=1 the pack is a simple permute + flatten:
        [B,C,F,H,W] -> [B,F,H,W,C] -> [B,F*H*W,C].
        """
        latents = latents.cast(self.transformer.config.dtype)
        # [B,C,F,H,W] -> [B,F,H,W,C]
        latents = latents.permute([0, 2, 3, 4, 1])
        # [B,F,H,W,C] -> [B,S,C]  where S = F*H*W
        latents = latents.flatten(1, 3)
        return latents

    def _pack_audio_latents_packed(self, latents: Tensor) -> Tensor:
        """Cast and pack audio latents [B,C,L,M] -> [B,L,C*M] (no spatial patch).

        Equivalent to the no-patch branch of _pack_audio_latents:
        [B,C,L,M] -> [B,L,C,M] -> [B,L,C*M].
        """
        latents = latents.cast(self.transformer.config.dtype)
        # [B,C,L,M] -> [B,L,C,M]
        latents = latents.permute((0, 2, 1, 3))
        # [B,L,C,M] -> [B,L,C*M]
        latents = latents.flatten(2, 3)
        return latents

    def prepare_scheduler(self, sigmas: Tensor) -> tuple[Tensor, Tensor]:
        """Pre-compute timesteps and per-step dt values from sigmas.

        Returns:
            (all_timesteps, all_dts) where timesteps = sigmas[:-1] as float32,
            and dts = sigmas[1:] - sigmas[:-1] (float32).
        """
        sigmas_curr = F.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = F.slice_tensor(sigmas, [slice(1, None)])
        all_dt = F.sub(sigmas_next, sigmas_curr)
        # Transformer timestep embedding is trained on [0, 1000] range
        # (num_train_timesteps=1000), matching diffusers which passes
        # sigmas * 1000 as the timestep. dt stays in [0,1] sigma scale
        # for the Euler update: latents += dt * noise_pred.
        all_timesteps = (sigmas_curr * 1000.0).cast(
            self.transformer.config.dtype
        )
        return all_timesteps, all_dt

    def scheduler_step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        dt: Tensor,
        num_noise_tokens: int,
    ) -> Tensor:
        """Apply a single Euler update step in sigma space."""
        latents_sliced = F.slice_tensor(
            latents,
            [
                slice(None),
                (slice(0, num_noise_tokens), "num_tokens"),
                slice(None),
            ],
        )
        noise_pred_sliced = F.slice_tensor(
            noise_pred,
            [
                slice(None),
                (slice(0, num_noise_tokens), "num_tokens"),
                slice(None),
            ],
        )
        latents_dtype = latents_sliced.dtype
        latents_sliced = latents_sliced.cast(DType.float32)
        latents_sliced = latents_sliced + dt * noise_pred_sliced
        return latents_sliced.cast(latents_dtype)

    def _postprocess_video_latents(
        self,
        latents: Tensor,
        latents_mean: Tensor,
        latents_std: Tensor,
    ) -> Tensor:
        """Denormalize video latents [B,C,F,H,W] using per-channel stats.

        Mirrors Flux2's _postprocess_latents: the compiled inner step of
        decode_video_latents.
        """
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        c = latents_mean.shape[0]
        mean_r = F.reshape(latents_mean.cast(latents.dtype), (1, c, 1, 1, 1))
        std_r = F.reshape(latents_std.cast(latents.dtype), (1, c, 1, 1, 1))
        return latents * std_r / scaling_factor + mean_r

    def _postprocess_audio_latents(
        self,
        latents: Tensor,
        latents_mean: Tensor,
        latents_std: Tensor,
    ) -> Tensor:
        """Denormalize packed audio latents [B,L,C*M] using per-channel stats.

        Operates on the packed tensor (before unpack), matching diffusers order:
          _denormalize_audio_latents -> _unpack_audio_latents
        stats shape [C*M] broadcasts against last dim of [B, L, C*M] directly.
        """
        return latents * latents_std.cast(latents.dtype) + latents_mean.cast(
            latents.dtype
        )

    def decode_video_latents(
        self,
        latents: Tensor,
        num_frames: int,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        """Decode packed video latents into a float32 [B,F,H,W,C] NumPy array.

        Mirrors Flux2's decode_latents: shape-dependent unpack (not compiled) is
        performed here, then the compiled _postprocess_video_latents handles
        denormalization, the VAE decoder runs, and finally the pixels are scaled
        to [0, 1] and permuted to channel-last.

        Args:
            latents: Packed latents [B, S, C].
            num_frames: Latent frame count.
            height: Latent height.
            width: Latent width.
            output_type: "latent" returns latents as-is; otherwise decodes to NumPy.

        Returns:
            float32 NumPy array [B, F, H, W, C] or raw latent Tensor.
        """
        if output_type == "latent":
            return latents
        # Shape-dependent unpack: [B,S,C] -> [B,C,F,H,W]  (not compiled).
        latents = self._unpack_latents(latents, num_frames, height, width)
        # Compiled postprocess: denormalize using per-channel stats.
        if (
            self._vae_latents_mean is not None
            and self._vae_latents_std is not None
        ):
            latents = self._postprocess_video_latents(
                latents, self._vae_latents_mean, self._vae_latents_std
            )
        else:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latents = latents / scaling_factor
        # VAE decode: [B,C,F,H,W].
        video = self.vae.decode(latents.cast(DType.bfloat16))
        # Scale pixels to [0, 1] and permute to channel-last [B,F,H,W,C].
        video = (video / 2.0 + 0.5).clip(min=0.0, max=1.0)
        video = video.permute((0, 2, 3, 4, 1))
        return self._to_numpy(video)

    def decode_audio_latents(
        self,
        latents: Tensor,
        audio_num_frames: int,
        latent_mel_bins: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        """Decode packed audio latents into a waveform Tensor (or return latents).

        Args:
            latents: Packed audio latents [B, L, C*M].
            audio_num_frames: Latent audio length used for unpacking.
            latent_mel_bins: Number of mel frequency bins in the latent space.
            output_type: "latent" returns latents as-is; otherwise decodes via vocoder.

        Returns:
            Waveform Tensor or raw latent Tensor.
        """
        if output_type == "latent":
            return latents
        # Denormalize on the PACKED latent [B,L,C*M] before unpack — matches diffusers:
        #   _denormalize_audio_latents(packed) -> _unpack_audio_latents
        # stats [C*M=128] broadcast against last dim of [B,L,128] directly.
        if (
            self._audio_latents_mean is not None
            and self._audio_latents_std is not None
        ):
            latents = self._postprocess_audio_latents(
                latents, self._audio_latents_mean, self._audio_latents_std
            )
        # Shape-dependent unpack: [B,L,C*M] -> [B,C,L,M]  (not compiled).
        latents = self._unpack_audio_latents(
            latents, audio_num_frames, latent_mel_bins
        )
        mel_spectrograms = self.audio_vae.decode(latents.cast(DType.bfloat16))
        _mel = np.from_dlpack(mel_spectrograms.cast(DType.float32).to(CPU()))
        print(
            f"[AUDIO] mel: shape={_mel.shape}, mean={_mel.mean():.4f}, std={_mel.std():.4f}, min={_mel.min():.4f}, max={_mel.max():.4f}",
            flush=True,
        )
        # Vocoder compiled as float32 (cuDNN conv_transpose hardcodes CUDNN_DATA_FLOAT).
        result = self.vocoder(mel_spectrograms.cast(DType.float32))
        _wav = np.from_dlpack(result.cast(DType.float32).to(CPU()))
        print(
            f"[AUDIO] wav: shape={_wav.shape}, mean={_wav.mean():.4f}, std={_wav.std():.4f}, min={_wav.min():.4f}, max={_wav.max():.4f}",
            flush=True,
        )
        return self._to_numpy(result)

    def _to_numpy(self, image: Tensor) -> np.ndarray:
        cpu_video: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_video)

    def execute(  # type: ignore[override]
        self,
        model_inputs: LTX2ModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent"] = "np",
    ) -> LTX2PipelineOutput:
        r"""
        Executes the LTX2 model with the prepared inputs.

        Args:
            model_inputs: Inputs containing tokens, latents, timesteps, sigmas, and IDs.
            callback_queue: Optional queue for streaming intermediate decoded outputs.
            output_type: Output mode ("np", "latent")

        Returns:
            LTX2PipelineOutput containing one output per batch element.
        """
        num_inference_steps = model_inputs.num_inference_steps
        guidance_scale = model_inputs.guidance_scale
        device = self.devices[0]

        extra_params = model_inputs.extra_params or {}
        video_latents_np: np.ndarray = model_inputs.latents
        audio_latents_np: np.ndarray | None = extra_params.get("audio_latents")
        video_coords_np: np.ndarray | None = extra_params.get("video_coords")
        audio_coords_np: np.ndarray | None = extra_params.get("audio_coords")

        latent_num_frames = int(extra_params["latent_num_frames"])
        latent_height = int(extra_params["latent_height"])
        latent_width = int(extra_params["latent_width"])

        video_seq_len = latent_num_frames * latent_height * latent_width
        num_inference_steps = model_inputs.num_inference_steps
        sigmas_key = f"{num_inference_steps}_{video_seq_len}"
        if sigmas_key in self._cached_sigmas:
            sigmas = self._cached_sigmas[sigmas_key]
        else:
            sigmas = Tensor.from_dlpack(model_inputs.sigmas).to(device)
            self._cached_sigmas[sigmas_key] = sigmas
        all_timesteps, all_dts = self.prepare_scheduler(sigmas)

        audio_seq_len = int(audio_latents_np.shape[1])
        num_inference_steps = model_inputs.num_inference_steps
        sigmas_key = f"{num_inference_steps}_{audio_seq_len}"
        if sigmas_key in self._cached_sigmas:
            sigmas = self._cached_sigmas[sigmas_key]
        else:
            sigmas = Tensor.from_dlpack(model_inputs.sigmas).to(device)
            self._cached_sigmas[sigmas_key] = sigmas
        all_timesteps, all_dts = self.prepare_scheduler(sigmas)

        # For faster tensor slicing inside the denoising loop.
        timesteps_seq = all_timesteps
        dts_seq = all_dts
        if hasattr(timesteps_seq, "driver_tensor"):
            timesteps_seq = timesteps_seq.driver_tensor
        if hasattr(dts_seq, "driver_tensor"):
            dts_seq = dts_seq.driver_tensor

        # 3. Encode text with Gemma3 (via transformers) using TokenBuffer data
        token_ids_np: np.ndarray = model_inputs.tokens.array
        if token_ids_np.ndim == 1:
            token_ids_np = np.expand_dims(token_ids_np, axis=0)

        mask_np: np.ndarray | None = model_inputs.mask
        if mask_np is None:
            mask_np = np.ones_like(token_ids_np, dtype=np.bool_)
        if mask_np.ndim == 1:
            mask_np = np.expand_dims(mask_np, axis=0)
        mask = Tensor.from_dlpack(mask_np).to(device)

        text_encoder_hidden_states = self._encode_tokens(
            token_ids_np, mask, False
        )
        # Reduce [B, seq_len] bool mask → [B] integer counts, matching
        sequence_lengths = extra_params.get("ltx2_valid_length")[-1]
        prompt_valid_length = (
            Tensor.from_dlpack(
                sequence_lengths,
            )
            .to(device)
            .cast(DType.int32)
        )
        prompt_embeds = self._pack_text_embeds(
            text_encoder_hidden_states, prompt_valid_length, device
        )
        do_classifier_free_guidance = model_inputs.guidance_scale > 1.0
        if do_classifier_free_guidance:
            if model_inputs.negative_tokens is not None:
                negative_ids_np: np.ndarray = model_inputs.negative_tokens.array
                if negative_ids_np.ndim == 1:
                    negative_ids_np = np.expand_dims(negative_ids_np, axis=0)
                mask_neg_np: np.ndarray | None = extra_params.get(
                    "ltx2_attn_mask_neg"
                )
                if mask_neg_np is None:
                    mask_neg_np = np.ones_like(negative_ids_np, dtype=np.bool_)
                if mask_neg_np.ndim == 1:
                    mask_neg_np = np.expand_dims(mask_neg_np, axis=0)
                mask_neg = Tensor.from_dlpack(mask_neg_np).to(device)

                negative_hidden_states = self._encode_tokens(
                    negative_ids_np, mask_neg
                )
                negative_sequence_lengths = extra_params.get(
                    "ltx2_valid_length"
                )[0]
                negative_prompt_valid_length = (
                    Tensor.from_dlpack(
                        negative_sequence_lengths,
                    )
                    .to(device)
                    .cast(DType.int32)
                )
                negative_prompt_embeds = self._pack_text_embeds(
                    negative_hidden_states,
                    negative_prompt_valid_length,
                    device,
                )
            else:
                # Use zeros for negative prompt if not provided
                negative_prompt_embeds = Tensor.zeros_like(prompt_embeds)
                negative_prompt_valid_length = Tensor.zeros_like(
                    prompt_valid_length
                )

            prompt_embeds = F.concat(
                [negative_prompt_embeds, prompt_embeds], axis=0
            )
            prompt_valid_length = F.concat(
                [negative_prompt_valid_length, prompt_valid_length], axis=0
            )

        # Left-align prompt_embeds to match diffusers' connector expectation.
        # _pack_text_embeds uses left-padding, so valid tokens are at the RIGHT end
        # of the seq-len dimension (positions [L-vl, L)).  The connector's 1D RoPE
        # is trained with content at the LEFT end (positions [0, vl)), so we shift
        # each batch item's valid tokens to the start of the sequence.
        prompt_embeds = self._left_align_text_embeds(
            prompt_embeds, prompt_valid_length
        )

        (
            connector_prompt_embeds,
            connector_audio_prompt_embeds,
            _,
        ) = self.connectors(prompt_embeds, prompt_valid_length)

        batch_size = video_latents_np.shape[0]
        batch_size = int(batch_size)

        video_latents = Tensor.from_dlpack(video_latents_np).to(device)
        # Pack video_latents: [B, C, F, H, W] -> [B, S, D]
        video_latents = self._pack_video_latents(video_latents)

        if audio_latents_np is None:
            raise ValueError(
                "ltx2_audio_latents is missing from extra_params; "
                "ensure the tokenizer sets 'ltx2_audio_latents' for LTX-2."
            )
        if audio_latents_np.ndim != 4:
            raise ValueError("ltx2_audio_latents must have shape [B, C, L, M]")
        (
            batch_size_audio,
            _audio_channels,
            audio_num_frames,
            latent_mel_bins,
        ) = audio_latents_np.shape
        if batch_size_audio != batch_size:
            raise ValueError(
                "Mismatch between video and audio batch sizes in LTX2 latents"
            )
        if "ltx2_audio_num_frames" in extra_params:
            audio_num_frames = int(extra_params["ltx2_audio_num_frames"])
            latent_mel_bins = int(extra_params["ltx2_latent_mel_bins"])
        audio_latents_arr = audio_latents_np.astype(np.float32)

        audio_latents = Tensor.from_dlpack(audio_latents_arr).to(device)
        audio_latents = self._pack_audio_latents_packed(audio_latents)

        video_coords = Tensor.from_dlpack(video_coords_np).to(device)

        audio_coords_np_f32 = audio_coords_np.astype(np.float32)
        audio_coords = Tensor.from_dlpack(audio_coords_np_f32).to(device)

        guidance_scale_tensor = Tensor.from_dlpack(
            np.array([guidance_scale], dtype=np.float32),
        ).to(device)

        num_noise_tokens = int(video_latents.shape[1])

        for i in tqdm(range(num_inference_steps), desc="Denoising"):
            timestep = timesteps_seq[i : i + 1]
            dt = dts_seq[i : i + 1]

            if do_classifier_free_guidance:
                latent_model_input, audio_latent_model_input, timestep = (
                    self.prepare_cfg_latents_timesteps(
                        video_latents, audio_latents, timestep
                    )
                )
            else:
                latent_model_input = video_latents
                audio_latent_model_input = audio_latents

            latent_model_input = latent_model_input.cast(prompt_embeds.dtype)
            audio_latent_model_input = audio_latent_model_input.cast(
                prompt_embeds.dtype
            )

            noise_pred_video, noise_pred_audio = self.transformer(
                latent_model_input,
                audio_latent_model_input,
                connector_prompt_embeds,
                connector_audio_prompt_embeds,
                timestep,
                timestep,
                video_coords,
                audio_coords,
            )

            if do_classifier_free_guidance:
                noise_pred_video, noise_pred_audio = self.apply_cfg_guidance(
                    noise_pred_video, noise_pred_audio, guidance_scale_tensor
                )

            with Tracer("scheduler_steps"):
                video_latents = self.scheduler_step_video(
                    video_latents, noise_pred_video, dt, num_noise_tokens
                )
                audio_latents = self.scheduler_step_audio(
                    audio_latents, noise_pred_audio, dt, num_noise_tokens
                )

            if callback_queue is not None:
                if hasattr(device, "synchronize"):
                    device.synchronize()
                callback_queue.put_nowait(
                    cast(
                        np.ndarray,
                        self.decode_video_latents(
                            video_latents,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            output_type,
                        ),
                    )
                )
                if hasattr(device, "synchronize"):
                    device.synchronize()
                callback_queue.put_nowait(
                    cast(
                        np.ndarray,
                        self.decode_audio_latents(
                            audio_latents,
                            audio_num_frames,
                            latent_mel_bins,
                            output_type,
                        ),
                    )
                )

        # 5) Decode final outputs per batch element.
        video_list = []
        with Tracer("decode_video_latents"):
            for b in range(batch_size):
                with Tracer("slice_batch"):
                    video_latents_b = video_latents[b : b + 1]
                video_list.append(
                    self.decode_video_latents(
                        video_latents_b,
                        latent_num_frames,
                        latent_height,
                        latent_width,
                        output_type,
                    )
                )

        audio_list = []
        with Tracer("decode_audio_latents"):
            for b in range(batch_size):
                with Tracer("slice_batch"):
                    audio_latents_b = audio_latents[b : b + 1]
                audio_list.append(
                    self.decode_audio_latents(
                        audio_latents_b,
                        audio_num_frames,
                        latent_mel_bins,
                        output_type,
                    )
                )

        return LTX2PipelineOutput(video_list, audio_list)
