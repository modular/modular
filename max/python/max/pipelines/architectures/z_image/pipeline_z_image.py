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
"""Z-Image diffusion pipeline (Graph API / ModuleV2)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    DiffusionPipelineOutput,
)
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from max.pipelines.lib.utils import BoundedCache
from max.profiler import Tracer, traced

from ..autoencoders import AutoencoderKLModel
from ..qwen3.text_encoder import Qwen3TextEncoderZImageModel
from .model import ZImageTransformerModel

_DEVICE_TENSOR_FIELDS = frozenset(
    {
        "tokens_tensor",
        "negative_tokens_tensor",
        "txt_ids_tensor",
        "img_ids_tensor",
        "negative_txt_ids_tensor",
        "negative_img_ids_tensor",
        "input_image_tensor",
        "latents_tensor",
        "sigmas_tensor",
        "h_carrier",
        "w_carrier",
    }
)


def _validate_z_image_context(context: PixelContext) -> None:
    """Fail fast before device uploads."""
    if context.latents.size == 0:
        raise ValueError(
            "ZImagePipeline requires non-empty latents in PixelContext."
        )
    for name in ("latent_image_ids", "sigmas", "timesteps"):
        if not hasattr(context, name):
            raise TypeError(
                f"ZImagePipeline requires PixelContext with attribute {name!r}; "
                f"{type(context).__name__} has no {name!r}."
            )
        arr = getattr(context, name)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            raise ValueError(
                f"ZImagePipeline requires non-empty {name} in PixelContext."
            )


@dataclass(kw_only=True)
class ZImageModelInputs:
    """Z-Image execution inputs with device tensors and host metadata."""

    tokens: TokenBuffer
    tokens_2: TokenBuffer | None = None
    negative_tokens: TokenBuffer | None = None
    negative_tokens_2: TokenBuffer | None = None
    timesteps: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    sigmas: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    latents: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    latent_image_ids: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    guidance: npt.NDArray[np.float32] | None = None
    true_cfg_scale: float = 1.0
    num_warmup_steps: int = 0
    input_image: npt.NDArray[np.uint8] | None = None
    strength: float = 0.6
    cfg_normalization: bool = False
    cfg_truncation: float = 1.0
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    explicit_negative_prompt: bool = False
    do_cfg: bool = False
    tokens_tensor: Buffer
    negative_tokens_tensor: Buffer | None = None
    txt_ids_tensor: Buffer
    img_ids_tensor: Buffer
    negative_txt_ids_tensor: Buffer | None = None
    negative_img_ids_tensor: Buffer | None = None
    input_image_tensor: Buffer | None = None
    latents_tensor: Buffer
    sigmas_tensor: Buffer
    h_carrier: Buffer
    w_carrier: Buffer

    @classmethod
    def kwargs_from_context(cls, context: PixelContext) -> dict[str, Any]:
        """Build kwargs for all fields except device tensors."""
        kwargs: dict[str, Any] = {}
        for dataclass_field in fields(cls):
            name = dataclass_field.name
            if name in _DEVICE_TENSOR_FIELDS:
                continue
            if not hasattr(context, name):
                continue
            value = getattr(context, name)
            if value is None:
                if dataclass_field.default is not MISSING:
                    kwargs[name] = dataclass_field.default
                elif dataclass_field.default_factory is not MISSING:
                    kwargs[name] = dataclass_field.default_factory()
                else:
                    kwargs[name] = None
            else:
                kwargs[name] = value
        return kwargs

    def __post_init__(self) -> None:
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(
                f"height must be a positive int. Got {self.height!r}"
            )
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(
                f"width must be a positive int. Got {self.width!r}"
            )
        if (
            not isinstance(self.num_inference_steps, int)
            or self.num_inference_steps <= 0
        ):
            raise ValueError(
                "num_inference_steps must be a positive int. "
                f"Got {self.num_inference_steps!r}"
            )
        if (
            not isinstance(self.num_images_per_prompt, int)
            or self.num_images_per_prompt <= 0
        ):
            raise ValueError(
                "num_images_per_prompt must be > 0. "
                f"Got {self.num_images_per_prompt!r}"
            )
        if self.sigmas.size == 0:
            raise ValueError(
                "ZImagePipeline requires non-empty sigmas in context."
            )
        if self.latent_image_ids.size == 0:
            raise ValueError(
                "ZImagePipeline requires non-empty latent image ids in context."
            )


class ZImagePipeline(DiffusionPipeline):
    """Diffusion pipeline for Z-Image generation (Graph API)."""

    unprefixed_weight_component = "transformer"
    default_num_inference_steps = 50
    default_residual_threshold = 0.06

    vae: AutoencoderKLModel
    text_encoder: Qwen3TextEncoderZImageModel
    transformer: ZImageTransformerModel

    components = {
        "vae": AutoencoderKLModel,
        "text_encoder": Qwen3TextEncoderZImageModel,
        "transformer": ZImageTransformerModel,
    }

    @traced(message="ZImagePipeline.init_remaining_components")
    def init_remaining_components(self) -> None:
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

        self.build_preprocess_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step()
        self.build_decode_latents()
        self.build_cfg_combine()
        self.build_duplicate_batch()
        self.build_cfg_finalize_batched()
        self.build_cfg_renormalization()
        self.build_postprocess_image()
        self.build_pad_seq()
        self.build_truncate_seq()
        self.build_concat_batch()

        self._cached_text_ids: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_sigmas: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_img_ids: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_img_ids_base_np: BoundedCache[str, np.ndarray] = (
            BoundedCache(32)
        )
        self._cached_shape_carriers: BoundedCache[int, Buffer] = BoundedCache(
            32
        )
        self._cached_prompt_token_tensors: BoundedCache[str, Buffer] = (
            BoundedCache(32)
        )
        self._cached_prompt_padding: BoundedCache[str, Buffer] = BoundedCache(
            32
        )
        self._cached_guidance: BoundedCache[str, Buffer] = BoundedCache(32)

    @traced(message="ZImagePipeline.build_preprocess_latents")
    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        target_dtype = self.transformer.config.dtype

        def _graph(latents: TensorValue) -> TensorValue:
            batch = latents.shape[0]
            c = latents.shape[1]
            h = latents.shape[2]
            w = latents.shape[3]
            latents = ops.rebind(
                latents, [batch, c, (h // 2) * 2, (w // 2) * 2]
            )
            latents = ops.reshape(latents, (batch, c, h // 2, 2, w // 2, 2))
            latents = ops.permute(latents, [0, 2, 4, 3, 5, 1])
            latents = ops.reshape(latents, (batch, (h // 2) * (w // 2), c * 4))
            return ops.cast(latents, target_dtype)

        self._patchify_and_pack = cast(
            Callable[[Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        DType.float32,
                        shape=["batch", "channels", "height", "width"],
                        device=device,
                    ),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_prepare_scheduler")
    def build_prepare_scheduler(self) -> None:
        device = self.transformer.devices[0]

        def _graph(
            timesteps: TensorValue, sigmas: TensorValue
        ) -> tuple[TensorValue, TensorValue]:
            all_timesteps = 1.0 - timesteps
            sigmas_curr = ops.slice_tensor(sigmas, [slice(0, -1)])
            sigmas_next = ops.slice_tensor(sigmas, [slice(1, None)])
            all_dt = sigmas_next - sigmas_curr
            return all_timesteps, all_dt

        self._prepare_scheduler = cast(
            Callable[[Buffer, Buffer], tuple[Buffer, Buffer]],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        DType.float32,
                        shape=["num_timesteps"],
                        device=device,
                    ),
                    TensorType(
                        DType.float32,
                        shape=["num_sigmas"],
                        device=device,
                    ),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_scheduler_step")
    def build_scheduler_step(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(
            latents: TensorValue, noise_pred: TensorValue, dt: TensorValue
        ) -> TensorValue:
            latents_dtype = latents.dtype
            latents = ops.cast(latents, DType.float32)
            latents = latents - dt * noise_pred
            return ops.cast(latents, latents_dtype)

        self._scheduler_step = cast(
            Callable[[Buffer, Buffer, Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(DType.float32, shape=[1], device=device),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_decode_latents")
    def build_decode_latents(self) -> None:
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype
        scaling = float(self.vae.config.scaling_factor)
        shift = float(self.vae.config.shift_factor or 0.0)

        def _unpack(
            latents: TensorValue,
            h_carrier: TensorValue,
            w_carrier: TensorValue,
        ) -> TensorValue:
            batch = latents.shape[0]
            ch = latents.shape[2]
            half_h = h_carrier.shape[0]
            half_w = w_carrier.shape[0]
            latents = ops.rebind(latents, [batch, half_h * half_w, ch])
            latents = ops.reshape(latents, (batch, half_h, half_w, ch))
            latents = ops.rebind(
                latents, [batch, half_h, half_w, (ch // 4) * 4]
            )
            latents = ops.reshape(
                latents, (batch, half_h, half_w, 2, 2, ch // 4)
            )
            latents = ops.permute(latents, [0, 5, 1, 3, 2, 4])
            latents = ops.reshape(
                latents, (batch, ch // 4, half_h * 2, half_w * 2)
            )
            return (latents / scaling) + shift

        self._unpack_and_postprocess = cast(
            Callable[[Buffer, Buffer, Buffer], Buffer],
            max_compile(
                _unpack,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", "seq", "channels"],
                        device=device,
                    ),
                    TensorType(DType.float32, shape=["half_h"], device=device),
                    TensorType(DType.float32, shape=["half_w"], device=device),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_postprocess_image")
    def build_postprocess_image(self) -> None:
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype

        def _graph(image: TensorValue) -> TensorValue:
            image = ops.cast(image, DType.float32)
            image = image * 0.5 + 0.5
            image = ops.where(image < 0.0, 0.0, image)
            image = ops.where(image > 1.0, 1.0, image)
            image = ops.permute(image, [0, 2, 3, 1])
            image = image * 255.0
            return ops.cast(image, DType.uint8)

        self._postprocess_image = cast(
            Callable[[Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", "channels", "height", "width"],
                        device=device,
                    ),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_cfg_combine")
    def build_cfg_combine(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(
            pos: TensorValue, neg: TensorValue, scale: TensorValue
        ) -> TensorValue:
            result = pos + scale * (pos - neg)
            return ops.cast(result, pos.dtype)

        self._cfg_combine = cast(
            Callable[[Buffer, Buffer, Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(DType.float32, shape=[], device=device),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_cfg_renormalization")
    def build_cfg_renormalization(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(pos: TensorValue, pred: TensorValue) -> TensorValue:
            ori_norm = ops.sqrt(
                ops.sum(ops.sum(pos * pos, axis=2), axis=1) + 1e-12
            )
            new_norm = ops.sqrt(
                ops.sum(ops.sum(pred * pred, axis=2), axis=1) + 1e-12
            )
            while ori_norm.rank > 1:
                ori_norm = ops.squeeze(ori_norm, -1)
            while new_norm.rank > 1:
                new_norm = ops.squeeze(new_norm, -1)
            safe_new = ops.where(new_norm > 1e-12, new_norm, 1e-12)
            ratio = ori_norm / safe_new
            ratio = ops.where(new_norm > ori_norm, ratio, 1.0)
            ratio = ops.unsqueeze(ops.unsqueeze(ratio, 1), 2)
            return pred * ratio

        self._cfg_renormalization = cast(
            Callable[[Buffer, Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                    TensorType(
                        dtype, shape=["batch", "seq", "channels"], device=device
                    ),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_duplicate_batch")
    def build_duplicate_batch(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(x: TensorValue) -> TensorValue:
            batch = x.shape[0]
            seq = x.shape[1]
            ch = x.shape[2]
            x = ops.unsqueeze(x, 0)
            x = ops.broadcast_to(x, [2, batch, seq, ch])
            return ops.reshape(x, [batch * 2, seq, ch])

        self._duplicate_batch = cast(
            Callable[[Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", "seq", "channels"],
                        device=device,
                    ),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_cfg_finalize_batched")
    def build_cfg_finalize_batched(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                dtype,
                shape=["double_batch", "seq", "channels"],
                device=device,
            ),
            TensorType(DType.float32, shape=[], device=device),
        ]

        def _no_norm(
            pred_cfg: TensorValue, scale: TensorValue
        ) -> tuple[TensorValue, TensorValue]:
            batch2 = pred_cfg.shape[0]
            batch = batch2 // 2
            seq = pred_cfg.shape[1]
            ch = pred_cfg.shape[2]
            pos = ops.rebind(pred_cfg[:batch], [batch, seq, ch])
            neg = ops.rebind(pred_cfg[batch:], [batch, seq, ch])
            result = ops.cast(pos + scale * (pos - neg), pos.dtype)
            return pos, result

        def _with_norm(
            pred_cfg: TensorValue, scale: TensorValue
        ) -> tuple[TensorValue, TensorValue]:
            pos, result = _no_norm(pred_cfg, scale)
            ori = ops.sqrt(ops.sum(ops.sum(pos * pos, axis=2), axis=1) + 1e-12)
            new = ops.sqrt(
                ops.sum(ops.sum(result * result, axis=2), axis=1) + 1e-12
            )
            while ori.rank > 1:
                ori = ops.squeeze(ori, -1)
            while new.rank > 1:
                new = ops.squeeze(new, -1)
            safe = ops.where(new > 1e-12, new, 1e-12)
            ratio = ori / safe
            ratio = ops.where(new > ori, ratio, 1.0)
            ratio = ops.unsqueeze(ops.unsqueeze(ratio, 1), 2)
            return pos, result * ratio

        self._cfg_finalize_no_norm = cast(
            Callable[[Buffer, Buffer], tuple[Buffer, Buffer]],
            max_compile(_no_norm, input_types=input_types),
        )
        self._cfg_finalize_with_norm = cast(
            Callable[[Buffer, Buffer], tuple[Buffer, Buffer]],
            max_compile(_with_norm, input_types=input_types),
        )

    @traced(message="ZImagePipeline.build_pad_seq")
    def build_pad_seq(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(embeds: TensorValue, pad: TensorValue) -> TensorValue:
            return ops.concat([embeds, pad], axis=1)

        self._pad_seq = cast(
            Callable[[Buffer, Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", "seq_a", "hidden"],
                        device=device,
                    ),
                    TensorType(
                        dtype,
                        shape=["batch", "seq_b", "hidden"],
                        device=device,
                    ),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_truncate_seq")
    def build_truncate_seq(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(
            embeds: TensorValue, target_carrier: TensorValue
        ) -> TensorValue:
            target_len = target_carrier.shape[0]
            return embeds[:, :target_len, :]

        self._truncate_seq = cast(
            Callable[[Buffer, Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", "seq", "hidden"],
                        device=device,
                    ),
                    TensorType(
                        dtype,
                        shape=["target_len"],
                        device=device,
                    ),
                ],
            ),
        )

    @traced(message="ZImagePipeline.build_concat_batch")
    def build_concat_batch(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]

        def _graph(a: TensorValue, b: TensorValue) -> TensorValue:
            return ops.concat([a, b], axis=0)

        self._concat_batch = cast(
            Callable[[Buffer, Buffer], Buffer],
            max_compile(
                _graph,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch_a", "seq", "hidden"],
                        device=device,
                    ),
                    TensorType(
                        dtype,
                        shape=["batch_b", "seq", "hidden"],
                        device=device,
                    ),
                ],
            ),
        )

    def _align_prompt_embeds(
        self,
        neg_embeds: Buffer,
        pos_embeds: Buffer,
        device: Device,
    ) -> Buffer:
        pos_len = int(pos_embeds.shape[1])
        neg_len = int(neg_embeds.shape[1])
        hidden = int(pos_embeds.shape[2])

        if neg_len == pos_len:
            return neg_embeds
        if neg_len > pos_len:
            carrier = Buffer.from_dlpack(
                np.empty(pos_len, dtype=np.float32)
            ).to(device)
            return self._truncate_seq(neg_embeds, carrier)
        pad_len = pos_len - neg_len
        pad = Buffer.zeros(
            (1, pad_len, hidden), pos_embeds.dtype, device=device
        )
        return self._pad_seq(neg_embeds, pad)

    @traced(message="ZImagePipeline.prepare_inputs")
    def prepare_inputs(self, context: PixelContext) -> ZImageModelInputs:  # type: ignore[override]
        _validate_z_image_context(context)
        kwargs = ZImageModelInputs.kwargs_from_context(context)
        device = self.transformer.devices[0]
        text_device = self.text_encoder.devices[0]

        kwargs["latents"] = np.asarray(context.latents)
        kwargs["sigmas"] = np.asarray(context.sigmas)
        kwargs["latent_image_ids"] = np.asarray(context.latent_image_ids)

        latents_np = np.ascontiguousarray(kwargs["latents"])
        latent_h = int(latents_np.shape[-2])
        latent_w = int(latents_np.shape[-1])
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_seq_len = int(np.asarray(context.latent_image_ids).shape[-2])

        tokens_np = self._select_tokens_for_text_encoder(
            context.tokens.array, context.mask
        )
        tokens_buf = self._cache_token_buffer(tokens_np, text_device)
        txt_ids_buf, img_ids_buf = self._prepare_conditioning_ids(
            text_seq_len=int(tokens_np.shape[0]),
            image_seq_len=image_seq_len,
            latent_image_ids=np.asarray(context.latent_image_ids),
            height=int(context.height),
            width=int(context.width),
            device=device,
        )

        neg_tokens_buf: Buffer | None = None
        neg_txt_ids_buf: Buffer | None = None
        neg_img_ids_buf: Buffer | None = None
        if context.negative_tokens is not None:
            neg_np = self._select_tokens_for_text_encoder(
                context.negative_tokens.array, context.negative_mask
            )
            neg_tokens_buf = self._cache_token_buffer(neg_np, text_device)
            if context.explicit_negative_prompt:
                neg_txt_ids_buf, neg_img_ids_buf = (
                    self._prepare_conditioning_ids(
                        text_seq_len=int(neg_np.shape[0]),
                        image_seq_len=image_seq_len,
                        latent_image_ids=np.asarray(context.latent_image_ids),
                        height=int(context.height),
                        width=int(context.width),
                        device=device,
                    )
                )
        do_cfg = (
            float(context.guidance_scale) > 0.0 and neg_tokens_buf is not None
        )

        input_image_buf: Buffer | None = None
        if context.input_image is not None:
            input_image_buf = self._numpy_image_to_buffer(
                image=np.ascontiguousarray(
                    context.input_image.astype(np.uint8, copy=False)
                ),
                batch_size=int(context.num_images_per_prompt),
            )

        latents_buf = Buffer.from_dlpack(latents_np).to(device)

        for n in (packed_h, packed_w):
            if n not in self._cached_shape_carriers:
                self._cached_shape_carriers[n] = Buffer.from_dlpack(
                    np.ascontiguousarray(np.empty(n, dtype=np.float32))
                ).to(device)

        num_steps = int(context.num_inference_steps)
        sigmas_key = f"sigmas::{num_steps}::{latent_h}x{latent_w}"
        if sigmas_key in self._cached_sigmas:
            sigmas_buf = self._cached_sigmas[sigmas_key]
        else:
            sigmas_buf = Buffer.from_dlpack(
                np.ascontiguousarray(context.sigmas)
            ).to(device)
            self._cached_sigmas[sigmas_key] = sigmas_buf

        return ZImageModelInputs(
            **kwargs,
            do_cfg=do_cfg,
            tokens_tensor=tokens_buf,
            negative_tokens_tensor=neg_tokens_buf,
            txt_ids_tensor=txt_ids_buf,
            img_ids_tensor=img_ids_buf,
            negative_txt_ids_tensor=neg_txt_ids_buf,
            negative_img_ids_tensor=neg_img_ids_buf,
            input_image_tensor=input_image_buf,
            latents_tensor=latents_buf,
            sigmas_tensor=sigmas_buf,
            h_carrier=self._cached_shape_carriers[packed_h],
            w_carrier=self._cached_shape_carriers[packed_w],
        )

    @staticmethod
    def _select_tokens_for_text_encoder(
        tokens: np.ndarray,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        if tokens.ndim == 2:
            tokens = tokens[0]
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[0]
            selected = mask.astype(np.bool_, copy=False)
            if not np.any(selected):
                raise ValueError("ZImage mask cannot exclude all tokens.")
            if not np.all(selected):
                tokens = tokens[selected]
        return np.ascontiguousarray(tokens.astype(np.int64, copy=False))

    def _cache_token_buffer(self, tokens: np.ndarray, device: Device) -> Buffer:
        digest = hashlib.sha1(tokens.tobytes()).hexdigest()
        key = f"tokens::{tokens.shape[0]}::{digest}::{device}"
        if key in self._cached_prompt_token_tensors:
            return self._cached_prompt_token_tensors[key]
        buf = Buffer.from_dlpack(tokens).to(device)
        self._cached_prompt_token_tensors[key] = buf
        return buf

    def _prepare_conditioning_ids(
        self,
        text_seq_len: int,
        image_seq_len: int,
        latent_image_ids: np.ndarray,
        height: int,
        width: int,
        device: Device,
    ) -> tuple[Buffer, Buffer]:
        text_seq_len_padded = text_seq_len + (-text_seq_len % 32)

        img_base_key = f"img_ids_base::{image_seq_len}_{height}x{width}"
        if img_base_key in self._cached_img_ids_base_np:
            img_ids_base = self._cached_img_ids_base_np[img_base_key]
        else:
            img_ids_base = np.asarray(latent_image_ids, dtype=np.int64)
            if img_ids_base.ndim == 3:
                img_ids_base = img_ids_base[0]
            img_ids_base = np.ascontiguousarray(img_ids_base)
            self._cached_img_ids_base_np[img_base_key] = img_ids_base

        img_key = (
            f"img_ids::{text_seq_len_padded}_{image_seq_len}_{height}x{width}"
        )
        if img_key in self._cached_img_ids:
            img_buf = self._cached_img_ids[img_key]
        else:
            img_np = img_ids_base.copy()
            img_np[:, 0] = img_np[:, 0] + text_seq_len_padded + 1
            img_buf = Buffer.from_dlpack(np.ascontiguousarray(img_np)).to(
                device
            )
            self._cached_img_ids[img_key] = img_buf

        txt_key = f"text_ids::{text_seq_len}"
        if txt_key in self._cached_text_ids:
            txt_buf = self._cached_text_ids[txt_key]
        else:
            txt_ids = np.zeros((text_seq_len, 3), dtype=np.int64)
            txt_ids[:, 0] = np.arange(1, text_seq_len + 1, dtype=np.int64)
            txt_buf = Buffer.from_dlpack(np.ascontiguousarray(txt_ids)).to(
                device
            )
            self._cached_text_ids[txt_key] = txt_buf

        return txt_buf, img_buf

    def _numpy_image_to_buffer(
        self,
        image: npt.NDArray[np.uint8],
        batch_size: int,
    ) -> Buffer:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected input image shape [H, W, 3], got {image.shape}."
            )
        img_array = (image.astype(np.float32) / 127.5) - 1.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        if batch_size > 1:
            img_array = np.tile(img_array, (batch_size, 1, 1, 1))
        img_array = np.ascontiguousarray(img_array)
        return Buffer.from_dlpack(img_array).to(self.vae.devices[0])

    def _get_cached_guidance(
        self,
        guidance_scale: float,
        device: Device,
    ) -> Buffer:
        key = f"{guidance_scale:.8f}::{device}"
        if key in self._cached_guidance:
            return self._cached_guidance[key]
        buf = Buffer.from_dlpack(np.array(guidance_scale, dtype=np.float32)).to(
            device
        )
        self._cached_guidance[key] = buf
        return buf

    @traced(message="ZImagePipeline.prepare_prompt_embeddings")
    def prepare_prompt_embeddings(
        self,
        tokens: Buffer,
        num_images_per_prompt: int,
    ) -> Buffer:
        del num_images_per_prompt
        return cast(Buffer, self.text_encoder(tokens))

    @traced(message="ZImagePipeline.decode_latents")
    def decode_latents(
        self,
        latents: Buffer,
        h_carrier: Buffer,
        w_carrier: Buffer,
    ) -> npt.NDArray[np.uint8]:
        latents = self._unpack_and_postprocess(latents, h_carrier, w_carrier)
        decoded = cast(Buffer, self.vae.decode(cast(Any, latents)))
        image = self._postprocess_image(decoded)
        result: np.ndarray
        if hasattr(image, "to"):
            image = image.to(CPU())
        if hasattr(image, "__dlpack__"):
            result = np.from_dlpack(image)
        elif hasattr(image, "to_numpy"):
            result = image.to_numpy()
        else:
            result = image.to_numpy()
        if result.dtype != np.uint8:
            result = result.astype(np.uint8, copy=False)
        return cast(npt.NDArray[np.uint8], result)

    @traced(message="ZImagePipeline.preprocess_latents")
    def preprocess_latents(self, latents: Buffer) -> Buffer:
        return self._patchify_and_pack(latents)

    @traced(message="ZImagePipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: ZImageModelInputs,
    ) -> DiffusionPipelineOutput:
        with Tracer("prepare_prompt_embeddings"):
            prompt_embeds = self.prepare_prompt_embeddings(
                tokens=model_inputs.tokens_tensor,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )

            negative_prompt_embeds: Buffer | None = None
            if (
                model_inputs.do_cfg
                and model_inputs.negative_tokens_tensor is not None
            ):
                negative_prompt_embeds = self.prepare_prompt_embeddings(
                    tokens=model_inputs.negative_tokens_tensor,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                )

        latents = model_inputs.latents_tensor
        sigmas = model_inputs.sigmas_tensor
        h_carrier = model_inputs.h_carrier
        w_carrier = model_inputs.w_carrier

        timesteps: np.ndarray = model_inputs.timesteps
        num_timesteps = timesteps.shape[0]
        if num_timesteps < 1:
            raise ValueError("No timesteps were provided for denoising.")

        device = self.transformer.devices[0]
        img_ids = model_inputs.img_ids_tensor
        txt_ids = model_inputs.txt_ids_tensor
        latents = self.preprocess_latents(latents)

        with Tracer("prepare_scheduler"):
            timesteps_np = np.ascontiguousarray(
                model_inputs.timesteps.astype(np.float32, copy=False)
            )
            timesteps_buf = Buffer.from_dlpack(timesteps_np).to(device)
            all_timesteps, all_dt = self._prepare_scheduler(
                timesteps_buf, sigmas
            )

            timesteps_seq: Any = all_timesteps
            dts_seq: Any = all_dt
            if hasattr(timesteps_seq, "driver_tensor"):
                timesteps_seq = timesteps_seq.driver_tensor
            if hasattr(dts_seq, "driver_tensor"):
                dts_seq = dts_seq.driver_tensor

        cfg_cutoff_step = 0
        if model_inputs.do_cfg:
            transformed_host = (1.0 - timesteps_np).astype(np.float32)
            if model_inputs.cfg_truncation > 1.0:
                cfg_cutoff_step = num_timesteps
            else:
                mask = transformed_host <= model_inputs.cfg_truncation
                cfg_cutoff_step = int(np.count_nonzero(mask))

        guidance_buf: Buffer | None = None
        if model_inputs.do_cfg:
            guidance_buf = self._get_cached_guidance(
                model_inputs.guidance_scale, device
            )

        use_batched_cfg = bool(
            model_inputs.do_cfg and not model_inputs.explicit_negative_prompt
        )
        cfg_prompt_embeds: Buffer | None = None
        neg_img_ids = img_ids
        neg_txt_ids = txt_ids

        if model_inputs.do_cfg and negative_prompt_embeds is not None:
            if model_inputs.explicit_negative_prompt:
                assert model_inputs.negative_img_ids_tensor is not None
                assert model_inputs.negative_txt_ids_tensor is not None
                neg_img_ids = model_inputs.negative_img_ids_tensor
                neg_txt_ids = model_inputs.negative_txt_ids_tensor
            else:
                neg_aligned = self._align_prompt_embeds(
                    negative_prompt_embeds, prompt_embeds, device
                )
                cfg_prompt_embeds = self._concat_batch(
                    prompt_embeds, neg_aligned
                )

        cfg_timestep_bufs: list[Buffer] | None = None
        if use_batched_cfg:
            transformed = (1.0 - timesteps_np).astype(np.float32)
            batch_size = int(prompt_embeds.shape[0])
            cfg_timestep_bufs = [
                Buffer.from_dlpack(
                    np.full((2 * batch_size,), float(t), dtype=np.float32)
                ).to(device)
                for t in transformed
            ]

        with Tracer("denoising_loop"):
            for i in range(num_timesteps):
                apply_cfg = i < cfg_cutoff_step
                timestep = timesteps_seq[i : i + 1]
                dt = dts_seq[i : i + 1]

                with Tracer(f"denoising_step_{i}"):
                    if apply_cfg and use_batched_cfg:
                        assert cfg_prompt_embeds is not None
                        assert cfg_timestep_bufs is not None
                        with Tracer("transformer"):
                            latents_cfg = self._duplicate_batch(latents)
                            noise_pred_cfg = self.transformer(
                                latents_cfg,
                                cfg_prompt_embeds,
                                cfg_timestep_bufs[i],
                                img_ids,
                                txt_ids,
                            )[0]
                        assert guidance_buf is not None
                        if model_inputs.cfg_normalization:
                            _, noise_pred = self._cfg_finalize_with_norm(
                                noise_pred_cfg, guidance_buf
                            )
                        else:
                            _, noise_pred = self._cfg_finalize_no_norm(
                                noise_pred_cfg, guidance_buf
                            )
                    elif apply_cfg:
                        with Tracer("transformer"):
                            noise_pred = self.transformer(
                                latents,
                                prompt_embeds,
                                timestep,
                                img_ids,
                                txt_ids,
                            )[0]
                        assert negative_prompt_embeds is not None
                        with Tracer("cfg_transformer"):
                            neg_noise_pred = self.transformer(
                                latents,
                                negative_prompt_embeds,
                                timestep,
                                neg_img_ids,
                                neg_txt_ids,
                            )[0]
                        assert guidance_buf is not None
                        noise_pred = self._cfg_combine(
                            noise_pred, neg_noise_pred, guidance_buf
                        )
                        if model_inputs.cfg_normalization:
                            noise_pred = self._cfg_renormalization(
                                noise_pred,
                                noise_pred,
                            )
                    else:
                        with Tracer("transformer"):
                            noise_pred = self.transformer(
                                latents,
                                prompt_embeds,
                                timestep,
                                img_ids,
                                txt_ids,
                            )[0]

                    with Tracer("scheduler_step"):
                        latents = self._scheduler_step(latents, noise_pred, dt)

        with Tracer("decode_outputs"):
            images = self.decode_latents(latents, h_carrier, w_carrier)

        return DiffusionPipelineOutput(images=images)
