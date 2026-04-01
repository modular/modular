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

"""Wan VAE autoencoder -- slim ComponentModel with init-time graph compilation.

All decoder/encoder graphs are compiled once at ``load_model()`` time with
symbolic spatial dims, so a single set of compiled models handles any
resolution without recompilation.

Module classes live in ``vae.py``; this file only contains the
ComponentModel wrapper and numpy/buffer conversion helpers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import Tracer

from .model_config import AutoencoderKLWanConfig
from .vae import (
    WAN_DECODER_CACHE_SLOTS,
    WAN_ENCODER_CHUNK_SIZE,
    Decoder3dCached,
    Encoder3dCached,
    VAEDecoderFirstFrameCached,
    VAEDecoderRestFrameCached,
    VAEEncoderFirstChunk,
    VAEEncoderRestChunk,
    VAEPostQuantConv,
    _use_nvidia_fcrs_conv3d,
)

logger = logging.getLogger(__name__)


def _buffer_to_numpy_f32(buf: Buffer, cpu: CPU | None = None) -> np.ndarray:
    """Convert a Buffer (possibly bf16) to f32 numpy on CPU."""
    cpu_buf = buf.to(cpu or CPU())
    if cpu_buf.dtype == DType.bfloat16:
        u16 = np.from_dlpack(
            cpu_buf.view(dtype=DType.uint16, shape=cpu_buf.shape)
        )
        return (u16.astype(np.uint32) << 16).view(np.float32)
    return np.from_dlpack(cpu_buf).astype(np.float32, copy=False)


def _numpy_f32_to_buffer(
    arr: np.ndarray, target_dtype: DType, device: Device
) -> Buffer:
    """Convert f32 numpy to Buffer on device with target dtype."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if target_dtype == DType.bfloat16:
        u16 = float32_to_bfloat16_as_uint16(arr)
        return (
            Buffer.from_numpy(u16)
            .to(device)
            .view(dtype=DType.bfloat16, shape=arr.shape)
        )
    return Buffer.from_numpy(arr).to(device)


class AutoencoderKLWanModel(ComponentModel):
    """Wan VAE model using MAX-native 3D modules (decoder + optional encoder).

    All graphs are compiled once at ``load_model()`` time with symbolic spatial
    dims, so a single set of compiled models handles any resolution.
    """

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession | None = None,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = AutoencoderKLWanConfig.generate(config, encoding, devices)
        self.config.dtype = DType.bfloat16

        self.pqc_model: Model | None = None
        self.first_frame_model: Model | None = None
        self.rest_frame_model: Model | None = None
        self.first_chunk_encoder: Model | None = None
        self.rest_chunk_encoder: Model | None = None

        self._session = session or InferenceSession(devices=devices)
        self._load_lock = threading.Lock()

        self.load_model()

    def load_model(self) -> Callable[[Buffer], Buffer]:
        """Load weights, remap layouts, and compile all graphs."""
        with self._load_lock:
            if self.pqc_model is not None:
                return self.decode_4d

            decoder_state_dict: dict[str, Any] = {}
            encoder_state_dict: dict[str, Any] = {}
            has_encoder = False
            target_dtype = self.config.dtype

            assert self.weights is not None
            weights_obj: Any = self.weights

            for key, value in weights_obj.items():
                is_decoder = key.startswith("decoder.") or key.startswith(
                    "post_quant_conv."
                )
                is_encoder = key.startswith("encoder.") or key.startswith(
                    "quant_conv."
                )
                if not (is_decoder or is_encoder):
                    continue

                weight_data = value.data()

                # -- 5D conv weights: PyTorch FCQRS -> MAX native QRSCF --
                if key.endswith(".weight") and len(weight_data.shape) == 5:
                    use_native_layout = not _use_nvidia_fcrs_conv3d(
                        self.config.device
                    )
                    if "time_conv" in key and (
                        key.startswith("encoder.")
                        or key.startswith("quant_conv.")
                    ):
                        use_native_layout = True
                    if use_native_layout:
                        buf = (
                            weight_data.to_buffer()
                            if hasattr(weight_data, "to_buffer")
                            else weight_data
                        )
                        t_f32 = cast_dlpack_to(
                            buf, weight_data.dtype, DType.float32, CPU()
                        )
                        weight_data = np.ascontiguousarray(
                            np.from_dlpack(t_f32).transpose(2, 3, 4, 1, 0)
                        )

                # -- 4D conv weights --
                if key.endswith(".weight") and len(weight_data.shape) == 4:
                    is_resample_conv = "resample" in key
                    if not is_resample_conv:
                        buf = (
                            weight_data.to_buffer()
                            if hasattr(weight_data, "to_buffer")
                            else weight_data
                        )
                        t_f32 = cast_dlpack_to(
                            buf, weight_data.dtype, DType.float32, CPU()
                        )
                        weight_data = np.ascontiguousarray(
                            np.from_dlpack(t_f32).transpose(2, 3, 1, 0)
                        )

                if is_decoder:
                    decoder_state_dict[key] = weight_data
                if is_encoder:
                    encoder_state_dict[key] = weight_data
                    has_encoder = True

            # Cast all weights to target dtype.
            cpu_device = CPU()
            for sd in (decoder_state_dict, encoder_state_dict):
                for key in sd:
                    tensor = sd[key]
                    if hasattr(tensor, "to_buffer") and hasattr(
                        tensor, "dtype"
                    ):
                        src_dtype = tensor.dtype
                        if src_dtype == target_dtype:
                            continue
                        buf = tensor.to_buffer()
                    else:
                        src_dtype = DType.float32
                        if src_dtype == target_dtype:
                            continue
                        buf = tensor
                    sd[key] = cast_dlpack_to(
                        buf, src_dtype, target_dtype, cpu_device
                    )

            # Compile decoder graphs with symbolic dims.
            self._compile_decoder_graphs(decoder_state_dict)

            # Compile encoder graphs (optional).
            if has_encoder:
                self._compile_encoder_graphs(encoder_state_dict)

            self.weights = None  # type: ignore[assignment]
            return self.decode_4d

    def _compile_decoder_graphs(
        self, decoder_state_dict: dict[str, Any]
    ) -> None:
        """Compile PQC + first-frame + rest-frame decoder with symbolic dims."""
        cfg = self.config
        dtype = cfg.dtype
        dev = self.devices[0]
        dev_ref = DeviceRef.from_device(dev)

        pqc_module = VAEPostQuantConv(cfg)
        pqc_module.load_state_dict(
            decoder_state_dict, weight_alignment=1, strict=False
        )
        pqc_input_types = [
            TensorType(dtype, [1, cfg.z_dim, 1, "height", "width"], device=dev)
        ]
        with Graph("wan_vae_pqc", input_types=pqc_input_types) as pqc_graph:
            out = pqc_module(pqc_graph.inputs[0].tensor)
            pqc_graph.output(out)
        self.pqc_model = self._session.load(
            pqc_graph, weights_registry=pqc_module.state_dict()
        )

        first_module = VAEDecoderFirstFrameCached(cfg)
        first_module.load_state_dict(
            decoder_state_dict, weight_alignment=1, strict=False
        )
        first_input_types = [
            TensorType(dtype, [1, cfg.z_dim, 1, "height", "width"], device=dev)
        ]
        with Graph(
            "wan_vae_first_frame", input_types=first_input_types
        ) as first_graph:
            outputs = first_module(first_graph.inputs[0].tensor)
            first_graph.output(*outputs)
        self.first_frame_model = self._session.load(
            first_graph, weights_registry=first_module.state_dict()
        )

        rest_module = VAEDecoderRestFrameCached(cfg)
        rest_module.load_state_dict(
            decoder_state_dict, weight_alignment=1, strict=False
        )

        # Build cache input types with level-specific symbolic dim names.
        # Caches at the same decoder level share dim names so concat in
        # forward_cached sees matching dims on non-concat axes.
        decoder_for_shapes = Decoder3dCached(
            dim=cfg.base_dim,
            z_dim=cfg.z_dim,
            dim_mult=tuple(cfg.dim_mult),
            num_res_blocks=cfg.num_res_blocks,
            temporal_upsample=tuple(reversed(cfg.temporal_downsample)),
            out_channels=cfg.out_channels,
            is_residual=cfg.is_residual,
            dtype=dtype,
            device=dev_ref,
        )
        cache_shape_info = decoder_for_shapes.cache_shapes(
            batch_size=1, latent_height=1, latent_width=1
        )

        # Map each cache to its decoder level for dim naming.
        cache_dim_names: list[tuple[str, str]] = []
        h_name, w_name = "height", "width"
        level = 0

        # conv_in cache
        cache_dim_names.append((h_name, w_name))
        # mid_block: 2 resnets x 2 caches = 4
        for _ in range(4):
            cache_dim_names.append((h_name, w_name))
        # up_blocks
        for up_block in decoder_for_shapes.up_blocks:
            for _ in up_block.resnets:
                cache_dim_names.append((h_name, w_name))
                cache_dim_names.append((h_name, w_name))
            if up_block.upsamplers is not None:
                if up_block._has_temporal_upsample:
                    cache_dim_names.append((h_name, w_name))
                level += 1
                h_name = f"h{level}"
                w_name = f"w{level}"
        # conv_out cache
        cache_dim_names.append((h_name, w_name))

        assert len(cache_dim_names) == WAN_DECODER_CACHE_SLOTS

        rest_input_types = [
            TensorType(dtype, [1, cfg.z_dim, 1, "height", "width"], device=dev)
        ]
        for i, shape in enumerate(cache_shape_info):
            channels = shape[1]
            cache_t = shape[2]
            ch, cw = cache_dim_names[i]
            rest_input_types.append(
                TensorType(dtype, [1, channels, cache_t, ch, cw], device=dev)
            )

        with Graph(
            "wan_vae_rest_frame", input_types=rest_input_types
        ) as rest_graph:
            z_input = rest_graph.inputs[0].tensor
            cache_inputs = tuple(inp.tensor for inp in rest_graph.inputs[1:])
            outputs = rest_module(z_input, *cache_inputs)
            rest_graph.output(*outputs)
        self.rest_frame_model = self._session.load(
            rest_graph, weights_registry=rest_module.state_dict()
        )

    def _compile_encoder_graphs(
        self, encoder_state_dict: dict[str, Any]
    ) -> None:
        """Compile first-chunk + rest-chunk encoder with symbolic dims."""
        cfg = self.config
        dtype = cfg.dtype
        dev = self.devices[0]
        dev_ref = DeviceRef.from_device(dev)

        first_module = VAEEncoderFirstChunk(cfg)
        first_module.load_state_dict(
            encoder_state_dict, weight_alignment=1, strict=False
        )
        first_input_type = TensorType(
            dtype, [1, 3, 1, "height", "width"], device=dev
        )
        with Graph(
            "wan_vae_enc_first", input_types=[first_input_type]
        ) as first_graph:
            outputs = first_module(first_graph.inputs[0].tensor)
            first_graph.output(*outputs)
        self.first_chunk_encoder = self._session.load(
            first_graph, weights_registry=first_module.state_dict()
        )

        rest_module = VAEEncoderRestChunk(cfg)
        rest_module.load_state_dict(
            encoder_state_dict, weight_alignment=1, strict=False
        )

        encoder_for_shapes = Encoder3dCached(
            dim=cfg.base_dim,
            z_dim=cfg.z_dim,
            in_channels=3,
            dim_mult=cfg.dim_mult,
            num_res_blocks=cfg.num_res_blocks,
            temporal_downsample=cfg.temporal_downsample,
            dtype=dtype,
            device=dev_ref,
        )
        # Encoder has no upsample so cache shapes use same dims throughout.
        cache_shape_info = encoder_for_shapes.cache_shapes(
            batch_size=1, height=None, width=None
        )

        rest_input_types = [
            TensorType(
                dtype,
                [1, 3, WAN_ENCODER_CHUNK_SIZE, "height", "width"],
                device=dev,
            )
        ]
        for i, shape in enumerate(cache_shape_info):
            channels = shape[1]
            cache_t = shape[2]
            assert channels is not None and cache_t is not None
            rest_input_types.append(
                TensorType(
                    dtype,
                    [1, channels, cache_t, f"eh{i}", f"ew{i}"],
                    device=dev,
                )
            )

        with Graph(
            "wan_vae_enc_rest", input_types=rest_input_types
        ) as rest_graph:
            rest_inputs = [inp.tensor for inp in rest_graph.inputs]
            outputs = rest_module(rest_inputs[0], *rest_inputs[1:])
            rest_graph.output(*outputs)
        self.rest_chunk_encoder = self._session.load(
            rest_graph, weights_registry=rest_module.state_dict()
        )

    def decode_5d(self, latents_5d: Buffer) -> Buffer:
        """Decode 5D latents [B, C, T, H, W] frame-by-frame."""
        if self.pqc_model is None:
            self.load_model()
        pqc_model = self.pqc_model
        first_frame_model = self.first_frame_model
        rest_frame_model = self.rest_frame_model
        assert pqc_model is not None
        assert first_frame_model is not None
        assert rest_frame_model is not None

        t_total = int(latents_5d.shape[2])
        if t_total <= 0:
            raise ValueError("Expected non-empty temporal dimension for decode")

        cpu = CPU()
        latents_np = _buffer_to_numpy_f32(latents_5d, cpu)
        device = self.devices[0]
        target_dtype = self.config.dtype

        decoded_frames: list[np.ndarray] = []
        caches: list[Buffer] | None = None

        with Tracer("wan_vae_decode"):
            for t_idx in range(t_total):
                z_t_np = np.ascontiguousarray(
                    latents_np[:, :, t_idx : t_idx + 1, :, :]
                )
                z_t_buf = _numpy_f32_to_buffer(z_t_np, target_dtype, device)

                # Post-quant conv
                pqc_outputs = pqc_model.execute(z_t_buf)
                if len(pqc_outputs) != 1:
                    raise ValueError(
                        f"Expected 1 output from post_quant_conv, "
                        f"got {len(pqc_outputs)}"
                    )
                z_t_buf = pqc_outputs[0]

                if t_idx == 0:
                    outputs = first_frame_model.execute(z_t_buf)
                else:
                    if caches is None:
                        raise ValueError(
                            "Cached framewise decoder expected caches "
                            "after first frame."
                        )
                    outputs = rest_frame_model.execute(z_t_buf, *caches)

                if len(outputs) != 1 + WAN_DECODER_CACHE_SLOTS:
                    raise ValueError(
                        "Cached framewise decoder produced "
                        f"{len(outputs)} tensors; "
                        f"expected {1 + WAN_DECODER_CACHE_SLOTS}."
                    )

                decoded_buf = outputs[0]
                caches = list(outputs[1:])
                decoded_frames.append(_buffer_to_numpy_f32(decoded_buf, cpu))

        stitched = np.ascontiguousarray(np.concatenate(decoded_frames, axis=2))
        return Buffer.from_numpy(stitched)

    def decode_4d(self, latents_4d: Buffer) -> Buffer:
        """Decode 4D latents by adding and removing a temporal dim."""
        shape_5d = (
            int(latents_4d.shape[0]),
            int(latents_4d.shape[1]),
            1,
            int(latents_4d.shape[2]),
            int(latents_4d.shape[3]),
        )
        z5d = latents_4d.view(dtype=latents_4d.dtype, shape=shape_5d)
        decoded_5d = self.decode_5d(z5d)
        # Remove temporal dimension from decoded output.
        cpu = CPU()
        decoded_np = _buffer_to_numpy_f32(decoded_5d, cpu)
        return Buffer.from_numpy(
            np.ascontiguousarray(decoded_np[:, :, 0, :, :])
        )

    def decode(
        self, latents: Buffer, return_dict: bool = False
    ) -> tuple[Buffer]:
        del return_dict
        if latents.rank == 5:
            return (self.decode_5d(latents),)
        return (self.decode_4d(latents),)

    def encode(self, video: Buffer) -> Buffer:
        """Encode a video tensor [B, 3, T, H, W] to latent space.

        Uses chunked encoding matching diffusers: first frame processed
        separately, then 4-frame chunks with temporal caching.

        Returns the mean of the diagonal Gaussian (argmax mode),
        shape [B, z_dim, T_latent, H_latent, W_latent].
        """
        if self.first_chunk_encoder is None:
            self.load_model()
        first_chunk_encoder = self.first_chunk_encoder
        rest_chunk_encoder = self.rest_chunk_encoder
        if first_chunk_encoder is None or rest_chunk_encoder is None:
            raise RuntimeError(
                "VAE encoder weights not available. "
                "Ensure the model checkpoint includes encoder weights."
            )

        video_np = _buffer_to_numpy_f32(video, CPU())
        target_dtype = self.config.dtype
        device = self.devices[0]
        cpu = CPU()

        t_total = video_np.shape[2]
        latent_chunks: list[np.ndarray] = []
        caches: list[Buffer] | None = None
        num_chunks = 1 + (t_total - 1) // WAN_ENCODER_CHUNK_SIZE

        with Tracer("wan_vae_encode"):
            for i in range(num_chunks):
                if i == 0:
                    chunk_np = np.ascontiguousarray(video_np[:, :, :1])
                else:
                    start = 1 + WAN_ENCODER_CHUNK_SIZE * (i - 1)
                    end = 1 + WAN_ENCODER_CHUNK_SIZE * i
                    chunk_np = np.ascontiguousarray(video_np[:, :, start:end])

                chunk_buf = _numpy_f32_to_buffer(chunk_np, target_dtype, device)

                if i == 0:
                    outputs = first_chunk_encoder.execute(chunk_buf)
                else:
                    assert caches is not None
                    outputs = rest_chunk_encoder.execute(chunk_buf, *caches)

                latent_chunks.append(_buffer_to_numpy_f32(outputs[0], cpu))
                caches = list(outputs[1:])

        full_latent = np.ascontiguousarray(
            np.concatenate(latent_chunks, axis=2)
        )
        return _numpy_f32_to_buffer(full_latent, target_dtype, device)

    def encode_zero_padded_video_condition(
        self,
        first_frame: np.ndarray,
        *,
        batch_size: int,
        num_frames: int,
    ) -> Buffer:
        """Encode a zero-padded I2V conditioning video without materializing it.

        The conditioning path only contains a real first frame; all later
        frames are zeros. Stream those chunks directly into the cached encoder
        so we avoid allocating the full ``[B, 3, T, H, W]`` input tensor.
        """
        if num_frames <= 0:
            raise ValueError("num_frames must be positive for I2V encoding.")
        if first_frame.ndim != 4:
            raise ValueError(
                "Expected first_frame with shape [B, 3, H, W], "
                f"got {first_frame.shape}."
            )

        image_f32 = np.ascontiguousarray(first_frame, dtype=np.float32)
        if image_f32.shape[0] == 1 and batch_size > 1:
            image_f32 = np.repeat(image_f32, batch_size, axis=0)
        elif image_f32.shape[0] != batch_size:
            raise ValueError(
                "first_frame batch dimension must be 1 or match batch_size, "
                f"got {image_f32.shape[0]} and {batch_size}."
            )

        chunks = [image_f32[:, :, np.newaxis, :, :]]
        if num_frames > 1:
            _, channels, height, width = image_f32.shape
            zero_chunk = np.zeros(
                (batch_size, channels, WAN_ENCODER_CHUNK_SIZE, height, width),
                dtype=np.float32,
            )
            remaining_frames = num_frames - 1
            while remaining_frames > 0:
                chunk_len = min(WAN_ENCODER_CHUNK_SIZE, remaining_frames)
                chunks.append(zero_chunk[:, :, :chunk_len])
                remaining_frames -= chunk_len

        return self._encode_chunk_sequence(chunks)

    def _encode_chunk_sequence(self, chunks: list[np.ndarray]) -> Buffer:
        """Encode a pre-split Wan VAE chunk sequence."""
        if self.first_chunk_encoder is None:
            self.load_model()
        first_chunk_encoder = self.first_chunk_encoder
        rest_chunk_encoder = self.rest_chunk_encoder
        if first_chunk_encoder is None or rest_chunk_encoder is None:
            raise RuntimeError(
                "VAE encoder weights not available. "
                "Ensure the model checkpoint includes encoder weights."
            )

        target_dtype = self.config.dtype
        device = self.devices[0]
        cpu = CPU()

        latent_chunks: list[np.ndarray] = []
        caches: list[Buffer] | None = None
        with Tracer("wan_vae_encode"):
            for i, chunk_np in enumerate(chunks):
                chunk_buf = _numpy_f32_to_buffer(chunk_np, target_dtype, device)
                if i == 0:
                    outputs = first_chunk_encoder.execute(chunk_buf)
                else:
                    assert caches is not None
                    outputs = rest_chunk_encoder.execute(chunk_buf, *caches)

                latent_chunks.append(_buffer_to_numpy_f32(outputs[0], cpu))
                caches = list(outputs[1:])

        full_latent = np.ascontiguousarray(
            np.concatenate(latent_chunks, axis=2)
        )
        return _numpy_f32_to_buffer(full_latent, target_dtype, device)

    def __call__(self, latents: Buffer) -> Buffer:
        if latents.rank == 5:
            return self.decode_5d(latents)
        return self.decode_4d(latents)
