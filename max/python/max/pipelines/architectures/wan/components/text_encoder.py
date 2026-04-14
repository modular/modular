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

"""Text encoder component for WanExecutor."""

from __future__ import annotations

import logging

import numpy as np
from max.driver import CPU, Buffer, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced

from ...umt5.model import _prepare_state_dict
from ...umt5.model_config import UMT5Config
from ...umt5.umt5 import UMT5EncoderModel

logger = logging.getLogger("max.pipelines")


class TextEncoder(CompiledComponent):
    """UMT5 text encoder component.

    Compiles the UMT5 encoder model and provides prompt embedding
    generation with padding to a fixed sequence length.
    """

    _model: Model

    # Diffusers pads tokens to 512 but trims final embeddings to 226
    # for cross-attention.
    embed_seq_len: int = 226

    @traced(message="TextEncoder.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
    ) -> None:
        super().__init__(manifest, session)

        config = manifest["text_encoder"]
        config_dict = config.huggingface_config.to_dict()
        encoding = config.quantization_encoding or "bfloat16"
        devices = load_devices(config.device_specs)

        umt5_config = UMT5Config.generate(config_dict, encoding, devices)
        # Force bfloat16 — some repos declare float32 but should run bf16.
        dtype = DType.bfloat16
        umt5_config.dtype = dtype

        self._device = devices[0]
        self._dtype = dtype

        # Load weights.
        paths = config.resolved_weight_paths()
        from max.graph.weights import load_weights

        weights = load_weights(paths)
        state_dict = _prepare_state_dict(weights, target_dtype=dtype)

        # Build module.
        dev_ref = DeviceRef.from_device(self._device)
        module = UMT5EncoderModel(umt5_config, dtype=dtype, device=dev_ref)
        module.load_state_dict(state_dict, weight_alignment=1, strict=True)

        # Build graph with symbolic sequence length.
        input_types = [
            TensorType(DType.int64, ["batch", "seq_len"], device=self._device),
            TensorType(DType.int64, ["batch", "seq_len"], device=self._device),
        ]
        with Graph("umt5_encoder", input_types=input_types) as graph:
            input_ids = graph.inputs[0].tensor
            attention_mask = graph.inputs[1].tensor
            out = module(input_ids, attention_mask)
            graph.output(out)

        self._model = self._load_graph(
            graph, weights_registry=module.state_dict()
        )

    @traced(message="TextEncoder.__call__")
    def __call__(
        self,
        token_ids: Buffer,
        attention_mask: Buffer | None,
        num_videos_per_prompt: int,
        max_sequence_length: int | None = None,
    ) -> Buffer:
        """Encode text tokens into prompt embeddings.

        Runs the UMT5 encoder and post-processes: pads/truncates to
        ``max_sequence_length``, repeats for ``num_videos_per_prompt``.

        Args:
            token_ids: Token IDs, shape ``(batch, seq)`` int64 on device.
            attention_mask: Attention mask, shape ``(batch, seq)`` int64,
                or ``None`` (derived from non-zero token IDs).
            num_videos_per_prompt: Number of videos per prompt for batching.
            max_sequence_length: Target embedding sequence length.
                Defaults to ``embed_seq_len`` (226).

        Returns:
            Prompt embeddings, shape ``(B, max_seq_len, hidden_dim)``
            in model dtype.
        """
        if max_sequence_length is None:
            max_sequence_length = self.embed_seq_len

        # Ensure 2D inputs.
        token_ids_np = np.from_dlpack(token_ids.to(CPU()))
        if token_ids_np.ndim == 1:
            token_ids_np = np.expand_dims(token_ids_np, axis=0)

        if attention_mask is not None:
            mask_np = np.from_dlpack(attention_mask.to(CPU()))
            if mask_np.ndim == 1:
                mask_np = np.expand_dims(mask_np, axis=0)
        else:
            mask_np = (token_ids_np != 0).astype(np.int64)

        # Transfer to device.
        text_input_ids = Buffer.from_dlpack(
            np.ascontiguousarray(token_ids_np, dtype=np.int64)
        ).to(self._device)
        text_attention_mask = Buffer.from_dlpack(
            np.ascontiguousarray(mask_np.astype(np.int64, copy=False))
        ).to(self._device)

        # Run encoder.
        raw = self._model.execute(text_input_ids, text_attention_mask)
        hidden_states = raw[0] if isinstance(raw, (list, tuple)) else raw

        # Post-process: pad/truncate to max_sequence_length, repeat for batch.
        batch_size = int(hidden_states.shape[0])
        hidden_dim = int(hidden_states.shape[2])
        hidden_np = _buffer_to_numpy_f32(hidden_states).reshape(
            batch_size, int(hidden_states.shape[1]), hidden_dim
        )
        mask_cpu = np.from_dlpack(text_attention_mask.to(CPU())).reshape(
            batch_size, int(text_attention_mask.shape[1])
        )

        embeds_np = np.zeros(
            (batch_size, max_sequence_length, hidden_dim), dtype=np.float32
        )
        for b in range(batch_size):
            seq_len = min(
                int(mask_cpu[b].sum()),
                hidden_np.shape[1],
                max_sequence_length,
            )
            embeds_np[b, :seq_len, :] = hidden_np[b, :seq_len, :]

        if num_videos_per_prompt > 1:
            embeds_np = np.repeat(embeds_np, num_videos_per_prompt, axis=0)

        # Convert to device buffer in model dtype.
        out_device = self._device
        if self._dtype == DType.bfloat16:
            u16 = float32_to_bfloat16_as_uint16(np.ascontiguousarray(embeds_np))
            return (
                Buffer.from_numpy(u16)
                .to(out_device)
                .view(dtype=DType.bfloat16, shape=embeds_np.shape)
            )
        return Buffer.from_numpy(np.ascontiguousarray(embeds_np)).to(out_device)


def _buffer_to_numpy_f32(buf: Buffer) -> np.ndarray:
    """Convert a Buffer (possibly bf16) to f32 numpy on CPU."""
    cpu_buf = buf.to(CPU())
    if cpu_buf.dtype == DType.bfloat16:
        u16 = np.from_dlpack(
            cpu_buf.view(dtype=DType.uint16, shape=cpu_buf.shape)
        )
        return (u16.astype(np.uint32) << 16).view(np.float32)
    return np.from_dlpack(cpu_buf).astype(np.float32, copy=False)
