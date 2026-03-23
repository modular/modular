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

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.shape import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import traced

if TYPE_CHECKING:
    from max.pipelines.lib.interfaces.cache_mixin import DenoisingCacheConfig

from .flux2 import Flux2Transformer2DModel
from .model_config import Flux2Config
from .nvfp4_weight_adapter import convert_nvfp4_state_dict

# Mapping from stacked QKV key infixes to the split (Q, K, V) infixes.
_STACKED_QKV_INFIXES = {
    ".attn.qkv_proj.": (".attn.to_q.", ".attn.to_k.", ".attn.to_v."),
    ".attn.add_qkv_proj.": (
        ".attn.add_q_proj.",
        ".attn.add_k_proj.",
        ".attn.add_v_proj.",
    ),
}


class Flux2TransformerModel(ComponentModel):
    model: Callable[..., Any]

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        *,
        cache_config: DenoisingCacheConfig | None = None,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
            cache_config=cache_config,
        )
        self.config = Flux2Config.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    @traced(message="Flux2TransformerModel.load_model")
    def load_model(self) -> None:
        state_dict = {key: value.data() for key, value in self.weights.items()}

        # Convert BFL single-file NVFP4 naming to MAX parameter naming.
        if getattr(self.config, "quant_config", None) is not None:
            state_dict = convert_nvfp4_state_dict(state_dict)

        # Detect stacked (fused) QKV weights and split into separate Q/K/V
        # so the model always sees the split format.
        stacked_qkv = any(
            ".attn.qkv_proj." in k or ".attn.add_qkv_proj." in k
            for k in state_dict
        )
        if stacked_qkv:
            state_dict = self._split_stacked_qkv(state_dict)

        # Klein/distilled checkpoints can omit guidance embedder weights.
        has_guidance_embedder = any(
            "time_guidance_embed.guidance_embedder." in k or "guidance_in." in k
            for k in state_dict
        )
        if not has_guidance_embedder and getattr(
            self.config, "guidance_embeds", True
        ):
            if hasattr(self.config, "model_copy"):
                self.config = self.config.model_copy(
                    update={"guidance_embeds": False}
                )
            else:
                self.config.guidance_embeds = False
        with F.lazy():
            flux = Flux2Transformer2DModel(
                self.config, cache_config=self.cache_config
            )
            flux.to(self.devices[0])

        self.model = flux.compile(
            *flux.input_types(),
            weights=state_dict,
        )

    @staticmethod
    def _split_stacked_qkv(
        state_dict: dict[str, WeightData],
    ) -> dict[str, WeightData]:
        """Split fused QKV weights into separate Q, K, V entries."""
        out: dict[str, WeightData] = {}
        for key, value in state_dict.items():
            matched = False
            for stacked, (q, k, v) in _STACKED_QKV_INFIXES.items():
                if stacked not in key:
                    continue
                matched = True
                if key.endswith((".weight", ".weight_scale")):
                    buf = value.to_buffer()
                    chunk = buf.shape[0] // 3
                    for infix, i in zip([q, k, v], range(3), strict=False):
                        split_name = key.replace(stacked, infix)
                        split_buf = buf[i * chunk : (i + 1) * chunk, :]
                        out[split_name] = WeightData(
                            split_buf,
                            split_name,
                            value.dtype,
                            Shape(split_buf.shape),
                        )
                elif key.endswith((".weight_scale_2", ".input_scale")):
                    # Per-tensor scales are shared across Q/K/V.
                    for infix in (q, k, v):
                        out[key.replace(stacked, infix)] = value
                break
            if not matched:
                out[key] = value
        return out

    @traced(message="Flux2TransformerModel.__call__")
    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor,
        prev_residual: Tensor | None = None,
        prev_output: Tensor | None = None,
        teacache_prev_modulated_input: Tensor | None = None,
        teacache_cached_residual: Tensor | None = None,
        teacache_accumulated_rel_l1: Tensor | None = None,
        force_compute: Tensor | None = None,
    ) -> Any:
        args: tuple[Any, ...] = (
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )
        if teacache_prev_modulated_input is not None:
            args = (
                *args,
                teacache_prev_modulated_input,
                teacache_cached_residual,
                teacache_accumulated_rel_l1,
                force_compute,
            )
        elif prev_residual is not None:
            args = (*args, prev_residual, prev_output)
        return self.model(*args)
