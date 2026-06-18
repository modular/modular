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
"""Denoiser sub-``Module`` for the FLUX.2 ModuleV3 executor (stub)."""

from __future__ import annotations

from typing import Any

from max.driver import DeviceSpec, load_devices
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.pipelines.modeling.config_enums import SupportedEncoding


class Denoiser(Module[[Tensor], Tensor]):
    """Stub fused FLUX.2 denoiser.

    Placeholder for the ModuleV3 port of the legacy fused
    :class:`max.pipelines.architectures.flux2.components.denoiser.Denoiser`,
    which concatenates image latents onto noise latents, runs the FLUX.2
    transformer, and applies a single Euler integration step. The full
    port is a sizeable transformer + Euler step implementation; it is
    deferred so the loop scaffolding in :class:`FLUXModule` -- in-graph
    noise initialization, ``ops.while_loop`` carry, ``num_inference_steps``
    -driven trip count -- can be built and verified independently.

    Forward today is a no-op: ``latents`` flow through unchanged. The
    surrounding loop in :class:`FLUXModule` therefore iterates
    ``num_inference_steps`` times without modifying state, which is
    exactly the property we want from the stub for validating
    compilation + execution of the loop primitive.

    No ``HasWeightAdapter`` is declared: there are no parameters to bind
    yet. The real port will add weight construction, key translation,
    and the full forward signature
    ``(latents, image_latents, text_embeddings, timestep, dt, guidance,
    latent_image_ids, text_ids) -> latents``.
    """

    def __init__(
        self,
        huggingface_config: dict[str, Any],
        quantization_encoding: SupportedEncoding | None,
        device_specs: list[DeviceSpec],
    ) -> None:
        # Hold onto config for the eventual real port; nothing is
        # consumed from it yet.
        self._huggingface_config = huggingface_config
        self._quantization_encoding = quantization_encoding
        devices = load_devices(device_specs)
        self.to(devices[0])

    def forward(self, latents: Tensor) -> Tensor:
        """No-op denoiser step.

        Returns ``latents`` unchanged so the surrounding
        ``ops.while_loop`` exercises the loop machinery without
        requiring any weights or real per-step computation.
        """
        return latents
