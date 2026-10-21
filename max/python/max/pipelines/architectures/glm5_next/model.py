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
"""The GLM-5.3-Flash (``glm5_next``) pipeline model.

Derives from the DeepSeek-V3.2 pipeline model, which already owns the sparse
MLA, DSA indexer, MoE and expert-parallel plumbing GLM-5.3-Flash inherits
through GLM-5.2. What this class adds is everything the hybrid schedule
implies: a second per-request pool for the KDA conv and recurrent state, which
the framework's KV accounting does not see, and a readiness check that names
the bring-up lanes still outstanding.

Kept deliberately thin. Where GLM-5.3-Flash needs behaviour DeepSeek-V3.2 does
not have, the fix belongs in the layer or the config, not here.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.kv_cache.cache_params import KVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.deepseekV3_2.model import DeepseekV3_2Model
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec

from .glm5_next import Glm5Next
from .model_config import SPARSE_ATTENTION, Glm5NextConfig

logger = logging.getLogger("max.pipelines")

__all__ = ["Glm5NextModel"]


def _mla_cache_dtype(config: Glm5NextConfig) -> DType:
    """Storage dtype of the latent cache.

    ``kv_params`` is a :class:`MultiKVCacheParams` tree holding the MLA latent
    and the indexer key cache separately; only the leaves carry a dtype.
    """
    params = config.kv_params
    leaf = (
        params.children["mla"]
        if isinstance(params, MultiKVCacheParams)
        else params
    )
    assert isinstance(leaf, KVCacheParams), (
        "The MLA latent cache must be a leaf KVCacheParams, got "
        f"{type(leaf).__name__}."
    )
    return leaf.dtype


class Glm5NextModel(DeepseekV3_2Model):
    """GLM-5.3-Flash pipeline model."""

    model_config_cls: ClassVar[type[Any]] = Glm5NextConfig

    def _first_mla_layer(self) -> int:
        """Reads the first sparse-MLA layer off the schedule.

        GLM-5.3-Flash interleaves KDA and sparse MLA on a period-4 schedule, so
        layer 0 is KDA and carries no ``kv_a_layernorm``. Derived rather than
        hardcoded to 3: the index follows ``layer_types``, and a sibling that
        retunes the period would otherwise probe the wrong layer and raise.
        """
        text_config = Glm5NextConfig._get_text_config(self.huggingface_config)
        for index, layer_type in enumerate(
            Glm5NextConfig.resolve_layer_types(text_config)
        ):
            if layer_type == SPARSE_ATTENTION:
                return index
        raise ValueError(
            "GLM-5.3-Flash needs at least one sparse-attention layer, but "
            "layer_types declares none."
        )

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> Glm5NextConfig:
        """Builds the config, then logs the two pools' per-sequence costs.

        Both pools are sized independently and either can cap concurrency
        first, so the crossover is worth having in the log of any run whose
        batch size later turns out to be the constraint.
        """
        config = super()._create_model_config(state_dict)
        assert isinstance(config, Glm5NextConfig)

        # The base class installs the shared blockscaled-FP8 map, which
        # describes every layer as quantized. Replace it: GLM-5.3-Flash leaves
        # the 34 KDA layers, the whole indexer and `kv_b_proj` in bfloat16, and
        # reading those as FP8 produces plausible tensors and garbage output.
        config.quant_config = config.resolve_quant_scheme(state_dict)

        self._glm_config = config
        state_bytes = config.per_request_state_bytes()
        logger.info(
            "GLM-5.3-Flash pools: %d KDA layers holding %.1f MiB per sequence "
            "at %s, context-independent; %d sparse-MLA layers holding %.1f KiB "
            "per token. The state pool binds below roughly 12K context and the "
            "KV cache above it.",
            len(config.kda_layers),
            state_bytes / 1024**2,
            config.state_dtype,
            len(config.sparse_attention_layers),
            len(config.sparse_attention_layers)
            * config.mla_head_dim
            * _mla_cache_dtype(config).size_in_bytes
            / 1024,
        )
        return config

    #: The resolved config, captured in :meth:`_create_model_config`. The base
    #: class keeps it as a local, and the state pools are allocated after the
    #: graph is built, so it has to be held somewhere.
    _glm_config: Glm5NextConfig | None = None

    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the GLM-5.3-Flash decoder graph.

        Its inputs are in the order :meth:`Glm5Next.input_types` declares
        them, which is the one place that order is written down.
        """
        assert isinstance(model_config, Glm5NextConfig)
        nn_model = Glm5Next(model_config, kv_params=self.kv_params)
        nn_model.attach_layers(nn_model.build_layers())
        nn_model.load_state_dict(
            self._weights_dict, weight_alignment=1, strict=True
        )
        return {
            GraphRole.DECODER: GraphSpec(
                nn_model,
                "glm5_next_graph",
                nn_model.input_types(self.kv_params),
                weights=nn_model.state_dict(),
            )
        }
