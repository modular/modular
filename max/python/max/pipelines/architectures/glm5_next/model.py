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

from max import tree
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import WeightData
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    RecurrentStateInputsPerDevice,
)
from max.nn.kv_cache.cache_params import KVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.deepseekV3_2.model import DeepseekV3_2Model
from max.pipelines.architectures.deepseekV3_2.model_config import (
    DeepseekV3_2Config,
)

from .glm5_next import Glm5Next
from .model_config import SPARSE_ATTENTION, Glm5NextConfig
from .state_cache import (
    INDEXER_CACHE_KEY,
    MLA_CACHE_KEY,
    STATE_CACHE_KEY,
)

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

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, WeightData],
        model_config: DeepseekV3_2Config,
    ) -> tuple[Graph, dict[str, Any]]:
        """Builds the GLM-5.3-Flash graph.

        Unpacks the input groups in the order
        :meth:`Glm5Next.input_types` declares them, which is the one place that
        order is written down.
        """
        del session
        assert isinstance(model_config, Glm5NextConfig)

        nn_model = Glm5Next(model_config)
        nn_model.attach_layers(nn_model.build_layers())
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        weights_registry = nn_model.state_dict()

        num_devices = len(self.devices)

        with Graph(
            "glm5_next_graph",
            input_types=nn_model.input_types(self.kv_params),
        ) as graph:
            (
                tokens,
                device_input_row_offsets,
                _host_input_row_offsets,
                return_n_logits,
                _data_parallel_splits,
                *variadic,
            ) = graph.inputs
            it = iter(variadic)

            signal_buffers = [next(it).buffer for _ in range(num_devices)]
            # The recurrent state is a child of the multi-cache, so one
            # unflatten resolves the attention leaves and the state together.
            kv_tree = self.kv_params.unflatten_kv_inputs(it)
            # A multi-cache unflattens to its children keyed by cache name.
            # `Tree` is a recursive union that cannot say so, so narrow once
            # here and read each child through `tree.leaves`, which types the
            # per-device list the sublayers take.
            assert isinstance(kv_tree, dict)
            if STATE_CACHE_KEY not in kv_tree:
                raise ValueError(
                    "the GLM-5.3-Flash graph needs KDA layers; the cache"
                    " declared no recurrent state child"
                )
            mla_kv = tree.leaves(
                kv_tree[MLA_CACHE_KEY], leaf=KVCacheInputsPerDevice
            )
            indexer_kv = tree.leaves(
                kv_tree[INDEXER_CACHE_KEY], leaf=KVCacheInputsPerDevice
            )
            state = tree.leaves(
                kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
            )
            # Consumed to keep the input prefix identical to
            # `DeepseekV3Inputs.buffers`; this graph does not read them yet.
            for _ in range(num_devices):
                next(it)
            ep_inputs = (
                [next(it) for _ in nn_model.ep_manager.input_types()]
                if nn_model.ep_manager is not None
                else None
            )

            outputs = nn_model(
                tokens.tensor,
                signal_buffers,
                mla_kv,
                indexer_kv,
                return_n_logits.tensor,
                device_input_row_offsets.tensor,
                state,
                ep_inputs,
            )
            graph.output(*outputs)

        return graph, weights_registry

    def load_model(self, session: InferenceSession) -> Any:
        """Compiles the model.

        The graph is GLM-5.3-Flash's own; see
        :meth:`_build_graph_for_compile`.
        """
        return super().load_model(session)
