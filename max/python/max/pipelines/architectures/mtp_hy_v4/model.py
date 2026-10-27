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
"""The MTP layer a Hy4 checkpoint ships: its config and module builder.

The layer runs only beside its target, inside a unified spec-decode graph,
and builds from the target's own checkpoint.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.common_layers.mesh_axis import TP
from max.nn.legacy.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.weights.quant import parse_quant_config
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig

from ..hy_v4.model_config import HYV4Config
from ..hy_v4.mtp import HYV4MultiTokenPredictor, draft_config

__all__ = ["MTPHYV4DraftArchConfig"]


class MTPHYV4DraftArchConfig(HYV4Config):
    """The MTP layer's config: its target's, with caches for its one layer."""

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Returns the target's MLA and indexer leaves, each holding the one
        layer."""
        target = HYV4Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
        draft = {}
        for name, params in target.children.items():
            assert isinstance(params, KVCacheParams)
            draft[name] = replace(params, num_layers=1)
        return MultiKVCacheParams.from_params(draft)

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[HYV4MultiTokenPredictor, HYV4Config]:
        """Builds the MTP layer from the target's checkpoint, which
        ``context`` names.

        It holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns. It embeds and projects through
        its target's embedding and head, which its proposer reads.

        Raises:
            ValueError: If the checkpoint ships no MTP layer.
        """
        assert isinstance(context.kv_params, MultiKVCacheParams)
        config = HYV4Config.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        if config.num_nextn_predict_layers < 1:
            raise ValueError(
                "Hy4 MTP requires a checkpoint with num_nextn_predict_layers"
                " > 0"
            )
        config.max_batch_context_length = (
            context.max_batch_total_tokens or config.max_batch_context_length
        )
        config.quant_config = parse_quant_config(
            context.pipeline_config.model.huggingface_config,
            cls.checkpoint_weights(context, config),
            config.dtype,
        )
        config = draft_config(config, context.kv_params)
        devices = [ref.to_device() for ref in config.devices]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            draft = HYV4MultiTokenPredictor(config)
            if len(devices) > 1:
                draft.shard()
        return draft, config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: HYV4Config
    ) -> dict[str, Any]:
        """Returns the weights the built layer compiles with."""
        del config
        return parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
