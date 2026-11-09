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
"""The MuseSparkDSparkModel draft's config, which builds its module."""

from __future__ import annotations

from typing import Any

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig

from ..speculators_common.draft_config import DSparkSpeculatorsDraftArchConfig
from .dspark_muse_spark import MuseSparkDraftConfig, MuseSparkDSpark
from .weight_adapters import convert_draft_state_dict


def check_pairing(target_text: Any, draft: MuseSparkDraftConfig) -> None:
    """Checks the drafter against the target it runs beside.

    Args:
        target_text: The target's Hugging Face ``text_config``.
        draft: The parsed drafter config.

    Raises:
        ValueError: If the pair cannot run together.
    """
    spec = draft.spec
    layers = target_text.num_hidden_layers
    if draft.target_num_hidden_layers != layers:
        raise ValueError(
            "The DSpark draft was trained against a"
            f" {draft.target_num_hidden_layers}-layer target, but the target"
            f" has {layers} layers."
        )
    if spec.target_layer_ids[-1] >= layers:
        raise ValueError(
            f"The DSpark draft taps layers {list(spec.target_layer_ids)}, but"
            f" the target has {layers}."
        )
    for field in ("hidden_size", "vocab_size"):
        if getattr(spec, field) != getattr(target_text, field):
            raise ValueError(
                f"The DSpark draft's {field} {getattr(spec, field)} differs"
                f" from the target's {getattr(target_text, field)}."
            )


class MuseSparkDSparkDraftArchConfig(DSparkSpeculatorsDraftArchConfig):
    """The Muse Spark DSpark draft's config, which builds its module.

    The draft is only served as --draft-model beside a Muse Spark target.
    """

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[MuseSparkDSpark, MuseSparkDraftConfig]:
        """Builds the drafter for the draft checkpoint context names.

        The module holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns.

        Raises:
            ValueError: If the method isn't dspark, or the drafter can't
                run beside the target.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        if not speculative.is_dspark():
            raise ValueError(
                "The MuseSparkDSparkModel draft is a DSpark draft and requires"
                " --speculative-method dspark, got"
                f" {speculative.speculative_method!r}."
            )
        config = MuseSparkDraftConfig.from_huggingface_config(
            draft_model.huggingface_config
        )
        check_pairing(context.huggingface_config.text_config, config)
        kv_params = context.kv_params
        assert isinstance(kv_params, KVCacheParams)
        devices = [ref.to_device() for ref in kv_params.devices]
        mesh = DeviceMesh(tuple(devices), (len(devices),), ("tp",))
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            draft = MuseSparkDSpark(
                config,
                kv_params,
                max_seq_len=context.max_seq_len,
                device=devices[0],
            )
        return draft, config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: MuseSparkDraftConfig
    ) -> dict[str, Any]:
        """Returns the weights the built drafter compiles with."""
        del config
        return convert_draft_state_dict(dict(context.weights.items()))

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns the drafter's own KV leaf, BF16 and full-length.

        The drafter's sliding window is its attention mask's, not its cache's.
        """
        del cache_dtype
        spec = MuseSparkDraftConfig.from_huggingface_config(
            huggingface_config
        ).spec
        speculative = pipeline_config.speculative
        assert speculative is not None
        return kv_cache_config.to_params(
            dtype=DType.bfloat16,
            n_kv_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            num_layers=spec.num_hidden_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            speculative_method=speculative.speculative_method,
            num_draft_tokens=speculative.draft_width,
        )
