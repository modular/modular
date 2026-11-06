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
"""The ModuleV3 DSpark stages' registration, which builds their module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.registration_draft import DraftRegistration
from max.tensor import default_dtype, defaults
from transformers import AutoConfig

from ..deepseekV41_modulev3.dspark_drafter import DSparkDrafter
from ..deepseekV41_modulev3.layers.rope import RopeParams
from ..deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
    num_dspark_stages,
)
from ..deepseekV41_modulev3.weight_adapters import (
    DSPARK_MODULE_PREFIX,
    convert_dspark_state_dict,
)

__all__ = [
    "DSparkDeepseekV41Draft",
    "DSparkDeepseekV41ModuleV3DraftConfig",
    "configure_dspark_draft",
    "convert_dspark_modulev3_safetensor_state_dict",
    "dspark_draft_registration",
]


@dataclass(kw_only=True)
class DSparkDeepseekV41ModuleV3DraftConfig(DeepseekV41Config):
    """The trunk's config, with the RoPE the stages read off its checkpoint."""

    rope: RopeParams | None = None


class DSparkDeepseekV41Draft(DSparkDrafter):
    """The DSpark stages, constructed on the mesh the framework sets.

    Under data-parallel attention every one-device replica holds the whole
    stage stack and drafts its own rows. The stages embed and project
    through the trunk's embedding and head, which their proposer reads.
    """

    def __init__(self, config: DSparkDeepseekV41ModuleV3DraftConfig) -> None:
        assert config.rope is not None
        with default_dtype(DType.bfloat16):
            super().__init__(
                config,
                defaults()[1],
                rope=config.rope,
                data_parallel_degree=config.data_parallel_degree,
            )


def configure_dspark_draft(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> DSparkDeepseekV41ModuleV3DraftConfig:
    """Finalizes the stages' config from the trunk's checkpoint.

    The stages read and write the trunk's KV leaves, which ``context``
    carries.

    Raises:
        ValueError: If the checkpoint names no token to fill the block with.
    """
    del state_dict
    pipeline_config = context.pipeline_config
    trunk = DeepseekV41Config.initialize(
        pipeline_config, max_seq_len=context.max_seq_len
    )
    if trunk.dspark_noise_token_id is None:
        raise ValueError(
            "DeepSeek-V4.1 DSpark needs dspark_noise_token_id to fill the"
            " draft block."
        )
    if context.kv_params is not None:
        assert isinstance(context.kv_params, KVCacheParamInterface)
        trunk.kv_params = context.kv_params
    # The trunk's mesh rule, so the stages place their rows on the axis the
    # trunk's split shards.
    trunk.data_parallel_degree = pipeline_config.model.data_parallel_degree
    return DSparkDeepseekV41ModuleV3DraftConfig(
        **{f.name: getattr(trunk, f.name) for f in fields(trunk)},
        rope=RopeParams.from_text_config(
            DeepseekV41Config.get_text_config(
                pipeline_config.model.huggingface_config
            )
        ),
    )


def convert_dspark_modulev3_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Returns the checkpoint's ``mtp.*`` weights, named as the stages name them.

    The trunk's weights load through the trunk's own adapter.
    """
    prefix = f"{DSPARK_MODULE_PREFIX}."
    return {
        name.removeprefix(prefix): value
        for name, value in convert_dspark_state_dict(
            state_dict,
            num_stages=num_dspark_stages(
                DeepseekV41Config.get_text_config(huggingface_config)
            ),
        ).items()
    }


dspark_draft_registration = DraftRegistration[
    DSparkDeepseekV41Draft, DSparkDeepseekV41ModuleV3DraftConfig
](method="dspark", configure=configure_dspark_draft)
