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
"""The ModuleV3 DSpark stages' config, which builds their module."""

from __future__ import annotations

from typing import Any

from max import functional as F
from max.dtype import DType
from max.nn.legacy.kv_cache import KVCacheParamInterface
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh

from ..deepseekV41_modulev3.dspark_drafter import DSparkDrafter
from ..deepseekV41_modulev3.layers.rope import RopeParams
from ..deepseekV41_modulev3.model_config import DeepseekV41Config
from ..deepseekV41_modulev3.weight_adapters import (
    DSPARK_MODULE_PREFIX,
    convert_dspark_state_dict,
)
from ..dspark_deepseekV41.model import DSparkDeepseekV41DraftArchConfigBase

__all__ = ["DSparkDeepseekV41ModuleV3DraftArchConfig"]


class DSparkDeepseekV41ModuleV3DraftArchConfig(
    DSparkDeepseekV41DraftArchConfigBase
):
    """The ModuleV3 stages' config, which builds their module."""

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DSparkDrafter, DeepseekV41Config]:
        """Builds the stages for the trunk's checkpoint named by ``context``.

        The module holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns. The stages embed and project
        through the trunk's embedding and head, which their proposer reads.
        Under data-parallel attention every one-device replica holds the
        whole stage stack and drafts its own rows.

        Raises:
            ValueError: If the method isn't ``dspark``, or the checkpoint
                names no token to fill the block with.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        assert speculative is not None
        if speculative.speculative_method != "dspark":
            raise ValueError(
                "DeepSeek-V4.1's DSpark stages require --speculative-method"
                f" dspark, got {speculative.speculative_method!r}."
            )
        config = DeepseekV41Config.initialize(
            pipeline_config, max_seq_len=context.max_seq_len
        )
        if config.dspark_noise_token_id is None:
            raise ValueError(
                "DeepSeek-V4.1 DSpark needs dspark_noise_token_id to fill the"
                " draft block."
            )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParamInterface)
            config.kv_params = context.kv_params
        dp = pipeline_config.model.data_parallel_degree
        devices = [ref.to_device() for ref in config.devices]
        # The trunk's mesh rule, so the stages place their rows on the axis
        # the trunk's split shards.
        axis = "dp" if dp > 1 else "tp"
        mesh = DeviceMesh(tuple(devices), (len(devices),), (axis,))
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            draft = DSparkDrafter(
                config,
                mesh,
                rope=RopeParams.from_text_config(
                    DeepseekV41Config.get_text_config(
                        pipeline_config.model.huggingface_config
                    )
                ),
                data_parallel_degree=dp,
            )
        return draft, config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: DeepseekV41Config
    ) -> dict[str, Any]:
        """Returns the checkpoint's ``mtp.*`` weights.

        The weights are named as the built stages name them.
        """
        prefix = f"{DSPARK_MODULE_PREFIX}."
        return {
            name.removeprefix(prefix): value
            for name, value in convert_dspark_state_dict(
                dict(context.weights.items()),
                num_stages=config.num_dspark_layers,
            ).items()
        }
