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
"""The ModuleV3 DFlash drafter's config, which builds its module."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext

from ..dflash_mimo_v2.drafter import (
    DFlashMiMoV2DraftArchConfigBase,
    DFlashMiMoV2DraftConfig,
    load_drafter,
)
from .dflash_mimo_v2 import DFlashMiMoV2

__all__ = ["DFlashMiMoV2ModuleV3DraftArchConfig"]


class DFlashMiMoV2ModuleV3DraftArchConfig(DFlashMiMoV2DraftArchConfigBase):
    """The ModuleV3 drafter's config, which builds its module."""

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DFlashMiMoV2, DFlashMiMoV2DraftConfig]:
        """Builds the drafter from the draft checkpoint ``context`` names.

        The module holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns. Its trained mask embedding is
        baked in as a constant. It embeds and projects through its target's
        embedding and head, which its proposer reads.

        Under data-parallel attention every one-device replica holds the
        whole drafter, as its target's replicas do.

        Raises:
            ValueError: If the method isn't ``dflash``.
        """
        config, _, mask_embedding, draft_config = cls.read_drafter(context)
        dp_degree = context.pipeline_config.model.data_parallel_degree
        config = replace(config, data_parallel_degree=dp_degree)
        draft_config = replace(draft_config, module=config)
        devices = [ref.to_device() for ref in config.devices]
        # The target's mesh rule, so the proposer places the drafter's
        # batch on the axis the target's split shards.
        axis_name = "dp" if dp_degree > 1 else "tp"
        mesh = DeviceMesh(tuple(devices), (len(devices),), (axis_name,))
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            draft = DFlashMiMoV2(config, mask_embedding=mask_embedding)
        return draft, draft_config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: DFlashMiMoV2DraftConfig
    ) -> dict[str, Any]:
        """Returns the weights the built drafter compiles with, checked
        against its config."""
        draft_model = context.pipeline_config.draft_model
        assert draft_model is not None
        state, _, _ = load_drafter(
            draft_model.huggingface_weight_repo, config.module
        )
        return dict(state)
