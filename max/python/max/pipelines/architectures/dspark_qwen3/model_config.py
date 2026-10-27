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
"""The ``Qwen3DSparkModel`` draft's config, which builds its module."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from max import functional as F
from max.driver import DeviceSpec, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import load_weights
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig

from ..nemotron_h_modulev3.model_config import is_sm100
from .dspark_qwen3 import (
    DSparkQwen3,
    DSparkQwen3DraftConfig,
    parse_dspark_draft_config,
)
from .weight_adapters import convert_draft_state_dict

logger = logging.getLogger("max.pipelines")

__all__ = [
    "Qwen3DSparkDraftArchConfig",
    "check_draft_runs_on",
    "dspark_nemotron_draft_width",
]

TARGET_EMBEDDING = "backbone.embeddings.weight"
"""The target checkpoint's embedding, which the draft's table must equal
but for the mask token's row."""


def check_draft_runs_on(
    device_specs: Sequence[DeviceSpec],
    draft: DSparkQwen3DraftConfig,
    *,
    devices_loader: Callable[[Sequence[DeviceSpec]], Sequence[Any]] = (
        load_devices
    ),
) -> None:
    """Checks that the devices run the draft's quantized matmuls.

    The draft's NVFP4 weights run on the block-scaled NVFP4 matmul, which is
    SM100-only. They are never dequantized, so another device has no way to
    run them.

    Args:
        device_specs: The devices the pipeline runs on.
        draft: The parsed draft config.
        devices_loader: Loads the devices from their specs.

    Raises:
        ValueError: If the draft is NVFP4 and a device is not SM100.
    """
    if not draft.quantized:
        return
    for device in devices_loader(device_specs):
        if not is_sm100(device):
            raise ValueError(
                "The NVFP4 DSpark draft runs its W4A16_NVFP4 linears on the"
                " block-scaled NVFP4 matmul, which needs SM100 GPUs, and"
                f" {device.architecture_name} is not one. Use a BF16 drafter"
                " for this model instead."
            )


def dspark_nemotron_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
    *,
    warn: bool = True,
) -> int:
    """Returns the draft width the checkpoint was trained for.

    The drafter only works at its trained block size. With
    ``sample_from_anchor: false`` the block's first slot holds the committed
    token and predicts nothing, so a block of 8 drafts 7 tokens.

    Args:
        speculative: The speculative config, whose width the checkpoint
            overrides.
        target_huggingface_config: Unused.
        draft_huggingface_config: The draft's Hugging Face config.
        warn: Whether to warn on an overridden width.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DSpark requires a draft model.")
    width = parse_dspark_draft_config(draft_huggingface_config).num_drafts
    requested = speculative.num_speculative_tokens
    if warn and requested is not None and requested != width:
        logger.warning(
            "The DSpark draft was trained at its block size, so"
            " num_speculative_tokens is being overridden from %d to %d.",
            requested,
            width,
        )
    return width


@dataclass
class Qwen3DSparkDraftArchConfig:
    """The config of the ``Qwen3DSparkModel`` draft, which builds its module.

    A DSpark checkpoint is only ever served as ``--draft-model`` next to a
    Nemotron-H target, in the unified spec-decode graph, so this builds the
    draft's module and KV leaf for that graph and clamps the draft side's
    maximum sequence length.
    """

    max_position_embeddings: int = 1 << 20

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Qwen3DSparkDraftArchConfig:
        del pipeline_config, max_seq_len
        assert model_config is not None
        hf = model_config.huggingface_config
        assert hf is not None
        return cls(max_position_embeddings=int(hf.max_position_embeddings))

    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del model_config
        return int(huggingface_config.max_position_embeddings)

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DSparkQwen3, DSparkQwen3DraftConfig]:
        """Builds the DSpark draft for the draft checkpoint ``context`` names.

        The module holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns.

        Raises:
            ValueError: If the method isn't ``dspark``, or the draft is NVFP4
                and a device is not SM100.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        if not speculative.is_dspark():
            raise ValueError(
                "The Qwen3DSparkModel draft is a DSpark draft and requires"
                " --speculative-method dspark, got"
                f" {speculative.speculative_method!r}."
            )
        config = parse_dspark_draft_config(draft_model.huggingface_config)
        check_draft_runs_on(pipeline_config.model.device_specs, config)
        kv_params = context.kv_params
        assert isinstance(kv_params, KVCacheParams)
        devices = [ref.to_device() for ref in kv_params.devices]
        mesh = DeviceMesh(tuple(devices), (len(devices),), ("tp",))
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            draft = DSparkQwen3(
                config,
                kv_params,
                num_devices=len(devices),
                max_seq_len=context.max_seq_len,
                device=devices[0],
            )
        return draft, config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: DSparkQwen3DraftConfig
    ) -> dict[str, Any]:
        """Returns the weights the built draft compiles with.

        The draft embeds through its target's table, all but the mask
        token's trained row, so its own table is checked against the target
        checkpoint's.

        Raises:
            ValueError: If the checkpoint doesn't match the module or the
                target's table.
        """
        pipeline_config = context.pipeline_config
        target = dict(
            load_weights(pipeline_config.model.resolved_weight_paths()).items()
        )
        return convert_draft_state_dict(
            {name: weight.data() for name, weight in context.weights.items()},
            config,
            target_embedding=target[TARGET_EMBEDDING].data(),
        )

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns the draft's KV leaf, from its own geometry.

        Pinned to BF16, since the draft fills the cache with its own
        unquantized projections. The cache is full-length: the draft's
        sliding window is the attention mask's, not the cache's.
        """
        del cache_dtype
        draft = parse_dspark_draft_config(huggingface_config)
        speculative = pipeline_config.speculative
        assert speculative is not None
        return kv_cache_config.to_params(
            dtype=DType.bfloat16,
            n_kv_heads=draft.num_key_value_heads,
            head_dim=draft.head_dim,
            num_layers=draft.num_hidden_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            speculative_method=speculative.speculative_method,
            num_draft_tokens=speculative.draft_width,
        )
