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
"""The Mistral Large 4 DSpark drafter's config, which builds its module.

The drafter ships inside the target's checkpoint, so its geometry comes from
the checkpoint's ``params.json`` (the ``mtp`` block) and its weights are the
checkpoint's ``mtp_dspark.*`` tensors. Nothing here reads a draft checkpoint of
its own: ``--self-draft --speculative-method dspark`` names this architecture
from the target's ``checkpoint_drafts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface, KVCacheParams
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.speculative import DrafterExport
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig
from typing_extensions import Self

from ..mistral_large4_modulev3.checkpoint import (
    checkpoint_dirs,
    params_json_candidates,
)
from ..mistral_large4_modulev3.dspark import (
    DSPARK_MODULE_PREFIX,
    DSparkDrafter,
    DSparkDrafterConfig,
    build_dspark_kv_params,
    convert_dspark_state_dict,
)
from ..mistral_large4_modulev3.model_config import (
    MistralLarge4NormalizedConfig,
    read_params_json,
)
from ..mistral_large4_modulev3.spec_adapters import MASK_TOKEN_ID

__all__ = [
    "DSparkMistralLarge4DraftArchConfig",
    "DSparkMistralLarge4DraftConfig",
    "checkpoint_dir",
    "normalized_checkpoint_config",
]


def checkpoint_dir(pipeline_config: PipelineConfig) -> Path:
    """The target checkpoint directory the drafter's tensors live in.

    A local checkpoint is its own directory; a serve or export from a repo id
    is the Hub cache snapshot its weights resolve into. Neither is derived from
    the weight filenames, which are repo-relative for a Hub serve.

    Args:
        pipeline_config: The pipeline's config.

    Returns:
        The first candidate directory, which is where a spec MEF's provenance
        record's file paths are relative to.

    Raises:
        ValueError: If the checkpoint resolves no directory on disk.
    """
    dirs = checkpoint_dirs(pipeline_config.model)
    if not dirs:
        raise ValueError(
            "the Mistral Large 4 DSpark drafter needs a local checkpoint"
            f" directory; {pipeline_config.model.model_path!r} resolves none"
        )
    return dirs[0]


def normalized_checkpoint_config(
    pipeline_config: PipelineConfig,
) -> MistralLarge4NormalizedConfig:
    """Reads the checkpoint's ``params.json``, which the drafter's ``mtp`` block is in.

    The file is fetched from the checkpoint's repo when a Hub export has not
    downloaded it: a weights download never brings it, and a drafter's geometry
    that read no ``params.json`` would export no taps.

    Args:
        pipeline_config: The pipeline's config.

    Returns:
        The normalized checkpoint geometry.

    Raises:
        ValueError: If no candidate holds a ``params.json`` with an ``mtp``
            block, or the block declares no taps.
    """
    for path in params_json_candidates(pipeline_config.model):
        if not path.is_file():
            continue
        normalized = MistralLarge4NormalizedConfig.from_params_json(
            read_params_json(path)
        )
        if normalized.mtp is None:
            raise ValueError(
                f"{path} declares no mtp block, so this checkpoint carries no"
                " DSpark drafter"
            )
        return normalized
    raise ValueError(
        "the Mistral Large 4 DSpark drafter reads its geometry from the"
        " checkpoint's params.json, and none sits beside"
        f" {pipeline_config.model.model_path!r} or its weights; an export from"
        " a repo id alone would capture no taps"
    )


@dataclass(frozen=True)
class DSparkMistralLarge4DraftConfig:
    """The drafter's geometry and what the pairing reads beside it.

    The pairing checks a draft against its target through this: the hidden
    size, the vocabulary, the tap layers whose columns the projection was
    trained on, the block width and the mask token's in-vocabulary
    placeholder.
    """

    module: DSparkDrafterConfig
    """The drafter module's own geometry."""
    drafter_export: DrafterExport
    """What the graph bakes in of the drafter, for an export's sidecar."""

    @property
    def hidden_size(self) -> int:
        """Returns the drafter's hidden width, the target's."""
        return self.module.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the target vocabulary the drafter scores against."""
        return self.module.vocab_size

    @property
    def num_target_layers(self) -> int | None:
        """Returns ``None``: the checkpoint declares no target depth.

        The taps index the target's layers directly, and the tap list is
        checked against the served target's depth by the pairing, so this
        drafter has no separately trained depth to compare.
        """
        return None

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the target layers whose residuals the drafter reads."""
        return list(self.module.target_layers)

    @property
    def block_size(self) -> int:
        """Returns the block width, the drafter's ``depth``."""
        return self.module.depth

    @property
    def mask_token_id(self) -> int:
        """Returns the block's fill token, which nothing embeds."""
        return MASK_TOKEN_ID


@dataclass
class DSparkMistralLarge4DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound and KV leaf.

    The drafter is served only beside its target, so nothing else of it is read
    from here; :meth:`build_module` builds the module.
    """

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Records the sequence length the pipeline resolved."""
        del pipeline_config, model_config
        return cls(max_seq_len=max_seq_len)

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Returns the drafter's own one-plane latent cache.

        The leaf declares the two draft dispatch slots an MLA group carries
        under speculation, which the block adapter's record for the drafter
        claims per device; a leaf built without them is two slots short and the
        engine refuses the export's arity.

        Args:
            huggingface_config: The target's config, unread: the drafter's
                geometry is the checkpoint's ``params.json``.
            pipeline_config: The pipeline's config.
            devices: The devices the cache spans.
            kv_cache_config: Page-size and budget configuration.
            cache_dtype: The target's cache dtype, unread: the drafter's latent
                is always bf16.
        """
        del huggingface_config, cache_dtype
        speculative = pipeline_config.speculative
        assert speculative is not None
        draft = DSparkDrafterConfig.from_normalized(
            normalized_checkpoint_config(pipeline_config)
        )
        leaf = build_dspark_kv_params(
            kv_cache_config,
            devices=devices,
            config=draft,
            speculative=speculative,
            data_parallel_degree=(pipeline_config.model.data_parallel_degree),
        )
        if leaf.speculative_method is None:
            raise ValueError(
                "the DSpark drafter's leaf was built without its speculative"
                " method, so it is missing the draft dispatch pair the block"
                " adapter's record claims per device"
            )
        return leaf

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DSparkDrafter, DSparkMistralLarge4DraftConfig]:
        """Builds the drafter for the target checkpoint ``context`` names.

        The module holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns. It embeds and projects through the
        target's embedding and head, which its proposer reads.

        Raises:
            ValueError: If the method isn't ``dspark``, the requested draft
                width isn't the block the drafter drafts, or the target
                captured other taps than the ones the projection was trained
                on.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        assert speculative is not None
        if speculative.speculative_method != "dspark":
            raise ValueError(
                "Mistral Large 4's DSpark drafter requires"
                f" --speculative-method dspark, got"
                f" {speculative.speculative_method!r}."
            )
        normalized = normalized_checkpoint_config(pipeline_config)
        draft_config = DSparkDrafterConfig.from_normalized(normalized)
        # The drafter samples its anchor slot, so its block is its draft count:
        # a narrower K would leave the block's own forward and the verify's
        # width out of step.
        if speculative.draft_width != draft_config.depth:
            raise ValueError(
                "Mistral Large 4's DSpark block drafts"
                f" {draft_config.depth} tokens, so --num-speculative-tokens"
                f" must be {draft_config.depth}; got"
                f" {speculative.draft_width}."
            )
        if context.target_layer_ids and (
            tuple(context.target_layer_ids) != draft_config.target_layers
        ):
            raise ValueError(
                "Mistral Large 4's DSpark projection was trained on the taps"
                f" {list(draft_config.target_layers)}; the target captures"
                f" {list(context.target_layer_ids)}."
            )
        kv_params = context.kv_params
        assert isinstance(kv_params, KVCacheParams)
        devices = [ref.to_device() for ref in kv_params.devices]
        axis = "dp" if pipeline_config.model.data_parallel_degree > 1 else "tp"
        mesh = DeviceMesh(tuple(devices), (len(devices),), (axis,))
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            drafter = DSparkDrafter(draft_config)
        return drafter, DSparkMistralLarge4DraftConfig(
            module=draft_config,
            drafter_export=DrafterExport(
                directory=checkpoint_dir(pipeline_config),
                target_layer_ids=list(draft_config.target_layers),
                num_speculative_tokens=speculative.draft_width,
                spec_block_size=draft_config.depth,
            ),
        )

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: DSparkMistralLarge4DraftConfig
    ) -> dict[str, Any]:
        """Returns the checkpoint's ``mtp_dspark.*`` weights, by module name.

        The checkpoint's own tensors are materialized here: the fused module
        compiles with these values, and a ``Weights`` handle is a name rather
        than something the compile can bind. Only the drafter's own namespace
        is materialized -- a self-drafting pairing would otherwise touch every
        tensor of the target checkpoint.
        """
        return dict(
            convert_dspark_state_dict(
                {
                    name: value.data()
                    for name, value in context.weights.items()
                    if name.startswith(DSPARK_MODULE_PREFIX)
                },
                num_layers=config.module.num_layers,
            )
        )
