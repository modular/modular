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
"""The MiMo-V2 base graph that also writes the DFlash drafter's context.

With speculation on, a serving engine runs this graph for every step the
speculative graph does not: a prefix-cache hit's suffix, a chunked prefill, a
step with no drafts. It writes the drafter's context for the rows it commits,
so the drafter never reads a position that only the plain base graph saw.
Its outputs are the base graph's; the drafter's context is its tail KV
group, and its only other weights are the writer's, under ``draft.``.

MAX serving runs every speculative step through the fused graph, so nothing
selects this architecture by checkpoint; an export names it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from max.engine import InferenceSession
from max.graph import BufferValue, Graph, TensorValue
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams, PagedCacheValues

from ..dflash_mimo_v2 import DFlashContextWriter, DFlashMiMoV2Config
from ..mimo_v2.mimo_v2 import TapHook
from ..mimo_v2.model import MiMoV2Model
from ..mimo_v2.model_config import MiMoV2Config
from .drafter import DrafterExport, drafter_export, load_drafter
from .model_config import (
    DFLASH_DIR,
    DRAFT,
    MiMoV2DFlashContextConfig,
    drafter_config,
    read_dflash_config,
    repo_file,
)


def context_writer_hook(
    config: DFlashMiMoV2Config,
) -> tuple[DFlashContextWriter, TapHook]:
    """Builds the drafter's context writer and the tap hook that runs it.

    The hook writes the context of every row the target computes into the
    tail KV group, at the rows' positions.
    """
    writer = DFlashContextWriter(config)

    def hook(
        taps: list[list[TensorValue]],
        tail_kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> None:
        # The hook hands the taps layer-major; the writer reads each device's.
        per_device = [
            list(device_taps) for device_taps in zip(*taps, strict=True)
        ]
        writer(
            per_device, input_row_offsets, tail_kv_collections, signal_buffers
        )

    return writer, hook


def prefixed_context_writer(
    config: DFlashMiMoV2Config,
    draft_state: Mapping[str, WeightData] | None,
) -> tuple[TapHook, dict[str, Any]]:
    """Builds the context writer's tap hook, its weights named ``draft.*``.

    Args:
        config: The drafter's config.
        draft_state: The drafter's weights, of which the writer loads its
            own; ``None`` only names the writer's weights.

    Returns:
        The tap hook, and the writer's weights registry under ``draft.``,
        empty without ``draft_state``.
    """
    writer, hook = context_writer_hook(config)
    registry: dict[str, Any] = {}
    if draft_state is not None:
        writer.load_state_dict(
            {name: draft_state[name] for name in writer.raw_state_dict()},
            weight_alignment=1,
            strict=True,
        )
        registry = {
            f"{DRAFT}.{k}": v
            for k, v in writer.state_dict(auto_initialize=False).items()
        }
    # Under ``draft.``, no writer weight can share a target weight's name in
    # the merged registry.
    for name, weight in writer.raw_state_dict().items():
        weight.name = f"{DRAFT}.{name}"
    return hook, registry


class MiMoV2DFlashContextModel(MiMoV2Model):
    """The base-ctx build: the base graph plus the drafter's context writer.

    The drafter is the one the target checkpoint ships in ``dflash/``.
    """

    model_config_cls: ClassVar[type[Any]] = MiMoV2DFlashContextConfig

    drafter_export: DrafterExport
    _drafter_config: DFlashMiMoV2Config
    _writer_registry: dict[str, Any]

    def _create_model_config(self, state_dict: dict[str, Any]) -> MiMoV2Config:
        config = super()._create_model_config(state_dict)
        assert isinstance(self.kv_params, MultiKVCacheParams)
        draft_kv = self.kv_params.children[DRAFT]
        assert isinstance(draft_kv, KVCacheParams)
        repo = self.pipeline_config.model.huggingface_weight_repo
        self._drafter_config = drafter_config(
            read_dflash_config(repo_file(repo, f"{DFLASH_DIR}/config.json")),
            draft_kv,
            config.devices,
            config.max_seq_len,
        )
        config.target_layer_ids = list(self._drafter_config.target_layer_ids)
        return config

    def _tap_hook(self, model_config: MiMoV2Config) -> TapHook | None:
        repo = self.pipeline_config.model.huggingface_weight_repo
        state, _, directory = load_drafter(
            repo, self._drafter_config, subfolder=DFLASH_DIR
        )
        hook, self._writer_registry = prefixed_context_writer(
            self._drafter_config, state
        )
        self.drafter_export = drafter_export(self._drafter_config, directory)
        return hook

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: MiMoV2Config,
    ) -> tuple[Graph, dict[str, Any]]:
        graph, registry = super()._build_graph_for_compile(
            session, state_dict, model_config
        )
        assert not registry.keys() & self._writer_registry.keys()
        return graph, {**registry, **self._writer_registry}
