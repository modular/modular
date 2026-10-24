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
"""The fused MiMo-V2.6-Flash + DFlash pipeline model on ModuleV3: one graph
per step.

The target's weights load through the ModuleV3 base architecture's adapter
and batching conventions; the drafter's through the graph-API drafter's
checkpoint readers, which the all-BF16 drafter satisfies unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.compilation import CompiledCallable
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.comm.ep import (
    EPBatchManager,
    EPConfig,
    calculate_ep_max_tokens_per_rank,
)
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.batch_processor import BatchProcessor
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.profiler import traced
from typing_extensions import override

from ..mimo_v2_modulev3.model import (
    MiMoV2ModuleV3Model,
    _validate_data_parallel_attention,
)
from ..mimo_v2_modulev3.model_config import MiMoV2Config
from .batch_processor import UnifiedDflashMiMoV2ModuleV3BatchProcessor
from .drafter import DrafterExport, drafter_export, load_drafter
from .model_config import (
    DRAFT,
    UnifiedDflashMiMoV2Config,
    drafter_config,
    read_dflash_config,
    repo_file,
    sampleable_vocab_size,
    verify_widened_row_budget,
)
from .unified_dflash_mimo_v2 import (
    DP_SIGNAL_BUFFER_INDEX,
    GRAPH_NAME,
    SIGNAL_BUFFER_INDEX,
    UnifiedDflashMiMoV2ModuleV3,
    UnifiedDflashMiMoV2Spec,
)

__all__ = ["UnifiedDflashMiMoV2ModuleV3Model"]


# The two bases annotate ``model`` for their own execute paths -- the
# mixin's engine ``Model`` (what this class stores and runs; its ``__call__``
# returns ``list[Buffer]``) and the ModuleV3 base's compiled callable
# (``ModuleV3Outputs``). The contracts genuinely conflict; deriving from the
# V2 fused model like GLM's spec port is not available to a V3-native target,
# so the conflict is scoped here rather than papered over in the lib.
class UnifiedDflashMiMoV2ModuleV3Model(  # type: ignore[misc]
    _UnifiedSpecDecodeModelMixin,
    AlwaysSignalBuffersMixin,
    MiMoV2ModuleV3Model,
):
    """Target and drafter in one compiled ModuleV3 graph per speculative step.

    The graph stages under the graph-API build's name and input layout, so
    Mach's block-spec executor binds either export through the same slots.
    Beside data-parallel attention -- one-device replicas, expert
    parallelism over every device on the P2P backend -- the graph declares
    the DP batch-split pair between the batch inputs and the KV tree, the
    per-device batch context lengths and the EP input tail past it, and the
    drafter holds full weights on every replica.
    """

    model_config_cls: ClassVar[type[Any]] = UnifiedDflashMiMoV2Config
    # The spec-decode batch processor is this architecture's own, not the
    # ModuleV3 base's single-replica one, which the base names.
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]]] = (
        UnifiedDflashMiMoV2ModuleV3BatchProcessor
    )

    drafter_export: DrafterExport
    _spec: UnifiedDflashMiMoV2Spec
    _draft_state: dict[str, WeightData]
    _mask_embedding: WeightData
    _compiled: CompiledCallable[..., tuple[Any, ...]]
    # Set by the inherited _init_distributed_runtime ahead of _build_module.
    _ep_batch_manager: EPBatchManager | None

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        del return_logits
        assert pipeline_config.speculative is not None
        dp_degree = pipeline_config.model.data_parallel_degree
        # The base model's own degree check, before anything loads: the
        # fused build used to reach it in _init_distributed_runtime, after
        # the whole target checkpoint and the drafter had loaded.
        _validate_data_parallel_attention(
            dp_degree, pipeline_config.runtime.ep_size, len(devices)
        )
        if pipeline_config.runtime.ep_size > 1 and dp_degree <= 1:
            # The canonical spec-decode signature declares the EP input tail
            # only beside data parallelism (see build_spec_decode_input_types
            # and spec_input_types), so a TP spec graph has no slot to bind
            # the EP buffers. The DP form -- one-device replicas on the P2P
            # expert backend -- is the supported EP export.
            raise NotImplementedError(
                "Expert parallelism for the fused DFlash spec graph serves "
                "data-parallel attention only; export the TP form without "
                "--ep-size, or set --data-parallel-degree to the device "
                "count beside --ep-size."
            )
        self.resolved_num_speculative_tokens = (
            pipeline_config.speculative.draft_width
        )
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            memory_plan=memory_plan,
            adapter=adapter,
            # The graph returns the verify logits and the tapped hidden
            # states whatever a plain target asked for.
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            max_batch_size=max_batch_size,
        )

    @traced
    @override
    def _load_decoder(self, session: InferenceSession) -> Any:
        del session
        target_state = self._load_state_dict()
        # Hold the checkpoint only as long as this method does.
        del self._checkpoint
        target_config = self._create_model_config(target_state)
        # The EP/DP runtime state the build reads: the target config's
        # ep_config, the EP batch manager the fused module threads, and --
        # off virtual devices -- the initialized comm buffers. The base
        # model's graph_specs does this ahead of its own module build.
        self._init_distributed_runtime(target_config)

        module = self._build_module(target_config)
        weights = {
            **{f"target.{k}": v for k, v in target_state.items()},
            **{f"{DRAFT}.{k}": v for k, v in self._draft_state.items()},
        }
        # ``Module.compile`` iterates the declared parameters and silently
        # drops a weights entry none of them matches, where the graph-API
        # build's ``load_state_dict(strict=True)`` failed loudly; keep that
        # loudness at load time rather than trusting the drop to be benign.
        if extra := set(weights) - set(dict(module.parameters)):
            raise ValueError(
                f"the adapted checkpoint holds {len(extra)} weights the "
                "fused module declares no parameter for, first: "
                f"{sorted(extra)[:4]}"
            )
        n_devices = len(self.devices)
        dp_degree = self.pipeline_config.model.data_parallel_degree
        self._compiled = module.compile(
            *module.input_types(),
            weights=weights,
            name=GRAPH_NAME,
            # The engine binds the signal buffers between the batch inputs
            # and the KV tree at every device count; a single-device trace
            # declares its own there instead. Under data parallelism the
            # batch-split pair precedes them, two slots further in.
            signal_buffer_index=(
                None
                if n_devices == 1
                else (
                    DP_SIGNAL_BUFFER_INDEX
                    if dp_degree > 1
                    else SIGNAL_BUFFER_INDEX
                )
            ),
        )
        self.state_dict = dict(self._compiled.weights)
        # The batch processor packs the EP comm buffers the runtime
        # initialized; the generic load path binds them, but this model
        # compiles its own module, so the binding is ours too.
        self._bind_ep_comm_initializer()
        # Execution drives the engine model directly; the batch processor
        # supplies this model's own signal buffers (see its Inputs.buffers).
        return self._compiled.engine_model

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> MiMoV2Config:
        del state_dict
        assert isinstance(self.kv_params, MultiKVCacheParams)
        target_kv = self.kv_params.children["target"]
        draft_kv = self.kv_params.children[DRAFT]
        assert isinstance(target_kv, MultiKVCacheParams)
        assert isinstance(draft_kv, KVCacheParams)
        # Verify rows and the block reach past the last committed position.
        target = MiMoV2Config.initialize(
            self.pipeline_config,
            max_seq_len=self.max_seq_len
            + spec_decode_cache_slack(self.kv_params),
        )
        target.kv_params = target_kv
        target.return_logits = ReturnLogits.VARIABLE
        target.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        # The base model's own config build threads the degree the same way;
        # the fused build reads it for its mesh, its signature and the
        # drafter's placement.
        target.data_parallel_degree = (
            self.pipeline_config.model.data_parallel_degree
        )
        draft_model = self.pipeline_config.draft_model
        assert draft_model is not None
        repo = draft_model.huggingface_weight_repo
        draft = drafter_config(
            read_dflash_config(repo_file(repo, "config.json")),
            draft_kv,
            target.devices,
            self.max_seq_len,
        )
        # The drafter's degree matches the target's, as Eagle3's draft
        # config does: one full drafter per one-device replica. This is its
        # single source -- the fused module builds the drafter straight
        # from the spec, so the drafter's own degree guard stays live.
        draft = replace(draft, data_parallel_degree=target.data_parallel_degree)
        target.target_layer_ids = list(draft.target_layer_ids)
        self._draft_state, self._mask_embedding, directory = load_drafter(
            repo, draft
        )
        speculative = self.pipeline_config.speculative
        assert speculative is not None
        self._spec = UnifiedDflashMiMoV2Spec(
            target=target,
            draft=draft,
            speculative_config=speculative,
            num_speculative_tokens=speculative.draft_width,
            sampleable_vocab_size=sampleable_vocab_size(self.pipeline_config),
        )
        self.drafter_export = drafter_export(
            draft,
            directory,
            num_speculative_tokens=speculative.draft_width,
        )
        return target

    @override
    def _build_ep_config(
        self, ep_size: int, n_devices: int, model_config: MiMoV2Config
    ) -> EPConfig:
        """The base config's buffers, widened for the verify rows.

        A full decode step merges ``draft_width`` draft rows beside each
        request's active token -- rows the scheduler's token budget never
        charges -- so the buffers the base sizes to
        ``max_batch_input_tokens`` alone would overflow, and the dispatch
        kernel rejects the wider shard. The widened budget also refuses a
        build without ``max_batch_size``; the base target keeps its own
        sizing, having no draft rows to carry.
        """
        config = super()._build_ep_config(ep_size, n_devices, model_config)
        return replace(
            config,
            max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=verify_widened_row_budget(
                    self.pipeline_config
                ),
                ep_size=ep_size,
                data_parallel_degree=(
                    self.pipeline_config.model.data_parallel_degree
                ),
                use_allreduce=config.use_allreduce,
            ),
        )

    def _build_module(
        self, target_config: MiMoV2Config
    ) -> UnifiedDflashMiMoV2ModuleV3:
        n_devices = len(target_config.devices)
        # One axis, named for what it carries: the batch axis under data
        # parallelism (the DP forward's split placement keys on "dp"), the
        # tensor axis otherwise -- the base model's own mesh rule.
        dp_degree = self.pipeline_config.model.data_parallel_degree
        axis_name = "dp" if dp_degree > 1 else "tp"
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), (axis_name,))
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            return UnifiedDflashMiMoV2ModuleV3(
                self._spec,
                self._mask_embedding,
                kv_params=self.kv_params,
                enable_structured_output=(
                    self.pipeline_config.needs_bitmask_constraints
                ),
                ep_batch_manager=self._ep_batch_manager,
            )
