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
"""Hy4 with MTP: target and draft in one ModuleV3 graph."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.experimental.compilation import CompiledCallable
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.realization_context import GraphRealizationContext
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import (
    Tensor,
    current_realization_context,
    default_device,
)
from max.graph import BufferType, DeviceRef, TensorType, TensorValue
from max.nn.kv_cache import MultiKVCacheParams
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    ModelInputs,
    ModuleV3PipelineModelWithKVCache,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib.interfaces.pipeline_model import UnifiedEagleOutputs
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from max.pipelines.weights.quant import parse_quant_config
from typing_extensions import override

from ..hy_v4.hy_v4 import HYV4TextModel
from ..hy_v4.model import check_parallelism
from ..hy_v4.model_config import HYV4Config
from .batch_processor import UnifiedMTPHYV4BatchProcessor
from .hy_v4_mtp import HYV4MultiTokenPredictor, draft_config
from .model_config import UnifiedMTPHYV4Config
from .spec_adapters import (
    INDEXER_DRAFT_KV,
    INDEXER_TARGET_KV,
    HYV4MTPProposer,
    HYV4Target,
    ModuleV3Handle,
)

__all__ = ["UnifiedMTPHYV4", "UnifiedMTPHYV4Inputs", "UnifiedMTPHYV4Model"]


@dataclass
class UnifiedMTPHYV4Inputs(UnifiedSpecDecodeInputs):
    """The canonical single-device spec-decode inputs, then the host prefill
    buffer length."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    batch_context_length: Buffer
    # Not graph inputs: the compiled graph appends its own. The overlap
    # pipeline's multi-device realize graph reads these.
    signal_buffers: list[Buffer] = field(default_factory=list)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *tree.leaves(self.kv_cache_inputs),
        ) + (
            *self._spec_decode_tail_buffers(include_in_thinking_phase=False),
            self.batch_context_length,
        )


class UnifiedMTPHYV4(Module[..., tuple[Tensor, ...]]):
    """Merge, Hy4 target, acceptance and the MTP draft, as one graph.

    The phases are the shared :class:`SequentialDriver`'s, built per trace
    over this module's own target and draft so the trace reads the weights
    it declares.

    Args:
        config: The target config; ``kv_params`` is the nested tree from
            :class:`UnifiedMTPHYV4Config`.
        speculative_config: The pipeline's speculative config.
        enable_structured_output: Whether the graph takes bitmask inputs.
    """

    def __init__(
        self,
        config: HYV4Config,
        speculative_config: SpeculativeConfig,
        enable_structured_output: bool,
    ) -> None:
        super().__init__()
        assert isinstance(config.kv_params, MultiKVCacheParams)
        self.kv_params = config.kv_params
        target_kv = config.kv_params.child("target", MultiKVCacheParams)
        draft_kv = config.kv_params.child("draft", MultiKVCacheParams)
        self.devices = config.devices
        self.speculative_config = speculative_config
        self.enable_structured_output = enable_structured_output
        self.language_model = HYV4TextModel(
            replace(config, kv_params=target_kv)
        )
        self.draft = HYV4MultiTokenPredictor(draft_config(config, draft_kv))

    def shard(self) -> None:
        """Shards the target as :meth:`HYV4TextModel.shard` and the draft
        block as one of its layers; ``enorm``, ``hnorm`` and ``eh_proj`` stay
        replicated."""
        self.language_model.shard()
        self.draft.shard()

    def driver(self) -> SequentialDriver[list[TensorValue]]:
        return SequentialDriver(
            HYV4Target(self.language_model).verify,
            HYV4MTPProposer(self.draft, self.language_model),
            target_model=ModuleV3Handle(self.language_model),
            draft_model=ModuleV3Handle(self.draft),
            input_spec=SpecDecodeInputTypeSpec(
                devices=self.devices, distributed=False
            ),
            speculative_config=self.speculative_config,
            enable_structured_output=self.enable_structured_output,
            use_greedy_acceptance=self.speculative_config.use_greedy_acceptance,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        return (
            *self.driver().input_types(self.kv_params),
            TensorType(DType.int32, [1], device=DeviceRef.CPU()),
        )

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        *spec_inputs, batch_context_length = inputs
        driver = self.driver()
        # The driver's own broadcasts share the buffers the compile declares
        # for the modules' collectives; there are none on one device.
        ctx = current_realization_context()
        assert isinstance(ctx, GraphRealizationContext)
        g = driver.decode_inputs(
            (t._graph_value for t in spec_inputs), self.kv_params
        )
        outputs = driver(
            g.tokens,
            g.input_row_offsets,
            g.draft_tokens,
            kv_collections=g.kv("target", "mla"),
            draft_kv_collections=g.kv("draft", "mla"),
            passthrough_kv={
                INDEXER_TARGET_KV: g.kv("target", "indexer"),
                INDEXER_DRAFT_KV: g.kv("draft", "indexer"),
            },
            return_n_logits=g.return_n_logits,
            signal_buffers=ctx.signal_buffers,
            batch_context_lengths=[TensorValue(batch_context_length)],
            seed=g.seed,
            temperature=g.temperature,
            top_k=g.top_k,
            max_k=g.max_k,
            top_p=g.top_p,
            min_top_p=g.min_top_p,
            pinned_bitmask=g.pinned_bitmask,
            wait_payload=g.wait_payload,
            device_bitmask_scratch=g.device_bitmask_scratch,
            draft_probs_full=g.draft_probs_full,
        )
        return tuple(Tensor.from_graph_value(v) for v in outputs)


class UnifiedMTPHYV4Model(ModuleV3PipelineModelWithKVCache[TextContext]):
    """Hy4 on one GPU or tensor- and expert-parallel on several, its MTP
    layer drafting ``num_speculative_tokens`` steps per iteration."""

    model_config_cls: ClassVar[type[Any]] = UnifiedMTPHYV4Config
    batch_processor_cls: ClassVar[type[UnifiedMTPHYV4BatchProcessor]] = (
        UnifiedMTPHYV4BatchProcessor
    )

    _nn_model: UnifiedMTPHYV4

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> HYV4Config:
        check_parallelism(self.pipeline_config, len(self.devices))
        config = UnifiedMTPHYV4Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.max_batch_context_length = (
            self.planned_max_batch_total_tokens
            or config.max_batch_context_length
        )
        config.quant_config = parse_quant_config(
            self.huggingface_config, state_dict, self.dtype
        )
        return config

    @override
    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        # As HYV4Model: fp32 tensors (iHC, sinks, router bias) are declared
        # explicitly.
        del state_dict, model_config
        return DType.bfloat16

    @override
    def _instantiate_module(self, model_config: HYV4Config) -> UnifiedMTPHYV4:
        speculative = self.pipeline_config.speculative
        assert speculative is not None
        args = (
            model_config,
            speculative,
            self.pipeline_config.needs_bitmask_constraints,
        )
        if len(self.devices) == 1:
            self._nn_model = UnifiedMTPHYV4(*args).to(self.devices[0])
            return self._nn_model
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), (TP,))
        with default_device(mesh):
            self._nn_model = UnifiedMTPHYV4(*args)
        self._nn_model.shard()
        return self._nn_model

    @override
    def _get_compile_input_types(self, model_config: Any) -> tuple[Any, ...]:
        del model_config
        return self._nn_model.input_types()

    @override
    def execute(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
        # The pipeline stages the bitmask in pinned host memory, which the
        # Tensor-level argument check rejects as an accelerator buffer; the
        # engine takes it, as it does for the graph-API spec-decode models.
        assert isinstance(self.model, CompiledCallable)
        accepted, next_tokens, next_draft_tokens = self.model.execute_raw(
            *model_inputs.buffers
        )
        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=accepted,
            next_tokens=next_tokens,
            next_draft_tokens=next_draft_tokens,
        )
