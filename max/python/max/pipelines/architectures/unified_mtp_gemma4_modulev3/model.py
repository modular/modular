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

"""Gemma4 with MTP on ModuleV3: target, draft and driver in one module."""

from __future__ import annotations

from functools import cached_property
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental import functional as F
from max.experimental.compilation import CompiledCallable
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightData
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    VisionRawInputs,
    create_empty_embeddings,
)
from max.pipelines.architectures.gemma4_modulev3.vision_model.vision_model import (
    Gemma4VisionModel,
)
from max.pipelines.lib import ModelInputs, UnifiedEagleOutputs
from max.pipelines.lib.graph_specs import GraphRole
from max.profiler import traced
from typing_extensions import override

from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..unified_mtp_gemma4.model import UnifiedMTPGemma4Model
from .batch_processor import UnifiedMTPGemma4ModuleV3BatchProcessor
from .gemma4_assistant_modulev3 import Gemma4AssistantModuleV3
from .unified_mtp_gemma4_modulev3 import (
    UnifiedMTPGemma4ModuleV3,
    UnifiedMTPGemma4ModuleV3Target,
)
from .weight_adapters import convert_draft_state_dict


class UnifiedMTPGemma4ModuleV3Model(UnifiedMTPGemma4Model):
    """Gemma4 with MTP on ModuleV3: target, draft and driver in one module.

    Inherits the graph-API MTP model's state-dict plumbing, draft-config
    derivation and vision packing; overrides the compile path to trace the
    :class:`UnifiedMTPGemma4ModuleV3` wrapper — whose ``forward`` runs the
    sequential spec-decode driver over the traced target and draft — into
    one compiled graph, plus the ModuleV3 batch processor whose buffer
    order matches the compiled signature.
    """

    batch_processor_cls: ClassVar[
        type[UnifiedMTPGemma4ModuleV3BatchProcessor]
    ] = UnifiedMTPGemma4ModuleV3BatchProcessor

    _compiled: CompiledCallable[..., tuple[Any, ...]]

    @cached_property
    def signal_buffers(self) -> list[Buffer]:
        # A compiled graph declares signal buffers only when it spans more
        # than one device.
        if len(self.devices) <= 1:
            return []
        return Signals.allocate(self.devices)

    # TODO(kathywu): Return ModuleV3 graph specs once
    # GraphSpec takes ModuleV3 modules, instead of compiling here.
    @traced
    @override
    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        del session
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)

        module, weights = self._build_module(model_config)
        self._compiled = module.compile(*module.input_types(), weights=weights)
        self.state_dict = dict(self._compiled.weights)
        # The engine Model keeps the graph-API ABI the V2 base's __init__
        # and execute expect (buffers in, Buffers out), while _compiled
        # stays for ModuleV3-style calls (vision packing, signal
        # queries).
        model = self._compiled.engine_model

        models: dict[str, Model] = {GraphRole.SPEC_DECODER: model}
        vision_model = self._compile_vision_model(model_config)
        if vision_model is not None:
            models[GraphRole.VISION] = vision_model
        return models

    def _build_module(
        self, model_config: Gemma4ForConditionalGenerationConfig
    ) -> tuple[UnifiedMTPGemma4ModuleV3, dict[str, WeightData]]:
        """Builds the fused module and the dual-prefixed weights registry.

        The traced target nests under ``target.*`` and the draft under
        ``draft.*`` (the names the sequential driver registered its
        ``target_model`` / ``draft_model`` under). The draft's
        ``embed_tokens`` is the SAME module object as the target's, so the
        state tree yields the leaf under both paths and the registry needs
        an entry for each.
        """
        # The tree's children are typed broadly (any leaf kind); the
        # draft's constructor takes plain KVCacheParams, so narrow here
        # as the graph-API model does at its own use site.
        assert isinstance(self._target_sliding_kv_params, KVCacheParams)
        assert isinstance(self._target_global_kv_params, KVCacheParams)
        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))

        # Under NVFP4 the packed dtype only feeds the quantized linears;
        # every other leaf (and the whole bf16 draft) builds in the
        # unquantized dtype.
        build_dtype = (
            model_config.unquantized_dtype
            if model_config.text_config.quant_config is not None
            else model_config.dtype
        )
        with F.lazy(), default_dtype(build_dtype), default_device(mesh):
            target = UnifiedMTPGemma4ModuleV3Target(model_config, mesh)
            draft = Gemma4AssistantModuleV3(
                self._draft_config,
                self._target_layer_types,
                self._target_sliding_kv_params,
                self._target_global_kv_params,
                mesh,
            )
            # Share the target's embedding with the draft BEFORE the trace:
            # the state tree keeps the alias, so both checkpoint paths load
            # the one shared weight.
            draft.embed_tokens = target.embed_tokens
            module = UnifiedMTPGemma4ModuleV3(
                target,
                draft,
                model_config,
                kv_params=self.kv_params,
                speculative_config=self.pipeline_config.speculative,
                enable_structured_output=(
                    self.pipeline_config.needs_bitmask_constraints
                ),
                use_greedy_acceptance=(
                    self.pipeline_config.speculative.use_greedy_acceptance
                    if self.pipeline_config.speculative is not None
                    else False
                ),
            )

        # The V2 language adapter yields bare paths, which the fused module's
        # target subtree expects: it nests under ``target.*`` directly.
        target_weights = self._target_state_dict
        draft_weights = convert_draft_state_dict(self._draft_state_dict)
        shared_embedding = target_weights["embed_tokens.weight"]
        weights = {
            **{f"target.{k}": v for k, v in target_weights.items()},
            **{f"draft.{k}": v for k, v in draft_weights.items()},
            # The aliased embed_tokens yields BOTH paths in the traced
            # tree, and the sharding machinery walks each — so the
            # registry must hold an entry under each name.
            "draft.embed_tokens.weight": shared_embedding,
        }
        return module, weights

    def _compile_vision_model(
        self, model_config: Gemma4ForConditionalGenerationConfig
    ) -> Model | None:
        """Compiles the vision tower, or ``None`` when text-only.

        Mirrors the ModuleV3 Gemma4 arch's vision compile (the tower is
        never quantized — modelopt ignore-lists it — so it always builds
        in the unquantized BF16 dtype, on device 0), but hands back the
        compiled callable's engine ``Model`` rather than the callable:
        the inherited graph-API flow calls it with Buffers and reads
        Buffers back, which is the engine Model's ABI.
        """
        if model_config.vision_config is None:
            return None

        device_ref = DeviceRef.from_device(self.devices[0])
        with F.lazy(), default_dtype(model_config.unquantized_dtype):
            vision_nn = Gemma4VisionModel(model_config)
            vision_nn.to(self.devices[0])

        patch_dim = 3 * model_config.vision_config.patch_size**2
        input_types = (
            TensorType(
                model_config.unquantized_dtype,
                shape=["total_patches", patch_dim],
                device=device_ref,
            ),
            TensorType(
                DType.int32, shape=["total_patches", 2], device=device_ref
            ),
            TensorType(
                DType.uint32, shape=["num_images_plus_1"], device=device_ref
            ),
            TensorType(
                DType.int32,
                shape=["num_pooled_tokens", "max_pool_patches"],
                device=device_ref,
            ),
            TensorType(DType.uint32, shape=[], device=DeviceRef.CPU()),
        )
        compiled = vision_nn.compile(
            *input_types, weights=self._vision_weights_dict
        )
        return compiled.engine_model

    def _run_vision_encoder(self, raw: VisionRawInputs) -> list[Buffer]:
        """Runs the eager vision tower over the packed pixels.

        The tower's engine ``Model`` takes the packed buffers device 0
        holds (the same replica reaches every device) and returns the
        embeddings buffer; broadcast it to each device as the graph-API
        model does.
        """
        if self.vision_model is None:
            raise ValueError(
                "This checkpoint is served text-only (no vision encoder"
                " is loaded); image and video inputs are not supported."
            )
        outputs = self.vision_model(
            raw.patches_flat[0],
            raw.pixel_position_ids[0],
            raw.cu_seqlens[0],
            raw.pool_gather_index[0],
            raw.max_seq_len,
        )
        device0_embeddings = outputs[0]
        return [device0_embeddings.to(device) for device in self.devices]

    def execute(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
        """Execute the compiled MTP graph and return the spec-decode outputs.

        Images appear only during prefill (``draft_tokens`` is ``[batch,
        0]``); decode steps carry the empty vision-merge inputs, so the
        graph's scatter is a no-op there.
        """
        model_outputs = self.model.execute(*model_inputs.buffers)
        assert len(model_outputs) == 3, (
            f"Expected 3 outputs, got {len(model_outputs)}"
        )
        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=model_outputs[0],
            next_tokens=model_outputs[1],
            next_draft_tokens=model_outputs[2],
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Kept from the graph-API model: hit on every text-only / decode
        step, so it must not allocate per call.
        """
        if not hasattr(self, "_cached_empty_embeddings"):
            self._cached_empty_embeddings = create_empty_embeddings(
                devices,
                self.huggingface_config.text_config.hidden_size,
                self.config.unquantized_dtype,
            )
        return self._cached_empty_embeddings
