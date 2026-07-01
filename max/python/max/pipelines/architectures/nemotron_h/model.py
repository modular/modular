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
"""Nemotron-H pipeline model (hybrid Mamba-2 + NoPE attention + relu2 MLP).

Cribbed from ``qwen3_5/model.py`` (hybrid KV-cache + slot-indexed state pools,
``Llama3Inputs`` subclass): a single language graph with a paged KV cache for
the 4 attention layers plus per-mamba-layer conv (bf16, in-place) and SSM (fp32,
in-place slot-indexed via ``mamba2_ssd_chunk_scan_varlen_fwd_inplace``) state
pools. The SSD kernel serves both prefill and
decode; the per-request slot is zeroed on claim, so passing ``has_initial_state
= True`` with the zeroed initial state is identical to a fresh prefill (no
separate prefill/decode graph needed).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferValue, DeviceRef, Graph, TensorValue
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm import (
    Signals,  # noqa: F401  (kept for parity; unused single-GPU)
)
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.modeling.types import RequestID
from max.profiler import traced

from ..llama3.model import Llama3Inputs, LlamaModelBase
from .model_config import NemotronHConfig, build_fp8_quant_config
from .nemotron_h import NemotronH
from .state_cache import NemotronHStateCache

logger = logging.getLogger("max.pipelines")


@dataclass
class NemotronHInputs(Llama3Inputs):
    """Inputs for Nemotron-H: ragged tokens + hybrid SSM/conv state.

    Beyond the standard Llama3 ragged inputs, carries a uint32 ``slot_idx`` into
    the per-request pools, the per-mamba-layer conv pools (bf16, mutated in
    place by ``causal_conv1d_varlen_fwd``), the per-mamba-layer SSM pools (fp32,
    mutated in-place by ``mamba2_ssd_chunk_scan_varlen_fwd_inplace``), and a
    ``[batch]`` ``has_initial_state`` (the slots are zeroed on claim, so a
    fresh request's zeroed initial state matches a from-scratch prefill).
    """

    slot_idx: Buffer | None = None
    conv_pools: list[Buffer] | None = None
    ssm_pools: list[Buffer] | None = None
    has_initial_state: Buffer | None = None
    request_ids: list[RequestID] | None = None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.slot_idx is not None
        assert self.conv_pools is not None
        assert self.ssm_pools is not None
        assert self.has_initial_state is not None
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            self.slot_idx,
            *self.conv_pools,
            *self.ssm_pools,
            self.has_initial_state,
        )


class NemotronHModel(LlamaModelBase):
    """Nemotron-H pipeline model (hybrid Mamba-2 + attention)."""

    model_config_cls: ClassVar[type[Any]] = NemotronHConfig
    # Bypass the Llama batch processor: this model builds inputs directly
    # (it needs the state pools + slot_idx the batch processor does not know).
    batch_processor_cls: ClassVar[Any] = None
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"

    _state_cache: NemotronHStateCache | None = None
    _slot_idx_prealloc: Buffer | None = None
    _has_initial_state_prealloc: Buffer | None = None
    _num_mamba_layers: int = 0

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states,
        )

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        max_batch_size = self.pipeline_config.runtime.max_batch_size
        assert max_batch_size is not None, (
            "max_batch_size must be set in runtime config"
        )

        with CompilationTimer("model") as timer:
            graph = self._build_graph(self.weights, self.adapter)
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        # Allocate the per-request state cache + per-step preallocs.
        self._state_cache = NemotronHStateCache(
            num_mamba_layers=self._num_mamba_layers,
            conv_dim=self._conv_dim,
            conv_kernel=self._conv_kernel,
            nheads=self._mamba_nheads,
            head_dim=self._mamba_head_dim,
            dstate=self._dstate,
            max_slots=max_batch_size,
            device=self.devices[0],
            conv_dtype=self._model_dtype,
        )
        if not is_virtual_device_mode():
            self._slot_idx_prealloc = Buffer(
                shape=[max_batch_size],
                dtype=DType.uint32,
                device=self.devices[0],
            )
            # has_initial_state is always all-True: a fresh request's slot is
            # zeroed on claim, so loading the (zero) initial state is identical
            # to a from-scratch prefill.
            self._has_initial_state_prealloc = Buffer.from_numpy(
                np.ones((max_batch_size,), dtype=np.bool_)
            ).to(self.devices[0])
        return model

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> Graph:
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )

        device_ref = DeviceRef.from_device(self.devices[0])
        assert isinstance(self.kv_params, KVCacheParams)
        model_config = NemotronHConfig.from_hf(
            self.pipeline_config,
            self.huggingface_config,
            self.dtype,
            self.kv_params,
            [device_ref],
        )

        # FP8 is per-module: a Linear is FP8 iff its weight_scale is present in
        # the checkpoint. Build the FP8 layer sets + per-tensor static config.
        quant_config = build_fp8_quant_config(
            self.huggingface_config, state_dict
        )
        model_config.populate_fp8_layers(state_dict)

        nn_model = NemotronH(
            model_config,
            quant_config=quant_config,
            return_logits=self.return_logits,
        )

        # Diagnostics: log missing/unused weights (strict=False drops silently).
        expected = set(nn_model.raw_state_dict().keys())
        provided = set(state_dict.keys())
        missing = expected - provided
        if missing:
            logger.warning(
                f"Nemotron-H load_state_dict: {len(missing)} MISSING weights"
                f" (not in checkpoint): {sorted(missing)[:30]}"
                + ("..." if len(missing) > 30 else "")
            )
        unused = provided - expected
        if unused:
            logger.info(
                f"Nemotron-H load_state_dict: {len(unused)} unused checkpoint"
                f" keys: {sorted(unused)[:20]}"
                + ("..." if len(unused) > 20 else "")
            )

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        self.state_dict = nn_model.state_dict()

        # Save dims for state-pool allocation.
        self._num_mamba_layers = nn_model.num_mamba_layers
        self._conv_dim = nn_model.conv_dim
        self._conv_kernel = nn_model.conv_kernel
        self._mamba_nheads = nn_model.mamba_nheads
        self._mamba_head_dim = nn_model.mamba_head_dim
        self._dstate = nn_model.dstate
        self._model_dtype = model_config.dtype

        num_mamba = self._num_mamba_layers
        with Graph(
            "nemotron_h",
            input_types=nn_model.input_types(self.kv_params),
        ) as graph:
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                *variadic,
            ) = graph.inputs

            # KV inputs come first, then slot_idx, conv pools, ssm pools, and
            # has_initial_state (matching NemotronH.input_types ordering).
            kv_count = (
                len(variadic) - 1 - num_mamba * 2 - 1
            )  # slot_idx + pools + has_initial_state
            kv_inputs = variadic[:kv_count]
            kv_collections = self._unflatten_kv_inputs(kv_inputs)

            idx = kv_count
            slot_idx_g: TensorValue = variadic[idx].tensor
            idx += 1
            conv_pools: list[BufferValue] = [
                variadic[idx + i].buffer for i in range(num_mamba)
            ]
            idx += num_mamba
            ssm_pools: list[BufferValue] = [
                variadic[idx + i].buffer for i in range(num_mamba)
            ]
            idx += num_mamba
            has_initial_state_g: TensorValue = variadic[idx].tensor

            outputs = nn_model(
                tokens.tensor,
                input_row_offsets.tensor,
                return_n_logits.tensor,
                kv_collections,
                slot_idx_g,
                conv_pools,
                ssm_pools,
                has_initial_state_g,
            )
            graph.output(*outputs)
            return graph

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, NemotronHInputs)
        assert model_inputs.kv_cache_inputs is not None

        model_outputs = self.model.execute(*model_inputs.buffers)

        # Both the conv pools and SSM pools are mutated in place by their
        # respective inplace ops; the only graph output is the logits (plus
        # the optional return_n_logits extras).
        logits = model_outputs[0]
        assert isinstance(logits, Buffer)
        if len(model_outputs) > 1:
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=logits,
                logit_offsets=model_outputs[2],
            )
        return ModelOutputs(logits=logits, next_token_logits=logits)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Any]],
        kv_cache_inputs: Any = None,
        return_n_logits: int = 1,
    ) -> NemotronHInputs:
        if len(replica_batches) != 1:
            raise ValueError("Nemotron-H does not support data parallelism > 1")
        context_batch = replica_batches[0]
        request_ids = [ctx.request_id for ctx in context_batch]

        assert self._state_cache is not None
        assert self._slot_idx_prealloc is not None
        assert self._has_initial_state_prealloc is not None
        for rid in request_ids:
            self._state_cache.claim(rid)
        slot_idx = self._state_cache.slot_idx_for(
            request_ids, self._slot_idx_prealloc
        )
        batch_size = len(request_ids)
        has_initial_state = self._has_initial_state_prealloc[:batch_size]

        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        signal_buffers = self.signal_buffers

        return NemotronHInputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            signal_buffers=signal_buffers,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            slot_idx=slot_idx,
            conv_pools=self._state_cache.conv_pools,
            ssm_pools=self._state_cache.ssm_pools,
            has_initial_state=has_initial_state,
            request_ids=request_ids,
        )

    def release(self, request_id: RequestID) -> None:
        if self._state_cache is not None:
            self._state_cache.release(request_id)
