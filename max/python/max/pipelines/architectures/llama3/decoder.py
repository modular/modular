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
"""The Llama 3 decoder an architecture registers for the ``DECODER`` role."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from max import tree
from max.driver import DLPackArray
from max.graph import TensorValue
from max.graph.weights import WeightData
from max.nn.legacy.data_parallelism import split_batch
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.layer import Module
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.module_registration import finalize_config
from max.pipelines.lib.registration_graph_api import decoder_outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig

from .data_parallel_llama import DataParallelLlama
from .distributed_llama import DistributedLlama3
from .llama3 import Llama3
from .model_config import Llama3Config


@dataclass(kw_only=True)
class Llama3DecoderConfig(Llama3Config):
    """The config an architecture registers with :class:`Llama3Decoder`.

    The framework finalizes it with the outputs the pipeline requires.
    """

    huggingface_model_type: str = "llama3"
    """The checkpoint config's ``model_type``, which names the
    tensor-parallel graph."""

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
        return_logits: ReturnLogits | ModelOutputConfig,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        """Completes the config in place with checkpoint-dependent fields."""
        if isinstance(return_logits, ModelOutputConfig):
            return_hidden_states = return_logits.return_hidden_states
            return_logits = return_logits.return_logits
        self.huggingface_model_type = getattr(
            huggingface_config, "model_type", "llama3"
        )
        super().finalize(
            huggingface_config=huggingface_config,
            state_dict=dict(state_dict),
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            norm_method=norm_method,
            attention_bias=attention_bias,
        )


# The checkpoint names Llama 3's weights from its root, so the decoder that
# holds one of these adds no prefix.


class _Llama3(Llama3):
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class _DistributedLlama3(DistributedLlama3):
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class _DataParallelLlama(DataParallelLlama):
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class Llama3Decoder(Module):
    """Llama 3, registered for the ``DECODER`` role.

    It runs on one device, tensor-parallel across several, or as one
    replica per device under data parallelism, as the config says.
    """

    graph_name = "llama3"
    """The name the graph compiled under before registration, which keys
    precompiled MEFs. The tensor-parallel graph compiled under the
    checkpoint's ``model_type``, which an instance sets."""

    supports_data_parallel = True
    override_quantization_encoding = True
    # The checkpoint's `rope_freqs.weight` isn't read (MODELS-550).
    strict_state_dict_loading = False

    def __init__(self, config: Llama3Config) -> None:
        """Builds the model for the config's devices and parallelism."""
        super().__init__()
        self.kv_params = config.kv_params
        self._devices = list(config.devices)
        self._replicas = (
            len(config.devices) if config.data_parallel_degree > 1 else 0
        )
        self.model: _Llama3 | _DistributedLlama3 | _DataParallelLlama
        if self._replicas:
            self.model = _DataParallelLlama(config)
        elif len(config.devices) > 1:
            self.model = _DistributedLlama3(config)
            self.graph_name = getattr(
                config, "huggingface_model_type", type(self).graph_name
            )
        else:
            self.model = _Llama3(config)

    def load_state_dict(
        self,
        state_dict: Mapping[str, DLPackArray | WeightData],
        *,
        override_quantization_encoding: bool = False,
        weight_alignment: int | None = None,
        strict: bool = True,
    ) -> None:
        """Loads the checkpoint, into every replica under data parallelism.

        Each replica's weights go under its index in ``models``.
        """
        if self._replicas:
            # Shallow copies share the buffer; each replica needs its own
            # because loading rewrites a value's dtype and shape in place.
            state_dict = {
                f"models.{replica}.{name}": copy.copy(value)
                for replica in range(self._replicas)
                for name, value in state_dict.items()
            }
        super().load_state_dict(
            state_dict,
            override_quantization_encoding=override_quantization_encoding,
            weight_alignment=weight_alignment,
            strict=strict,
        )

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        """Runs the model on the batch and returns its logits."""
        kv_collections = tree.leaves(
            self.kv_params.unflatten_kv_inputs(iter(tree.leaves(inputs.kv))),
            leaf=KVCacheInputsPerDevice,
        )
        if isinstance(self.model, _DataParallelLlama):
            assert inputs.data_parallel_splits is not None
            tokens, row_offsets = split_batch(
                self._devices,
                inputs.tokens,
                inputs.input_row_offsets,
                inputs.data_parallel_splits,
            )
            return decoder_outputs(
                self.model(
                    [
                        (
                            tokens[i],
                            kv_collections[i],
                            inputs.return_n_logits,
                            row_offsets[i],
                        )
                        for i in range(self._replicas)
                    ]
                )
            )
        if isinstance(self.model, _DistributedLlama3):
            assert inputs.signals is not None
            return decoder_outputs(
                self.model(
                    inputs.tokens,
                    list(inputs.signals),
                    kv_collections,
                    inputs.return_n_logits,
                    inputs.input_row_offsets,
                )
            )
        return decoder_outputs(
            self.model(
                inputs.tokens,
                kv_collections[0],
                inputs.return_n_logits,
                inputs.input_row_offsets,
            )
        )


class Llama3TargetDecoder(Llama3Decoder):
    """Llama 3 as ``LlamaForCausalLM`` registers it.

    It is also the target of Llama 3's speculative-decoding pairings, which
    call the model it wraps directly.

    TODO(kathywu): Drop :meth:`build_module` once speculative decoding
    composes the registered decoder itself.
    """

    @classmethod
    def build_module(
        cls, context: ModelLoadContext, config_cls: type[Llama3DecoderConfig]
    ) -> tuple[Llama3 | DistributedLlama3, Llama3Config]:
        """Builds the model a speculative-decoding graph calls as its target.

        Args:
            context: The checkpoint and outputs the target is built for.
            config_cls: The registered config class.

        Returns:
            The model on one device or tensor-parallel, with the checkpoint
            loaded, and its finalized config.

        Raises:
            ValueError: If the config is data-parallel, whose replicas only
                the decoder's own graph builds.
        """
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        config = finalize_config(
            config_cls.initialize(
                context.pipeline_config, max_seq_len=context.max_seq_len
            ),
            context.huggingface_config,
            state_dict,
            ModelOutputConfig(
                return_logits=context.return_logits,
                return_hidden_states=context.return_hidden_states,
            ),
        )
        assert isinstance(config, Llama3Config)
        if context.target_layer_ids:
            config.target_layer_ids = list(context.target_layer_ids)
        if config.data_parallel_degree > 1:
            raise ValueError(
                "Llama 3 builds data-parallel replicas inside its graph;"
                " there is no single module to return."
            )
        module: Llama3 | DistributedLlama3 = (
            DistributedLlama3(config)
            if len(config.devices) > 1
            else Llama3(config)
        )
        module.load_state_dict(
            state_dict,
            override_quantization_encoding=cls.override_quantization_encoding,
            weight_alignment=1,
            strict=cls.strict_state_dict_loading,
        )
        return module, config
