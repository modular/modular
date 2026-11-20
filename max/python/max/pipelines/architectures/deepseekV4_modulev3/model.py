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
"""The DeepSeek-V4-Flash pipeline model on ModuleV3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache import KVCacheInputs
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import ModelInputs, ModuleV3PipelineModelWithKVCache
from max.pipelines.lib.interfaces.batch_processor import (
    ModuleV3SingleReplicaBatchProcessor,
)
from max.tensor import default_device
from max.tensor.sharding import DeviceMesh
from typing_extensions import override

from .deepseekV4 import DeepseekV4
from .layers.quantization import fp8_block_quant_config
from .model_config import DeepseekV4Config

__all__ = ["DeepseekV4Model"]


# Extracted from deepseekV4/model.py at dfc8109fc0c.
def set_checkpoint_dtypes(
    config: DeepseekV4Config,
    huggingface_config: Any,
    *,
    norm_dtype: DType,
    embed_dtype: DType,
) -> None:
    """Sets what the config reads off the adapted checkpoint.

    Args:
        config: The config to finish.
        huggingface_config: The checkpoint's HuggingFace config.
        norm_dtype: The adapted dtype of the checkpoint's norm weights.
        embed_dtype: The stored dtype of the checkpoint's bf16 weights,
            ``embed`` among them.

    Raises:
        ValueError: If the encoding is fp8 but the checkpoint declares no fp8
            ``quantization_config``.
    """
    config.norm_dtype = norm_dtype
    if config.quantization_encoding == "float8_e4m3fn":
        # The encoding names the fp8 projections only. The activation dtype
        # is the reference's default dtype, bf16, which is also what the
        # checkpoint stores ``embed`` / ``ffn.gate`` /
        # ``indexer.weights_proj`` in and the reference declares them as;
        # those follow ``config.dtype``, and so does ``head`` (bf16 GEMM, see
        # ``head_logits``). The other weights the reference declares float32
        # (norms, compressor ``wkv`` / ``wgate``) are declared so by the
        # modules and upcast by the adapter.
        quant_config = fp8_block_quant_config(
            getattr(huggingface_config, "quantization_config", None),
            config.num_hidden_layers,
        )
        if quant_config is None:
            raise ValueError(
                "quantization_encoding float8_e4m3fn needs a checkpoint "
                "that declares its fp8 quantization_config"
            )
        config.quant_config = quant_config
        config.dtype = embed_dtype


@dataclass
class DeepseekV4Inputs(ModelInputs):
    """Inputs for one DeepSeek-V4 ModuleV3 execution."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *tree.leaves(self.kv_cache_inputs),
        )


class DeepseekV4ModuleV3BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, DeepseekV4Inputs]
):
    """Ragged single-replica batching, in the ModuleV3 input order."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> DeepseekV4Inputs:
        return DeepseekV4Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )


class DeepseekV4Model(ModuleV3PipelineModelWithKVCache[TextContext]):
    """DeepSeek-V4-Flash pipeline model (ModuleV3).

    The KV params come from ``DeepseekV4Config.construct_kv_params`` through
    ``model_config_cls``, so the cache tree is the V2 model's seven leaves.
    """

    model_config_cls: ClassVar[type[Any]] = DeepseekV4Config
    batch_processor_cls: ClassVar[type[DeepseekV4ModuleV3BatchProcessor]] = (
        DeepseekV4ModuleV3BatchProcessor
    )

    @override
    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV4Config:
        config = self.arch_config_as(DeepseekV4Config)
        config.return_logits = self.return_logits
        config.return_hidden_states = self.return_hidden_states
        set_checkpoint_dtypes(
            config,
            self.huggingface_config,
            norm_dtype=state_dict["norm.weight"].dtype,
            embed_dtype=state_dict["embed.weight"].dtype,
        )
        return config

    @override
    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        # ``set_checkpoint_dtypes`` resolved this to the activation dtype; the
        # fp8 encoding names only the block-quantized projections.
        return model_config.dtype

    @override
    def _instantiate_module(self, model_config: Any) -> Any:
        if (
            self.return_logits is not ReturnLogits.LAST_TOKEN
            or self.return_hidden_states is not ReturnHiddenStates.NONE
        ):
            raise NotImplementedError(
                f"return_logits={self.return_logits} with "
                f"return_hidden_states={self.return_hidden_states} is not "
                "implemented for DeepSeek-V4 on ModuleV3; only last-token "
                "logits are."
            )
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), ("tp",))
        with default_device(mesh):
            return DeepseekV4(model_config, self.kv_params, mesh)
