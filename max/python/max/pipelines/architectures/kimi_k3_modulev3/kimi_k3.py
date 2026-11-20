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
"""The Kimi K3 text tower (ModuleV3), registered for the ``DECODER`` role.

A speculative pairing builds the tower through the framework's registered
builder, which hands the decoder the config the pairing stamped (its tap
ids and the draft-aware MLA plan cap); serving on its own still builds
the tap-free forward.
"""

from __future__ import annotations

from typing import ClassVar

from max import tree
from max.dtype import DType
from max.nn import Module
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import default_device, default_dtype

from .model import (
    KimiK3Model,
    expert_parallel_kimi_k3_model,
    tensor_parallel_kimi_k3_model,
)
from .model_config import KimiK3Config
from .spec_mla import kimi_k3_mla_table_rows


class KimiK3Decoder(Module[[DecoderInputs], DecoderOutputs]):
    """Kimi K3's text tower, registered for the ``DECODER`` role.

    It runs tensor-parallel, or with its routed experts spread across the
    devices under expert parallelism. A speculative pairing builds the
    same decoder, on a config whose finalize stamped the draft's tap ids
    (``return_hidden_states``/``target_layer_ids``) and whose
    ``apply_runtime`` sized the draft-aware MLA plan cap, so the tower it
    builds reads the attention-residual mixture the drafter was trained
    against (``KimiK3TapMode``); a DSpark pairing pairs with the TP trunk
    alone (``KimiK3Config.expert_parallel_config`` refuses the rest).
    """

    graph_name: ClassVar[str] = "KimiK3Model"

    def __init__(self, config: KimiK3Config) -> None:
        super().__init__()
        mesh = config.mesh
        if mesh is None:
            raise ValueError("Kimi K3 needs a mesh: apply the runtime first")
        # Everything but the routed experts is bf16 under MXFP4.
        dtype = (
            DType.bfloat16 if config.quant_config is not None else config.dtype
        )
        with default_device(mesh), default_dtype(dtype):
            model = KimiK3Model(
                config.model,
                kv_params=config.kv_params,
                return_logits=config.return_logits,
                return_hidden_states=config.return_hidden_states,
                target_layer_ids=config.target_layer_ids,
                tap_mode=config.tap_mode,
                mla_buffer_size=config.max_batch_context_length,
                # The rope table is indexed at one request's positions,
                # never the batch-wide window the cap spans; see
                # `kimi_k3_mla_table_rows`.
                mla_rope_table_rows=kimi_k3_mla_table_rows(
                    max_seq_len=config.max_seq_len,
                    page_size=config.attn_kv_params.page_size,
                    # The draft width the config's own cache leaf carries:
                    # `construct_kv_params` sized it from the pipeline's
                    # speculative config.
                    num_draft_tokens=config.attn_kv_params.num_draft_tokens,
                ),
                quant_config=config.quant_config,
                ep_config=config.ep_config,
                max_tokens_per_expert=config.max_tokens_per_expert,
            )
            if config.ep_config is not None:
                expert_parallel_kimi_k3_model(model, mesh)
            elif mesh.num_devices > 1:
                tensor_parallel_kimi_k3_model(model, mesh)
        # The checkpoint names the tower's weights from its root.
        model.name_transparent = True
        self.model = model
        # Only the MLA cache's prefill plan reads the batch's context length.
        self._reads_context_length = config.mla_latent_cache

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        """Runs the tower; it reads the exchange off the tail of its inputs."""
        assert inputs.batch_context_lengths is not None
        return self.model(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            *tree.leaves(inputs.kv),
            *(
                inputs.batch_context_lengths[:1]
                if self._reads_context_length
                else ()
            ),
            *(tree.leaves(inputs.ep) if inputs.ep is not None else ()),
        )
