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
"""TP=1 vs TP=2 parity for ``KimiK3Model`` loaded the way serving loads it.

A safetensors checkpoint goes through the weight adapter into a module built
under ``F.lazy()`` and compiled through ``compile_and_init_graphs``, so the
loader, not the module, splits the real arrays across devices. The eager TP
tests assign weights to the module directly and cannot catch a fused weight
that is only laid out right on the module's own parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _tp_model_test_utils import (
    kv_params_and_inputs,
    random_weights,
    rel_error,
    tiny_model_config,
)
from max import functional as F
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import load_weights
from max.pipelines.architectures.kimi_k3_modulev3.model import (
    KimiK3Model,
    KimiK3ModelConfig,
    tensor_parallel_kimi_k3_model,
)
from max.pipelines.architectures.kimi_k3_modulev3.weight_adapters import (
    convert_kimi_k3_state_dict,
)
from max.pipelines.lib.graph_specs import (
    GraphRole,
    GraphSpec,
    compile_and_init_graphs,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import (
    _cast_to_parameter_dtypes,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from safetensors.numpy import save_file

_TEXT = "language_model.model"


class _FakeTextConfig(SimpleNamespace):
    """Minimal stand-in for the HuggingFace text sub-config."""

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _text_config(config: KimiK3ModelConfig) -> _FakeTextConfig:
    """The ``config.json`` text config ``config`` is read back from."""
    return _FakeTextConfig(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_hidden_layers=config.num_hidden_layers,
        first_k_dense_replace=config.first_k_dense_replace,
        # 1-indexed in the checkpoint's config.
        linear_attn_config={
            "kda_layers": [i + 1 for i in config.kda_layer_indices],
            "full_attn_layers": [i + 1 for i in config.full_attn_layer_indices],
            "num_heads": config.kda_num_heads,
            "head_dim": config.kda_head_dim,
            "short_conv_kernel_size": config.kda_conv_kernel_size,
        },
        num_attention_heads=config.num_attention_heads,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        attn_res_block_size=config.attn_res_block_size,
        intermediate_size=config.intermediate_size,
        routed_expert_hidden_size=config.routed_expert_hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_experts=config.num_experts,
        num_experts_per_token=config.num_experts_per_token,
        num_shared_experts=config.num_shared_experts,
        rms_norm_eps=config.rms_norm_eps,
        activation_situ_beta=config.situ_beta,
        activation_situ_linear_beta=config.situ_linear_beta,
    )


def _write_checkpoint(raw: dict[str, np.ndarray], path: Path) -> None:
    """Writes ``random_weights``' arrays under their checkpoint names."""
    checkpoint: dict[str, np.ndarray] = {}
    for name, array in raw.items():
        if name == "lm_head.weight":
            checkpoint["language_model.lm_head.weight"] = array
        elif name.endswith(".conv_weight"):
            # Unfused into the checkpoint's three `[C, 1, K]` depthwise convs.
            attn = name.removesuffix(".conv_weight")
            for axis, conv in zip("qkv", np.split(array, 3), strict=True):
                checkpoint[f"{_TEXT}.{attn}.{axis}_conv1d.weight"] = conv[
                    :, None, :
                ]
        elif name.endswith(".a_log"):
            # Per-head in the checkpoint; the adapter broadcasts it.
            attn = name.removesuffix(".a_log")
            checkpoint[f"{_TEXT}.{attn}.A_log"] = np.ascontiguousarray(
                array[:, 0]
            )
        elif name.endswith(".dt_bias"):
            checkpoint[f"{_TEXT}.{name}"] = array.reshape(-1)
        else:
            checkpoint[f"{_TEXT}.{name}"] = array
    save_file(checkpoint, str(path))


def _pipeline_config(num_devices: int) -> SimpleNamespace:
    """The fields of a float32, ``num_devices``-GPU TP ``PipelineConfig``
    the weight adapter reads."""
    return SimpleNamespace(
        model=SimpleNamespace(
            quantization_encoding="float32",
            _resolved_dtype_cast=(None, None),
            data_parallel_degree=1,
            device_specs=[None] * num_devices,
        ),
        runtime=SimpleNamespace(ep_size=1),
    )


def _serve_logits(
    config: KimiK3ModelConfig,
    checkpoint: Path,
    devices: Sequence[Accelerator],
    input_ids: np.ndarray,
    input_row_offsets: np.ndarray,
    prompt_lens: list[int],
) -> np.ndarray:
    """Loads and runs the model as ``ModuleV3PipelineModelWithKVCache``
    and Kimi K3's ``_instantiate_module`` do."""
    weights = load_weights([checkpoint])
    state_dict = convert_kimi_k3_state_dict(
        dict(weights.items()),
        _text_config(config),
        pipeline_config=_pipeline_config(len(devices)),
    )
    mesh = DeviceMesh(tuple(devices), (len(devices),), ("tp",))
    device_refs = [DeviceRef.from_device(d) for d in devices]
    with default_device(mesh):
        kv_params, kv_inputs = kv_params_and_inputs(
            config, devices, prompt_lens
        )
        with F.lazy(), default_dtype(DType.float32):
            module = KimiK3Model(config, kv_params=kv_params)
            if len(devices) > 1:
                tensor_parallel_kimi_k3_model(module, mesh)
    spec = GraphSpec(
        module,
        type(module).__qualname__,
        modulev3_ragged_kv_symbolic_inputs(
            kv_params=kv_params, device_refs=device_refs
        ),
        weights=_cast_to_parameter_dtypes(module, state_dict),
    )
    session = InferenceSession(devices=[*devices])
    run = compile_and_init_graphs(session, {GraphRole.DECODER: spec})[
        GraphRole.DECODER
    ]
    outputs = run(
        Tensor(input_ids, dtype=DType.int64, device=devices[0]),
        Tensor(np.array([1], dtype=np.int64), dtype=DType.int64, device=CPU()),
        Tensor(input_row_offsets, dtype=DType.uint32, device=devices[0]),
        *kv_inputs,
    )
    return outputs.next_token_logits.to_numpy()


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache and gated_delta_conv1d_fwd are both staged as undeclared"
    " custom ops the interpreter can't bind"
)
def test_lazy_loaded_tp2_matches_tp1(tmp_path: Path) -> None:
    hidden, vocab = 32, 40
    config = tiny_model_config(hidden, vocab)
    checkpoint = tmp_path / "model.safetensors"
    _write_checkpoint(random_weights(config), checkpoint)

    total_seq_len, batch_size = 6, 2
    input_ids = np.arange(total_seq_len, dtype=np.int64) % vocab
    input_row_offsets = np.array(
        [i * (total_seq_len // batch_size) for i in range(batch_size + 1)],
        dtype=np.uint32,
    )
    prompt_lens = [total_seq_len // batch_size] * batch_size

    out1 = _serve_logits(
        config,
        checkpoint,
        [Accelerator(0)],
        input_ids,
        input_row_offsets,
        prompt_lens,
    )
    out2 = _serve_logits(
        config,
        checkpoint,
        [Accelerator(0), Accelerator(1)],
        input_ids,
        input_row_offsets,
        prompt_lens,
    )

    err = rel_error(out2, out1)
    assert err < 1e-2, f"lazy-loaded TP=1 vs TP=2 mismatch: {err}"
