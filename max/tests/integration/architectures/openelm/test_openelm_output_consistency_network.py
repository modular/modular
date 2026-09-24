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
"""Logit consistency check: MAX pipeline vs. HuggingFace reference for OpenELM.

Compares logits at the final token position for the same prompt tokenized
once and fed to both. Downloads apple/OpenELM-270M-Instruct and the
LLaMA-2 tokenizer from the HuggingFace Hub; the tokenizer repo is gated,
so this requires an HF_TOKEN with access already granted. Kept in its own
"_network"-suffixed file so BUILD.bazel can glob it into a separate
"requires-network"-tagged target.

Drives the real KV-cache-backed graph (MultiKVCacheParams grouped by
per-layer KV head count, PagedKVCacheManager, real graph execution)
rather than a hand-rolled forward loop.
"""

import json
from types import SimpleNamespace

import numpy as np
import torch

PROMPT = "The capital of France is"


class TestOutputConsistency:
    """End-to-end logit comparison: HuggingFace reference vs. MAX pipeline."""

    MODEL_REPO_ID = "apple/OpenELM-270M-Instruct"
    TOKENIZER_REPO_ID = "meta-llama/Llama-2-7b-hf"

    def _model_path(self) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(self.MODEL_REPO_ID)

    def _get_hf_logits(self, token_ids: np.ndarray) -> np.ndarray:
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import (
            get_class_from_dynamic_module,
        )

        model_path = self._model_path()
        hf_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        openelm_cls = get_class_from_dynamic_module(
            "modeling_openelm.OpenELMForCausalLM", model_path
        )

        model = openelm_cls(hf_config)
        model.load_state_dict(load_file(f"{model_path}/model.safetensors"))
        model.eval()

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        return outputs.logits[0, -1, :].float().numpy()

    def _get_max_logits(self, token_ids: np.ndarray) -> np.ndarray:
        from max.driver import CPU, Buffer
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType
        from max.graph.weights import WeightData
        from max.nn.kv_cache import MultiKVCacheParams
        from max.nn.transformer import ReturnLogits
        from max.pipelines.architectures.openelm.model import (
            OpenELMLanguageModel,
        )
        from max.pipelines.architectures.openelm.model_config import (
            OpenELMConfig,
            compute_layer_configs,
        )
        from max.pipelines.architectures.openelm.weight_adapters import (
            convert_safetensor_state_dict,
        )
        from max.pipelines.context import TextContext, TokenBuffer
        from max.pipelines.kv_cache.config import KVCacheConfig
        from max.pipelines.kv_cache.paged_kv_cache.cache_manager import (
            PagedKVCacheManager,
        )
        from max.pipelines.request.base import RequestID
        from safetensors import safe_open

        model_path = self._model_path()
        with open(f"{model_path}/config.json") as f:
            hf_config = SimpleNamespace(**json.load(f))

        device = DeviceRef.CPU()
        kv_params = OpenELMConfig.construct_kv_params(
            huggingface_config=hf_config,
            pipeline_config=SimpleNamespace(
                model=SimpleNamespace(data_parallel_degree=1)
            ),
            devices=[device],
            kv_cache_config=KVCacheConfig(),
            cache_dtype=DType.float32,
        )
        config = OpenELMConfig(
            dtype=DType.float32,
            max_seq_len=512,
            model_dim=hf_config.model_dim,
            vocab_size=hf_config.vocab_size,
            head_dim=hf_config.head_dim,
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_base=float(getattr(hf_config, "rope_freq_constant", 10000.0)),
            normalize_qk_projections=getattr(
                hf_config, "normalize_qk_projections", False
            ),
            layer_configs=compute_layer_configs(hf_config),
            devices=[device],
            kv_params=kv_params,
        )

        nn_model = OpenELMLanguageModel(
            config, return_logits=ReturnLogits.LAST_TOKEN
        )

        class _Weights:
            def __init__(self, weight_data: WeightData) -> None:
                self._weight_data = weight_data

            def data(self) -> WeightData:
                return self._weight_data

        raw_state = {}
        with safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
            for key in f.keys():
                arr = f.get_tensor(key).float().numpy()
                raw_state[key] = _Weights(WeightData.from_numpy(arr, key))
        state_dict = convert_safetensor_state_dict(raw_state)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        weights_registry = nn_model.state_dict(auto_initialize=False)

        session = InferenceSession(devices=[CPU()])
        kv_manager = PagedKVCacheManager(
            params=kv_params,
            total_num_pages=64,
            session=session,
            max_batch_size=1,
        )

        device_ref = DeviceRef.CPU()
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        # Use `from max import tree; tree.leaves(...)` here on current main.
        flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

        with Graph(
            "OpenELMForCausalLM",
            input_types=[
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *flattened_kv_types,
            ],
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_args = graph.inputs
            assert isinstance(kv_params, MultiKVCacheParams)
            kv_collections = kv_params.unflatten_basic_kv_tree(iter(kv_args))
            kv_collections_single_device = [pd[0] for pd in kv_collections]
            outputs = nn_model(
                tokens=tokens.tensor,
                kv_collections=kv_collections_single_device,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets.tensor,
            )
            graph.output(*outputs)

        compiled = session.load(graph, weights_registry=weights_registry)

        ctx = TextContext(
            request_id=RequestID(),
            max_length=512,
            tokens=TokenBuffer(token_ids.astype(np.int64)),
        )
        kv_manager.claim(ctx)
        kv_manager.alloc(ctx)
        kv_buffers = kv_manager.runtime_inputs([[ctx]]).flatten()

        seq_len = len(token_ids)
        result = compiled.execute(
            Buffer.from_numpy(token_ids.astype(np.int64)).to(CPU()),
            Buffer.from_numpy(np.array([0, seq_len], dtype=np.uint32)).to(
                CPU()
            ),
            Buffer.from_numpy(np.array([1], dtype=np.int64)).to(CPU()),
            *kv_buffers,
        )
        logits = result[0].to_numpy().astype(np.float32)
        return logits[-1]

    def _tokenize(self) -> np.ndarray:
        from huggingface_hub import snapshot_download
        from transformers import LlamaTokenizer

        tokenizer_path = snapshot_download(self.TOKENIZER_REPO_ID)
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        return np.array(tokenizer.encode(PROMPT), dtype=np.int64)

    def test_top1_token_matches(self) -> None:
        """Greedy argmax token must be identical between HF and MAX."""
        token_ids = self._tokenize()
        hf_logits = self._get_hf_logits(token_ids)
        max_logits = self._get_max_logits(token_ids)

        hf_top1 = int(np.argmax(hf_logits))
        max_top1 = int(np.argmax(max_logits))

        assert hf_top1 == max_top1, (
            f"Top-1 token differs: HF predicts {hf_top1}, MAX predicts {max_top1}"
        )

    def test_top5_tokens_match(self) -> None:

        token_ids = self._tokenize()
        hf_logits = self._get_hf_logits(token_ids)
        max_logits = self._get_max_logits(token_ids)

        hf_top5 = set(np.argsort(hf_logits)[-5:])
        max_top5 = set(np.argsort(max_logits)[-5:])
        overlap = len(hf_top5 & max_top5)

        assert overlap >= 3, (
            f"Only {overlap}/5 top tokens overlap between HF and MAX.\n"
            f"HF top-5:  {sorted(hf_top5)}\n"
            f"MAX top-5: {sorted(max_top5)}"
        )

    def test_logit_magnitude_is_similar(self) -> None:

        token_ids = self._tokenize()
        hf_logits = self._get_hf_logits(token_ids)
        max_logits = self._get_max_logits(token_ids)

        hf_max = float(np.max(hf_logits))
        max_max = float(np.max(max_logits))
        diff = abs(hf_max - max_max)

        assert diff < 10.0, (
            f"Max logit diverges: HF={hf_max:.3f}, MAX={max_max:.3f}, diff={diff:.3f}"
        )
