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

"""Run a ComponentModel text encoder for verification with padded prompts.

Bypasses the regular text-generation pipeline path to enable
attention_mask-based verification of pre-norm hidden states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.graph.weights import load_weights
from max.pipelines.architectures.flux2.tokenizer import tokenize_klein_text
from transformers import AutoConfig, AutoTokenizer


def run_max_text_encoder(
    model_path: str,
    component_model_class: type,
    devices: list[Device],
    prompts: list[str],
    padded_length: int | None = 512,
    revision: str | None = None,
    print_outputs: bool = False,
) -> list[dict[str, Any]]:
    """Run a ComponentModel text encoder for verification.

    When ``padded_length`` is an integer each prompt is padded to that
    length via ``apply_chat_template`` + ``padding="max_length"``. When it
    is ``None`` each prompt is tokenized at its natural length with an
    all-ones attention mask. The returned ``embeddings`` field holds the
    final pre-norm hidden state with shape ``[seq_len, hidden_size]``.
    """
    model_dir = snapshot_download(
        repo_id=model_path,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"],
    )

    hf_config = AutoConfig.from_pretrained(model_dir)
    config_dict = hf_config.to_dict()
    num_hidden_layers = int(config_dict["num_hidden_layers"])
    config_dict["hidden_state_layers"] = [num_hidden_layers]

    session = InferenceSession(devices=devices)
    weight_paths = sorted(Path(model_dir).glob("*.safetensors"))
    weights = load_weights(weight_paths)
    encoder = component_model_class(
        config=config_dict,
        encoding="bfloat16",
        devices=devices,
        weights=weights,
        session=session,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    results: list[dict[str, Any]] = []
    for prompt in prompts:
        # Shared with production Flux2Tokenizer.encode() so the verification
        # path tokenizes identically to what MAX serves.
        encoded = tokenize_klein_text(
            tokenizer, prompt, max_length=padded_length
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.bool_)

        tokens_buffer = Buffer.from_numpy(input_ids).to(devices[0])
        output_buffer = encoder(tokens_buffer, attention_mask=attention_mask)

        # Buffer may be bfloat16; numpy doesn't support that via DLPack, so go via torch.
        h = torch.from_dlpack(output_buffer).float().cpu().numpy()
        if h.ndim == 3:
            h = h[0]  # [1, S, D] -> [S, D]

        if print_outputs:
            print(f"Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
            print(f"  hidden state shape={h.shape}")

        results.append({"prompt": prompt, "embeddings": h})

    return results
