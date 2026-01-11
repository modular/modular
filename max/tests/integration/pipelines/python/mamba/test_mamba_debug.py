# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Debug test for Mamba model to track token generation."""

import logging

import hf_repo_lock
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAMBA_130M_HF_REPO_ID = "state-spaces/mamba-130m-hf"
MAMBA_130M_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    MAMBA_130M_HF_REPO_ID
)

logger = logging.getLogger("max.pipelines")


def test_mamba_token_by_token_comparison():
    """Compare MAX vs PyTorch token-by-token generation with detailed logging.

    This test generates tokens one at a time and compares:
    1. Token IDs at each step
    2. Top-5 logits at each step
    3. When and where divergence occurs
    """
    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    prompt = "Why is the sky blue?"
    num_tokens = 50

    # =================================================================
    # PyTorch Reference Generation
    # =================================================================
    print("\n" + "=" * 70)
    print("PYTORCH REFERENCE GENERATION")
    print("=" * 70)

    torch.manual_seed(0)
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        MAMBA_130M_HF_REPO_ID,
        revision=MAMBA_130M_HF_REVISION,
        trust_remote_code=True,
    )
    torch_model = AutoModelForCausalLM.from_pretrained(
        MAMBA_130M_HF_REPO_ID,
        revision=MAMBA_130M_HF_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: {prompt!r}")
    print(f"Prompt tokens: {input_ids[0].tolist()}")
    print(f"Prompt token strings: {[tokenizer.decode([t]) for t in input_ids[0]]}")

    # Generate tokens step by step
    torch_tokens = []
    torch_logits_list = []
    current_ids = input_ids

    for step in range(num_tokens):
        with torch.no_grad():
            outputs = torch_model(current_ids)
            logits = outputs.logits

        # Get logits for the last token
        last_logits = logits[0, -1, :].cpu().numpy()
        torch_logits_list.append(last_logits)

        # Get next token (greedy)
        next_token_id = int(torch.argmax(logits[0, -1, :]).item())
        torch_tokens.append(next_token_id)

        # Get top 5 for comparison
        top5_values, top5_indices = torch.topk(logits[0, -1, :], 5)

        print(f"\nPyTorch Step {step}:")
        print(f"  Sequence so far ({len(current_ids[0])} tokens): {tokenizer.decode(current_ids[0])!r}")
        print(f"  Top 5 predictions:")
        for i in range(5):
            token_id = top5_indices[i].item()
            token_str = tokenizer.decode([token_id])
            logit_val = top5_values[i].item()
            print(f"    [{token_id:5d}] {token_str!r:20s} logit={logit_val:8.3f}")
        print(f"  ✓ Chosen: [{next_token_id:5d}] {tokenizer.decode([next_token_id])!r}")

        # Append to sequence
        next_token = torch.tensor([[next_token_id]]).to(device)
        current_ids = torch.cat([current_ids, next_token], dim=1)

    print(f"\nPyTorch final output: {tokenizer.decode(current_ids[0])!r}")
    print(f"PyTorch token sequence: {current_ids[0].tolist()}")

    # =================================================================
    # Print Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"PyTorch generated {num_tokens} tokens successfully")
    print(f"Expected token sequence: {torch_tokens}")
    print(f"Now run MAX pipeline and compare token-by-token...")
    print("\nTo test MAX, you can run the pipeline manually and compare:")
    print(f"  ./bazelw run //max/python/max/entrypoints:pipelines -- \\")
    print(f"    generate --model {MAMBA_130M_HF_REPO_ID} \\")
    print(f"    --prompt '{prompt}' --top-k=1 --max-new-tokens={num_tokens}")
