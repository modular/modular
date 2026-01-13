# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAMBA_130M_HF_REPO_ID = "state-spaces/mamba-130m-hf"
MAMBA_130M_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    MAMBA_130M_HF_REPO_ID
)

logger = logging.getLogger("max.pipelines")


def test_mamba_token_by_token_comparison() -> None:
    """Compare MAX vs PyTorch token-by-token generation with detailed logging.

    This test generates tokens one at a time and compares:
    1. Token IDs at each step
    2. Top-5 logits at each step
    3. When and where divergence occurs
    """
    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    prompt = "The capital of France is"
    num_tokens = 200

    # =================================================================
    # PyTorch Reference Generation
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PYTORCH REFERENCE GENERATION")
    logger.info("=" * 70)

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
    logger.info(f"\nPrompt: {prompt!r}")
    logger.info(f"Prompt tokens: {input_ids[0].tolist()}")
    logger.info(
        f"Prompt token strings: {[tokenizer.decode([t]) for t in input_ids[0]]}"
    )

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

        logger.info(f"\nPyTorch Step {step}:")
        logger.info(
            f"  Sequence so far ({len(current_ids[0])} tokens): {tokenizer.decode(current_ids[0])!r}"
        )
        logger.info("  Top 5 predictions:")
        for i in range(5):
            token_id = top5_indices[i].item()
            token_str = tokenizer.decode([token_id])
            logit_val = top5_values[i].item()
            logger.info(
                f"    [{token_id:5d}] {token_str!r:20s} logit={logit_val:8.3f}"
            )
        logger.info(
            f"  ✓ Chosen: [{next_token_id:5d}] {tokenizer.decode([next_token_id])!r}"
        )

        # Append to sequence
        next_token = torch.tensor([[next_token_id]]).to(device)
        current_ids = torch.cat([current_ids, next_token], dim=1)

    logger.info(f"\nPyTorch final output: {tokenizer.decode(current_ids[0])!r}")
    logger.info(f"PyTorch token sequence: {current_ids[0].tolist()}")

    # =================================================================
    # logger.info Summary
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"PyTorch generated {num_tokens} tokens successfully")
    logger.info(f"Expected token sequence: {torch_tokens}")
    logger.info("Now run MAX pipeline and compare token-by-token...")
    logger.info("\nTo test MAX, you can run the pipeline manually and compare:")
    logger.info("  ./bazelw run //max/python/max/entrypoints:pipelines -- \\")
    logger.info(f"    generate --model {MAMBA_130M_HF_REPO_ID} \\")
    logger.info(
        f"    --prompt '{prompt}' --top-k=1 --max-new-tokens={num_tokens}"
    )
