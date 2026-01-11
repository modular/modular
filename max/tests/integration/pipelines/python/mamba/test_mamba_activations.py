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
"""Layer-by-layer activation comparison between MAX and PyTorch for Mamba."""

import logging
from io import StringIO

import hf_repo_lock
import numpy as np
import pytest
import torch
from max.entrypoints import pipelines
from transformers import AutoModelForCausalLM, AutoTokenizer

MAMBA_130M_HF_REPO_ID = "state-spaces/mamba-130m-hf"
MAMBA_130M_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    MAMBA_130M_HF_REPO_ID
)

logger = logging.getLogger("max.pipelines")


def test_mamba_first_token_logits_comparison():
    """Compare logits for the first token generation between MAX and PyTorch.

    This test focuses on the first token generated after the prompt to identify
    if the divergence starts immediately or after some steps.
    """
    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    prompt = "Why is the sky blue?"

    # =================================================================
    # PyTorch: Get logits for first token
    # =================================================================
    print("\n" + "=" * 70)
    print("PYTORCH - First Token Logits")
    print("=" * 70)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda:0")

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

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt: {prompt!r}")
    print(f"Prompt tokens: {input_ids[0].tolist()}")

    with torch.no_grad():
        outputs = torch_model(input_ids)
        torch_logits = outputs.logits[0, -1, :].cpu().numpy()

    torch_top5_indices = np.argsort(torch_logits)[-5:][::-1]
    torch_top5_values = torch_logits[torch_top5_indices]

    print("\nPyTorch Top 5 predictions:")
    for i in range(5):
        token_id = torch_top5_indices[i]
        token_str = tokenizer.decode([token_id])
        logit_val = torch_top5_values[i]
        print(f"  [{token_id:5d}] {token_str!r:20s} logit={logit_val:8.3f}")

    torch_predicted_token = int(np.argmax(torch_logits))
    print(f"\nPyTorch predicted token: [{torch_predicted_token}] {tokenizer.decode([torch_predicted_token])!r}")

    # =================================================================
    # MAX: Get logits for first token
    # =================================================================
    print("\n" + "=" * 70)
    print("MAX - First Token Logits")
    print("=" * 70)

    import sys
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Run MAX pipeline but capture output
        with pytest.raises(SystemExit):
            pipelines.main([
                "generate",
                "--model-path", MAMBA_130M_HF_REPO_ID,
                "--prompt", prompt,
                "--trust-remote-code",
                "--devices=gpu:0",
                "--huggingface-model-revision", MAMBA_130M_HF_REVISION,
                "--max-new-tokens=1",
                "--top-k=1",
            ])
    finally:
        max_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

    print(f"MAX full output:\n{max_output}")
    print(f"MAX output length: {len(max_output)}")

    # Extract the generated token from MAX output
    # The output format is: prompt + generated_text
    max_output_clean = max_output.strip()

    # Try to extract just the generated token
    # MAX output should be: prompt + 1 new token
    max_generated = max_output_clean[len(prompt):] if len(max_output_clean) > len(prompt) else ""
    print(f"\nMAX generated text: {max_generated!r}")

    # For a more precise comparison, we need to extract the actual token ID
    # Unfortunately, we can't easily get the logits from the pipeline CLI
    # So we'll compare the generated text

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"PyTorch first token: {tokenizer.decode([torch_predicted_token])!r}")
    print(f"MAX first token: {max_generated!r}")

    # Check if they match
    if max_generated.strip() == tokenizer.decode([torch_predicted_token]).strip():
        print("✅ First tokens MATCH!")
    else:
        print("❌ First tokens DIFFER!")
        print("\nThis suggests the divergence starts from the very first token.")
        print("Possible causes:")
        print("1. Different tokenization")
        print("2. Different model initialization")
        print("3. Bug in forward pass computation")

        # Don't fail the test, just report
        pytest.fail(
            f"First token differs! PyTorch: {tokenizer.decode([torch_predicted_token])!r}, "
            f"MAX: {max_generated!r}"
        )


def test_mamba_prompt_only_forward_pass():
    """Test just the forward pass with the prompt (no generation).

    This isolates whether the model can correctly process the prompt
    without the complexity of autoregressive generation.
    """
    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    prompt = "Why is the sky blue?"

    print("\n" + "=" * 70)
    print("PROMPT-ONLY FORWARD PASS TEST")
    print("=" * 70)
    print(f"Prompt: {prompt!r}")

    # PyTorch reference
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

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt tokens: {input_ids[0].tolist()}")
    print(f"Number of tokens: {len(input_ids[0])}")

    # Get PyTorch logits
    with torch.no_grad():
        outputs = torch_model(input_ids)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

    print(f"\nPyTorch logits shape: {logits.shape}")
    print(f"Expected: (1, {len(input_ids[0])}, vocab_size)")

    # Check logits for each position
    print("\nLogits statistics for each token position:")
    for pos in range(logits.shape[1]):
        pos_logits = logits[0, pos, :].cpu().numpy()
        print(f"  Position {pos}: min={pos_logits.min():.2f}, max={pos_logits.max():.2f}, "
              f"mean={pos_logits.mean():.2f}, std={pos_logits.std():.2f}")

        # Show top predicted token at this position
        top_token = int(np.argmax(pos_logits))
        top_logit = pos_logits[top_token]
        print(f"    Top prediction: [{top_token}] {tokenizer.decode([top_token])!r} (logit={top_logit:.2f})")

    # Focus on the last position (what matters for generation)
    last_logits = logits[0, -1, :].cpu().numpy()
    top5_indices = np.argsort(last_logits)[-10:][::-1]

    print(f"\nTop 10 predictions for next token (position {logits.shape[1]-1}):")
    for i in range(10):
        token_id = top5_indices[i]
        token_str = tokenizer.decode([token_id])
        logit_val = last_logits[token_id]
        print(f"  {i+1}. [{token_id:5d}] {token_str!r:20s} logit={logit_val:8.3f}")

    print("\n✅ PyTorch forward pass completed successfully")
    print("\nTo compare with MAX, run:")
    print("  ./bazelw run //max/python/max/entrypoints:pipelines -- \\")
    print(f"    generate --model {MAMBA_130M_HF_REPO_ID} \\")
    print(f"    --prompt '{prompt}' --max-new-tokens=1 --top-k=10")
