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
"""First-token activation comparison between MAX and PyTorch for Mamba."""

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


def test_mamba_first_token_logits_comparison() -> None:
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
    print(
        f"\nPyTorch predicted token: [{torch_predicted_token}] {tokenizer.decode([torch_predicted_token])!r}"
    )

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
            pipelines.main(
                [
                    "generate",
                    "--model-path",
                    MAMBA_130M_HF_REPO_ID,
                    "--prompt",
                    prompt,
                    "--trust-remote-code",
                    "--devices=gpu:0",
                    "--huggingface-model-revision",
                    MAMBA_130M_HF_REVISION,
                    "--max-new-tokens=1",
                    "--top-k=1",
                ]
            )
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
    max_generated = (
        max_output_clean[len(prompt) :]
        if len(max_output_clean) > len(prompt)
        else ""
    )
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
    if (
        max_generated.strip()
        == tokenizer.decode([torch_predicted_token]).strip()
    ):
        print("✅ First tokens MATCH!")
    else:
        print("❌ First tokens DIFFER!")
        print(
            "\nThis suggests the divergence starts from the very first token."
        )
        print("Possible causes:")
        print("1. Different tokenization")
        print("2. Different model initialization")
        print("3. Bug in forward pass computation")

        # Don't fail the test, just report
        pytest.fail(
            f"First token differs! PyTorch: {tokenizer.decode([torch_predicted_token])!r}, "
            f"MAX: {max_generated!r}"
        )
