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

import logging
from io import StringIO

import hf_repo_lock
import pytest
from max.entrypoints import pipelines

MAMBA_130M_HF_REPO_ID = "state-spaces/mamba-130m-hf"
MAMBA_130M_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    MAMBA_130M_HF_REPO_ID
)

logger = logging.getLogger("max.pipelines")


def test_mamba_130m_hf_generation(capsys: pytest.CaptureFixture[str]) -> None:
    """Test running state-spaces/mamba-130m-hf using the mamba pipeline."""
    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                MAMBA_130M_HF_REPO_ID,
                "--prompt",
                "The capital of France is",
                "--trust-remote-code",
                "--devices=cpu",
                "--huggingface-model-revision",
                MAMBA_130M_HF_REVISION,
                "--max-new-tokens=20",
                "--top-k=1",
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0, "Expected output from model generation"


def test_mamba_130m_hf_generation_gpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test running state-spaces/mamba-130m-hf using the mamba pipeline on GPU."""
    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                MAMBA_130M_HF_REPO_ID,
                "--prompt",
                "The capital of France is",
                "--trust-remote-code",
                "--devices=gpu:0",
                "--huggingface-model-revision",
                MAMBA_130M_HF_REVISION,
                "--max-new-tokens=20",
                "--top-k=1",
                "--max-batch-size=1",
                "--max-length=512",
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0, "Expected output from model generation"


@pytest.mark.skip(
    reason="Manual test - compares MAX vs PyTorch output. Run with: pytest -k test_mamba_130m_compare_with_torch_cpu --override-ini='-m='"
)
def test_mamba_130m_compare_with_torch_cpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compare MAX pipeline output with PyTorch/transformers on CPU.

    This test is disabled by default but can be run manually to verify that
    the MAX Mamba implementation produces the same output as the reference
    PyTorch implementation.

    To run this test:
        ./bazelw test //max/tests/integration/pipelines/python/mamba:test_mamba_130m \
            --test_arg=-k --test_arg=test_mamba_130m_compare_with_torch_cpu \
            --test_arg=--override-ini --test_arg="-m="
    """
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    prompt = "Why is the sky blue?"
    max_new_tokens = 50
    top_k = 1  # Greedy decoding for deterministic comparison

    # Run MAX pipeline
    print("\n=== Running MAX Pipeline ===")
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        with pytest.raises(SystemExit):
            pipelines.main(
                [
                    "generate",
                    "--model-path",
                    MAMBA_130M_HF_REPO_ID,
                    "--prompt",
                    prompt,
                    "--trust-remote-code",
                    "--devices=cpu",
                    "--huggingface-model-revision",
                    MAMBA_130M_HF_REVISION,
                    f"--max-new-tokens={max_new_tokens}",
                    f"--top-k={top_k}",
                ]
            )
    finally:
        max_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

    # Strip metrics from MAX output (lines starting with "Prompt size:", etc.)
    max_lines = max_output.split("\n")
    max_text_lines = []
    for line in max_lines:
        # Stop when we hit the metrics section
        if (
            line.startswith("Prompt size:")
            or line.startswith("Output size:")
            or line.startswith("Startup time:")
        ):
            break
        max_text_lines.append(line)
    max_output = "\n".join(max_text_lines)

    print(f"MAX output: {max_output}")

    # Run PyTorch reference
    print("\n=== Running PyTorch Reference ===")
    torch.manual_seed(0)  # For reproducibility
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        MAMBA_130M_HF_REPO_ID,
        revision=MAMBA_130M_HF_REVISION,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MAMBA_130M_HF_REPO_ID,
        revision=MAMBA_130M_HF_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )

    torch_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"PyTorch output: {torch_output}")

    # Compare outputs
    print("\n=== Comparison ===")
    # MAX outputs only generated tokens, PyTorch includes prompt + generated
    # Extract only the generated portion from PyTorch by removing the prompt
    max_output_clean = max_output.strip()
    # PyTorch output starts with the prompt, remove it
    if torch_output.startswith(prompt):
        torch_generated = torch_output[len(prompt) :].strip()
    else:
        torch_generated = torch_output.strip()

    print(f"MAX generated: {max_output_clean[:100]}...")
    print(f"PyTorch generated: {torch_generated[:100]}...")
    print(f"MAX length: {len(max_output_clean)}")
    print(f"PyTorch generated length: {len(torch_generated)}")

    # Check if outputs match exactly
    if max_output_clean == torch_generated:
        print("✅ Generated outputs match exactly!")
    else:
        print("❌ Generated outputs differ!")
        print(f"\nMAX:\n{max_output_clean}")
        print(f"\nPyTorch:\n{torch_generated}")

        # Find first difference
        for i, (c1, c2) in enumerate(
            zip(max_output_clean, torch_generated, strict=False)
        ):
            if c1 != c2:
                print(f"\nFirst difference at position {i}:")
                print(f"  MAX: {max_output_clean[max(0, i - 20) : i + 20]!r}")
                print(
                    f"  PyTorch: {torch_generated[max(0, i - 20) : i + 20]!r}"
                )
                break

        # This is a manual test, so we don't assert - just report the difference
        pytest.fail(
            f"Generated outputs differ!\nMAX: {max_output_clean[:200]}...\nPyTorch: {torch_generated[:200]}..."
        )


@pytest.mark.skip(
    reason="Manual test - compares MAX vs PyTorch output on GPU. Run with: pytest -k test_mamba_130m_compare_with_torch_gpu"
)
def test_mamba_130m_compare_with_torch_gpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compare MAX pipeline output with PyTorch/transformers on GPU.

    This test is disabled by default but can be run manually to verify that
    the MAX Mamba implementation produces the same output as the reference
    PyTorch implementation on GPU.

    To run this test:
        ./bazelw test //max/tests/integration/pipelines/python/mamba:test_mamba_130m \
            --test_arg=-k --test_arg=test_mamba_130m_compare_with_torch_gpu \
            --test_arg=--override-ini --test_arg="-m="
    """
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    prompt = "Why is the sky blue?"
    max_new_tokens = 50
    top_k = 1  # Greedy decoding for deterministic comparison

    # Run MAX pipeline
    print("\n=== Running MAX Pipeline (GPU) ===")
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
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
                    f"--max-new-tokens={max_new_tokens}",
                    f"--top-k={top_k}",
                    "--max-batch-size=1",
                    "--max-length=512",
                ]
            )
    finally:
        max_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

    # Strip metrics from MAX output (lines starting with "Prompt size:", etc.)
    max_lines = max_output.split("\n")
    max_text_lines = []
    for line in max_lines:
        # Stop when we hit the metrics section
        if (
            line.startswith("Prompt size:")
            or line.startswith("Output size:")
            or line.startswith("Startup time:")
        ):
            break
        max_text_lines.append(line)
    max_output = "\n".join(max_text_lines)

    print(f"MAX output: {max_output}")

    # Run PyTorch reference
    print("\n=== Running PyTorch Reference (GPU) ===")
    torch.manual_seed(0)  # For reproducibility
    torch.cuda.manual_seed(0)
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(
        MAMBA_130M_HF_REPO_ID,
        revision=MAMBA_130M_HF_REVISION,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MAMBA_130M_HF_REPO_ID,
        revision=MAMBA_130M_HF_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )

    torch_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"PyTorch output: {torch_output}")

    # Compare outputs
    print("\n=== Comparison ===")
    # MAX outputs only generated tokens, PyTorch includes prompt + generated
    # Extract only the generated portion from PyTorch by removing the prompt
    max_output_clean = max_output.strip()
    # PyTorch output starts with the prompt, remove it
    if torch_output.startswith(prompt):
        torch_generated = torch_output[len(prompt) :].strip()
    else:
        torch_generated = torch_output.strip()

    print(f"MAX generated: {max_output_clean[:100]}...")
    print(f"PyTorch generated: {torch_generated[:100]}...")
    print(f"MAX length: {len(max_output_clean)}")
    print(f"PyTorch generated length: {len(torch_generated)}")

    # Check if outputs match exactly
    if max_output_clean == torch_generated:
        print("✅ Generated outputs match exactly!")
    else:
        print("❌ Generated outputs differ!")
        print(f"\nMAX:\n{max_output_clean}")
        print(f"\nPyTorch:\n{torch_generated}")

        # Find first difference
        for i, (c1, c2) in enumerate(
            zip(max_output_clean, torch_generated, strict=False)
        ):
            if c1 != c2:
                print(f"\nFirst difference at position {i}:")
                print(f"  MAX: {max_output_clean[max(0, i - 20) : i + 20]!r}")
                print(
                    f"  PyTorch: {torch_generated[max(0, i - 20) : i + 20]!r}"
                )
                break

        # This is a manual test, so we don't assert - just report the difference
        pytest.fail(
            f"Generated outputs differ!\nMAX: {max_output_clean[:200]}...\nPyTorch: {torch_generated[:200]}..."
        )


@pytest.mark.skip(
    reason="Manual test - compares MAX vs PyTorch output on GPU with 200 tokens. Run with: pytest -k test_mamba_130m_compare_with_torch_gpu_200_tokens"
)
def test_mamba_130m_compare_with_torch_gpu_200_tokens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compare MAX pipeline output with PyTorch/transformers on GPU for 200 tokens.

    This test is disabled by default but can be run manually to verify that
    the MAX Mamba implementation produces the same output as the reference
    PyTorch implementation on GPU for longer sequences (200 tokens).

    To run this test:
        ./bazelw test //max/tests/integration/pipelines/python/mamba:test_mamba_130m \
            --test_arg=-k --test_arg=test_mamba_130m_compare_with_torch_gpu_200_tokens \
            --test_arg=--override-ini --test_arg=-m=
    """
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    prompt = "The capital of France is"
    max_new_tokens = 200
    top_k = 1  # Greedy decoding for deterministic comparison

    # Run MAX pipeline
    print("\n=== Running MAX Pipeline (GPU) - 200 tokens ===")
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
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
                    f"--max-new-tokens={max_new_tokens}",
                    f"--top-k={top_k}",
                    "--max-batch-size=1",
                    "--max-length=512",
                ]
            )
    finally:
        max_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

    # Strip metrics from MAX output (lines starting with "Prompt size:", etc.)
    max_lines = max_output.split("\n")
    max_text_lines = []
    for line in max_lines:
        # Stop when we hit the metrics section
        if (
            line.startswith("Prompt size:")
            or line.startswith("Output size:")
            or line.startswith("Startup time:")
        ):
            break
        max_text_lines.append(line)
    max_output = "\n".join(max_text_lines)

    print(f"MAX output (first 200 chars): {max_output[:200]}")

    # Run PyTorch reference
    print("\n=== Running PyTorch Reference (GPU) - 200 tokens ===")
    torch.manual_seed(0)  # For reproducibility
    torch.cuda.manual_seed(0)
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(
        MAMBA_130M_HF_REPO_ID,
        revision=MAMBA_130M_HF_REVISION,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MAMBA_130M_HF_REPO_ID,
        revision=MAMBA_130M_HF_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )

    torch_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"PyTorch output (first 200 chars): {torch_output[:200]}")

    # Compare outputs
    print("\n=== Comparison ===")
    # MAX outputs only generated tokens, PyTorch includes prompt + generated
    # Extract only the generated portion from PyTorch by removing the prompt
    max_output_clean = max_output.strip()
    # PyTorch output starts with the prompt, remove it
    if torch_output.startswith(prompt):
        torch_generated = torch_output[len(prompt) :].strip()
    else:
        torch_generated = torch_output.strip()

    print(f"MAX generated (first 100 chars): {max_output_clean[:100]}...")
    print(f"PyTorch generated (first 100 chars): {torch_generated[:100]}...")
    print(f"MAX length: {len(max_output_clean)}")
    print(f"PyTorch generated length: {len(torch_generated)}")

    # Check if outputs match exactly
    if max_output_clean == torch_generated:
        print("✅ Generated outputs match exactly for 200 tokens!")
    else:
        print("❌ Generated outputs differ!")
        print(f"\nMAX:\n{max_output_clean}")
        print(f"\nPyTorch:\n{torch_generated}")

        # Find first difference
        for i, (c1, c2) in enumerate(
            zip(max_output_clean, torch_generated, strict=False)
        ):
            if c1 != c2:
                print(f"\nFirst difference at position {i}:")
                print(f"  MAX: {max_output_clean[max(0, i - 20) : i + 20]!r}")
                print(
                    f"  PyTorch: {torch_generated[max(0, i - 20) : i + 20]!r}"
                )
                break

        # This is a manual test, so we don't assert - just report the difference
        pytest.fail(
            f"Generated outputs differ!\nMAX: {max_output_clean[:300]}...\nPyTorch: {torch_generated[:300]}..."
        )
