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
    reason="Debugging test - shows detailed token processing. Run with: pytest -k test_mamba_130m_debug_token_processing --override-ini='-m='"
)
def test_mamba_130m_debug_token_processing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Debug Mamba token processing to identify repeated text patterns.

    This test enables detailed logging to track how tokens are processed
    through the generation pipeline and identify where repeated text occurs.

    To run this test:
        ./bazelw test //max/tests/integration/pipelines/python/mamba:test_mamba_130m \
            --test_arg=-k --test_arg=test_mamba_130m_debug_token_processing \
            --test_arg=--override-ini --test_arg="-m=" \
            --test_arg=--test_arg="-s"
    """
    import logging
    import sys

    # Enable debug logging for our modules
    logging.getLogger("max.pipelines").setLevel(logging.DEBUG)

    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    # Test cases with different prompt lengths to trigger issues
    test_cases = [
        ("A", 5, "Very short prompt"),
        ("The capital of France is", 10, "Medium prompt"),
        (
            "Why is the sky blue? The scientific explanation involves",
            15,
            "Longer prompt",
        ),
    ]

    for prompt, max_new_tokens, description in test_cases:
        print(f"\n=== DEBUG: {description} ===")
        print(f"Prompt: '{prompt}'")
        print(f"Max new tokens: {max_new_tokens}")

        # Capture the output with debugging enabled
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
                        "--devices=cpu",  # Use CPU for more predictable behavior
                        "--huggingface-model-revision",
                        MAMBA_130M_HF_REVISION,
                        f"--max-new-tokens={max_new_tokens}",
                        "--top-k=1",  # Greedy decoding for reproducibility
                        "--max-batch-size=1",
                    ]
                )
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

        # Extract just the generated text (remove metrics)
        lines = output.split("\n")
        text_lines = []
        for line in lines:
            if not any(
                line.startswith(prefix)
                for prefix in [
                    "Prompt size:",
                    "Output size:",
                    "Startup time:",
                    "INFO:",
                    "Time to first token:",
                    "Prompt eval throughput:",
                    "Time per Output Token:",
                    "Eval throughput:",
                    "Total Latency:",
                    "Total Throughput:",
                    "Building graph",
                    "Compiling model",
                ]
            ):
                text_lines.append(line)

        generated_text = "\n".join(text_lines).strip()

        print(f"Generated output: '{generated_text}'")

        # Check for repeated patterns
        if prompt.lower() in generated_text.lower():
            print("⚠️  WARNING: Prompt appears to be repeated in output!")

        # Count duplicate words/phrases
        words = generated_text.lower().split()
        if len(words) > 5:
            unique_words = set(words)
            if len(unique_words) < len(words) * 0.7:  # If < 70% unique
                print("⚠️  WARNING: High repetition detected!")
                print(
                    f"Total words: {len(words)}, Unique words: {len(unique_words)}"
                )


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


@pytest.mark.skip(
    reason="Manual debug test - focuses on longer token sequences that show repeated text. Run with: pytest -k test_mamba_130m_debug_long_sequences"
)
def test_mamba_130m_debug_long_sequences(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Debug test for longer sequences (50-200 tokens) where repeated text occurs.

    This test focuses on longer token sequences that tend to showcase repeated
    text issue. It enables debug logging and provides detailed analysis of
    generation process.

    To run this test:
        ./bazelw test //max/tests/integration/pipelines/python/mamba:test_mamba_130m \
            --test_arg=-k --test_arg=test_mamba_130m_debug_long_sequences \
            --test_arg=--override-ini --test_arg="-m=" \
            --test_arg=--test_arg=-s
    """
    import logging
    import sys

    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    # Enable debug logging for detailed analysis
    logging.getLogger("max.pipelines").setLevel(logging.DEBUG)

    test_cases = [
        # Short prompt (should work fine)
        {
            "prompt": "The capital of France is",
            "max_new_tokens": 20,
            "description": "Short prompt test",
        },
        # Medium prompt (may show issues)
        {
            "prompt": "In a world where technology has advanced beyond recognition, artificial intelligence has become an integral part of our daily lives. The year is 2045, and line between human and machine has blurred in ways we never thought possible. This story begins",
            "max_new_tokens": 50,
            "description": "Medium prompt test (~60 tokens)",
        },
        # Long prompt (more likely to show repeated text)
        {
            "prompt": "The history of artificial intelligence dates back to ancient times when humans first began to imagine machines that could think and reason. From the earliest myths of mechanical beings to modern era of deep learning and neural networks, the quest for artificial intelligence has been a long and fascinating journey. In the 1950s, Alan Turing proposed the famous Turing test, which became a benchmark for determining machine intelligence. The field has seen multiple winters and summers, with periods of optimism followed by disillusionment. Today, with the advent of large language models and advanced machine learning techniques, we are witnessing what many call the age of artificial general intelligence. This rapid progress raises important questions about the future of humanity, the nature of consciousness, and the ethical implications of creating machines that can potentially surpass human intelligence in virtually every domain. The story of AI continues to unfold with each passing day, bringing new discoveries and challenges that reshape our understanding of both technology and ourselves. As we stand at this crucial juncture in history, it is worth reflecting on how far we have come and considering the implications of",
            "max_new_tokens": 150,
            "description": "Long prompt test (~200+ tokens) - likely to show repeated text",
        },
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"DEBUG TEST CASE {i + 1}: {test_case['description']}")
        prompt = test_case.get("prompt", "")
        prompt_words = len(prompt.split()) if isinstance(prompt, str) else 0
        print(f"Prompt length: {prompt_words} words")
        print(f"Max new tokens: {test_case['max_new_tokens']}")
        print(f"{'=' * 60}")

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
                        test_case["prompt"],
                        "--trust-remote-code",
                        "--devices=gpu:0",
                        "--huggingface-model-revision",
                        MAMBA_130M_HF_REVISION,
                        f"--max-new-tokens={test_case['max_new_tokens']}",
                        "--top-k=1",
                        "--max-batch-size=1",
                        "--max-length=512",
                    ]
                )
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

        # Strip metrics from output for analysis
        lines = output.split("\n")
        text_lines = []
        for line in lines:
            if not any(
                line.startswith(prefix)
                for prefix in [
                    "Prompt size:",
                    "Output size:",
                    "Startup time:",
                    "INFO:",
                    "Time to first token:",
                    "Prompt eval throughput:",
                    "Time per Output Token:",
                    "Eval throughput:",
                    "Total Latency:",
                    "Total Throughput:",
                    "Building graph",
                    "Compiling model",
                ]
            ):
                text_lines.append(line)

        clean_output = "\n".join(text_lines).strip()

        print(f"Generated output ({len(clean_output)} chars):")
        print(
            f"'{clean_output[:200]}{'...' if len(clean_output) > 200 else ''}'"
        )

        # Check for repeated text patterns
        words = clean_output.lower().split()
        repeated_sequences = []

        # Check for 2-word repeats
        for i in range(len(words) - 3):
            seq1 = " ".join(words[i : i + 2])
            seq2 = " ".join(words[i + 2 : i + 4])
            if seq1 == seq2 and len(seq1) > 3:  # Not just "the the"
                repeated_sequences.append(
                    f"'{seq1}' (2-word repeat at position {i})"
                )

        # Check for 3-word repeats
        for i in range(len(words) - 5):
            seq1 = " ".join(words[i : i + 3])
            seq2 = " ".join(words[i + 3 : i + 6])
            if seq1 == seq2 and len(seq1) > 5:
                repeated_sequences.append(
                    f"'{seq1}' (3-word repeat at position {i})"
                )

        if repeated_sequences:
            print("\n🚨 REPEATED TEXT DETECTED:")
            for repeat in repeated_sequences[:5]:  # Show first 5
                print(f"  - {repeat}")
        else:
            print("\n✅ No obvious repeated text patterns detected")

        print(f"\nDebug analysis complete for case {i + 1}")


@pytest.mark.skip(
    reason="Manual performance test - compares generation speed with vs without proper SSM state caching"
)
def test_mamba_130m_performance_benchmark(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Performance benchmark to measure impact of SSM state caching.

    This test measures generation speed for different sequence lengths to quantify
    performance improvement from implementing proper SSM state caching.

    To run this test:
        ./bazelw test //max/tests/integration/pipelines/python/mamba:test_mamba_130m \
            --test_arg=-k --test_arg=test_mamba_130m_performance_benchmark \
            --test_arg=--override-ini --test_arg="-m="
    """
    import sys
    import time

    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    test_lengths = [10, 50, 100, 200]
    results = []

    for length in test_lengths:
        print(f"\nBenchmarking generation with {length} new tokens...")

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        start_time = time.time()

        try:
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
                        f"--max-new-tokens={length}",
                        "--top-k=1",
                        "--max-batch-size=1",
                        "--max-length=512",
                    ]
                )
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            end_time = time.time()

        # Extract timing info from output
        total_time = None
        for line in output.split("\n"):
            if "Total Latency:" in line:
                try:
                    total_time = float(line.split(":")[1].strip().split()[0])
                except:
                    pass

        elapsed = end_time - start_time
        results.append(
            {
                "length": length,
                "measured_time": elapsed,
                "reported_latency": total_time,
                "tokens_per_second": length / elapsed if elapsed > 0 else 0,
            }
        )

    print(f"\n{'=' * 60}")
    print("PERFORMANCE BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"{'Tokens':<10} {'Time (s)':<12} {'Tokens/s':<15}")
    print("-" * 40)

    for result in results:
        print(
            f"{result['length']:<10} {result['measured_time']:<12.2f} {result['tokens_per_second']:<15.1f}"
        )

    # Analyze complexity
    if len(results) >= 2:
        ratio1 = (
            results[1]["measured_time"] / results[0]["measured_time"]
            if results[1]["measured_time"] and results[0]["measured_time"]
            else 0
        )  # 50/10
        ratio2 = (
            results[2]["measured_time"] / results[1]["measured_time"]
            if results[2]["measured_time"] and results[1]["measured_time"]
            else 0
        )  # 100/50
        ratio3 = (
            results[3]["measured_time"] / results[2]["measured_time"]
            if results[3]["measured_time"] and results[2]["measured_time"]
            else 0
        )  # 200/100

        print("\nComplexity analysis:")
        print(f"10→50 tokens: {ratio1:.2f}x time increase (5x tokens)")
        print(f"50→100 tokens: {ratio2:.2f}x time increase (2x tokens)")
        print(f"100→200 tokens: {ratio3:.2f}x time increase (2x tokens)")

        if ratio1 > 10 or ratio2 > 4 or ratio3 > 4:
            print(
                "🚨 INDICATES O(n²) or worse complexity - SSM caching needed!"
            )
        else:
            print("✅ Appears to be close to O(n) complexity")
