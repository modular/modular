# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from pathlib import Path
from max.pipelines.config import (
    HuggingFaceRepo,
    WeightsFormat,
    SupportedEncoding,
)


def test_huggingface_repo__formats_available():
    # Test a GGUF repo
    hf_repo = HuggingFaceRepo(
        repo_id="modularai/llama-3.1",
    )

    assert WeightsFormat.gguf in hf_repo.formats_available
    assert WeightsFormat.safetensors not in hf_repo.formats_available

    # Test a Safetensors repo
    hf_repo = HuggingFaceRepo(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    assert WeightsFormat.safetensors in hf_repo.formats_available
    assert WeightsFormat.gguf not in hf_repo.formats_available


def test_huggingface_repo__gguf_architecture():
    # Test a llama based gguf repo.
    hf_repo = HuggingFaceRepo(repo_id="modularai/llama-3.1")

    assert hf_repo.gguf_architecture == "llama"

    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert hf_repo.gguf_architecture is None


def test_huggingface_repo__encodings_supported():
    # Test a llama based gguf repo.
    hf_repo = HuggingFaceRepo(repo_id="modularai/llama-3.1")
    assert SupportedEncoding.bfloat16 in hf_repo.supported_encodings

    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert SupportedEncoding.q4_k not in hf_repo.supported_encodings
    assert SupportedEncoding.bfloat16 in hf_repo.supported_encodings


def test_huggingface_repo__file_exists():
    # Test a llama based gguf repo.
    hf_repo = HuggingFaceRepo(repo_id="modularai/llama-3.1")
    assert hf_repo.file_exists("llama-3.1-8b-instruct-bf16.gguf")
    assert not hf_repo.file_exists(
        "this_definitely_should_not_exist.safetensors"
    )


def test_huggingface_repo__get_files_for_encoding():
    # Test a llama based gguf repo.
    hf_repo = HuggingFaceRepo(repo_id="modularai/llama-3.1")
    files = hf_repo.files_for_encoding(SupportedEncoding.bfloat16)
    assert WeightsFormat.gguf in files
    assert len(files[WeightsFormat.gguf]) == 1
    assert files[WeightsFormat.gguf][0] == Path(
        "llama-3.1-8b-instruct-bf16.gguf"
    )

    files = hf_repo.files_for_encoding(
        SupportedEncoding.bfloat16, weights_format=WeightsFormat.safetensors
    )
    assert len(files) == 0

    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    files = hf_repo.files_for_encoding(SupportedEncoding.bfloat16)
    assert WeightsFormat.safetensors in files
    assert len(files[WeightsFormat.safetensors]) == 1
    assert files[WeightsFormat.safetensors][0] == Path("model.safetensors")

    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id="Qwen/QwQ-32B-Preview")
    files = hf_repo.files_for_encoding(SupportedEncoding.bfloat16)
    assert WeightsFormat.safetensors in files
    assert len(files[WeightsFormat.safetensors]) == 17
    assert files[WeightsFormat.safetensors] == [
        Path("model-00001-of-00017.safetensors"),
        Path("model-00002-of-00017.safetensors"),
        Path("model-00003-of-00017.safetensors"),
        Path("model-00004-of-00017.safetensors"),
        Path("model-00005-of-00017.safetensors"),
        Path("model-00006-of-00017.safetensors"),
        Path("model-00007-of-00017.safetensors"),
        Path("model-00008-of-00017.safetensors"),
        Path("model-00009-of-00017.safetensors"),
        Path("model-00010-of-00017.safetensors"),
        Path("model-00011-of-00017.safetensors"),
        Path("model-00012-of-00017.safetensors"),
        Path("model-00013-of-00017.safetensors"),
        Path("model-00014-of-00017.safetensors"),
        Path("model-00015-of-00017.safetensors"),
        Path("model-00016-of-00017.safetensors"),
        Path("model-00017-of-00017.safetensors"),
    ]

    # Test a Safetensors repo, with the wrong encoding requested.
    hf_repo = HuggingFaceRepo(repo_id="Qwen/QwQ-32B-Preview")
    files = hf_repo.files_for_encoding(SupportedEncoding.float32)
    assert len(files) == 0


def test_huggingface_repo__encoding_for_file():
    # This repo, has one safetensors file, and its a bf16 file.
    hf_repo = HuggingFaceRepo(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model_encoding = hf_repo.encoding_for_file("model.safetensors")
    assert model_encoding == SupportedEncoding.bfloat16

    # This repo, has many safetensors file, and they are bf16.
    hf_repo = HuggingFaceRepo(repo_id="Qwen/QwQ-32B-Preview")
    model_encoding = hf_repo.encoding_for_file(
        "model-00014-of-00017.safetensors"
    )
    assert model_encoding == SupportedEncoding.bfloat16

    # This repo, has a few GGUF files, and they are a variety of encodings.
    hf_repo = HuggingFaceRepo(repo_id="modularai/llama-3.1")
    model_encoding = hf_repo.encoding_for_file("llama-3.1-8b-instruct-f32.gguf")
    assert model_encoding == SupportedEncoding.float32

    model_encoding = hf_repo.encoding_for_file(
        "llama-3.1-8b-instruct-q4_k_m.gguf"
    )
    assert model_encoding == SupportedEncoding.q4_k
