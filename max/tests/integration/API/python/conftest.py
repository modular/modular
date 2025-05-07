# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
from os import getenv
from pathlib import Path

import hf_repo_lock
import max.driver as md
import pytest
from max.engine import InferenceSession
from max.pipelines.lib import generate_local_model_path

MODULAR_AI_LLAMA_3_1_HF_REPO_ID = "modularai/llama-3.1"
MODULAR_AI_LLAMA_3_1_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    MODULAR_AI_LLAMA_3_1_HF_REPO_ID
)

LLAMA_3_1_HF_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_3_1_HF_REVISION = hf_repo_lock.revision_for_hf_repo(LLAMA_3_1_HF_REPO_ID)

SMOLLM2_HF_REPO_ID = "HuggingFaceTB/SmolLM2-135M"
SMOLLM2_HF_REVISION = hf_repo_lock.revision_for_hf_repo(SMOLLM2_HF_REPO_ID)

EXAONE_2_4B_HF_REPO_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
EXAONE_2_4B_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    EXAONE_2_4B_HF_REPO_ID
)

DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID = (
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID
)

TINY_RANDOM_LLAMA_HF_REPO_ID = (
    "trl-internal-testing/tiny-random-LlamaForCausalLM"
)
TINY_RANDOM_LLAMA_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    TINY_RANDOM_LLAMA_HF_REPO_ID
)


GEMMA_3_1B_IT_HF_REPO_ID = "google/gemma-3-1b-it"
GEMMA_3_1B_IT_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    GEMMA_3_1B_IT_HF_REPO_ID
)


logger = logging.getLogger("max.pipelines")


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = getenv("MODULAR_PATH")
    assert modular_path is not None

    return Path(modular_path)


@pytest.fixture
def mo_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated BasicMLP model."""
    return (
        modular_path / "SDK" / "integration-test" / "API" / "c" / "mo-model.api"
    )


@pytest.fixture
def dynamic_model_path(modular_path: Path) -> Path:
    """Returns the path to the dynamic shape model."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "dynamic-model.mlir"
    )


@pytest.fixture
def no_input_path(modular_path: Path) -> Path:
    """Returns the path to a model spec without inputs."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "no-inputs.mlir"
    )


@pytest.fixture
def scalar_input_path(modular_path: Path) -> Path:
    """Returns the path to a model spec with scalar inputs."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "scalar-input.mlir"
    )


@pytest.fixture
def aliasing_outputs_path(modular_path: Path) -> Path:
    """Returns the path to a model spec with outputs that alias each other."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "aliasing-outputs.mlir"
    )


@pytest.fixture
def named_inputs_path(modular_path: Path) -> Path:
    """Returns the path to a model spec that adds a series of named tensors."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "named-inputs.mlir"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--custom-ops-path",
        type=str,
        default="",
        help="Path to custom Ops package",
    )


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    devices: list[md.Device] = []
    for i in range(md.accelerator_count()):
        devices.append(md.Accelerator(i))

    devices.append(md.CPU())

    return InferenceSession(devices=devices)


def pytest_collection_modifyitems(items):
    # Prevent pytest from trying to collect Click commands and dataclasses as tests
    for item in items:
        if item.name.startswith("Test"):
            item.add_marker(pytest.mark.skip)


@pytest.fixture
def graph_testdata() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = getenv("GRAPH_TESTDATA")
    assert path is not None
    return Path(path)


@pytest.fixture
def llama_3_1_8b_instruct_local_path():
    try:
        model_path = generate_local_model_path(
            LLAMA_3_1_HF_REPO_ID, LLAMA_3_1_HF_REVISION
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {LLAMA_3_1_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = LLAMA_3_1_HF_REPO_ID
    return model_path


@pytest.fixture
def smollm2_135m_local_path():
    try:
        model_path = generate_local_model_path(
            SMOLLM2_HF_REPO_ID, SMOLLM2_HF_REVISION
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {SMOLLM2_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = SMOLLM2_HF_REPO_ID
    return model_path


@pytest.fixture
def exaone_2_4b_local_path():
    try:
        model_path = generate_local_model_path(
            EXAONE_2_4B_HF_REPO_ID, EXAONE_2_4B_HF_REVISION
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {EXAONE_2_4B_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = EXAONE_2_4B_HF_REPO_ID
    return model_path


@pytest.fixture
def deepseek_r1_distill_llama_8b_local_path():
    try:
        model_path = generate_local_model_path(
            DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID,
            DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REVISION,
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID
    return model_path


@pytest.fixture
def modular_ai_llama_3_1_local_path():
    try:
        model_path = generate_local_model_path(
            MODULAR_AI_LLAMA_3_1_HF_REPO_ID, MODULAR_AI_LLAMA_3_1_HF_REVISION
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {MODULAR_AI_LLAMA_3_1_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = MODULAR_AI_LLAMA_3_1_HF_REPO_ID
    return model_path


@pytest.fixture
def tiny_random_llama_local_path():
    try:
        model_path = generate_local_model_path(
            TINY_RANDOM_LLAMA_HF_REPO_ID, TINY_RANDOM_LLAMA_HF_REVISION
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {TINY_RANDOM_LLAMA_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = TINY_RANDOM_LLAMA_HF_REPO_ID
    return model_path


@pytest.fixture
def gemma_3_1b_it_local_path():
    try:
        model_path = generate_local_model_path(
            GEMMA_3_1B_IT_HF_REPO_ID, GEMMA_3_1B_IT_HF_REVISION
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {GEMMA_3_1B_IT_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = GEMMA_3_1B_IT_HF_REPO_ID
    return model_path
