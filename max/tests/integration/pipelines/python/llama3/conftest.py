# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
import os
from pathlib import Path

import hf_repo_lock
import pytest
from hypothesis import settings
from max.engine import InferenceSession
from max.pipelines.lib import generate_local_model_path

LLAMA_3_1_HF_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_3_1_HF_REVISION = hf_repo_lock.revision_for_hf_repo(LLAMA_3_1_HF_REPO_ID)

LLAMA_3_1_LORA_HF_REPO_ID = "FinGPT/fingpt-mt_llama3-8b_lora"
LLAMA_3_1_LORA_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    LLAMA_3_1_LORA_HF_REPO_ID
)

pytest_plugins = "test_common.registry"

# When running in CI, graph tests can take around 300ms for a single run.
# These seem to be due to CI running under very high cpu usage.
# A similar effect can be achieved locally be running with each test multiple times `--runs_per_test=3`.
# They all launch at the same time leading to exceptionally heavy cpu usage.
# We have reasonable test suite timeouts. Use those instead of hypothesis deadlines.
settings.register_profile("graph_tests", deadline=None)
settings.load_profile("graph_tests")

logger = logging.getLogger("max.pipelines")


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    return Path(os.environ["MODULAR_PATH"])


@pytest.fixture(scope="session")
def testdata_directory() -> Path:
    """Returns the path to the Modular .derived directory."""
    return Path(os.environ["PIPELINES_TESTDATA"])


@pytest.fixture(scope="session")
def session() -> InferenceSession:
    return InferenceSession()


@pytest.fixture(scope="session")
def llama_3_1_8b_instruct_local_path():
    assert isinstance(LLAMA_3_1_HF_REVISION, str), (
        "LLAMA_3_1_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )
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
def llama_3_1_8b_lora_local_path():
    assert isinstance(LLAMA_3_1_LORA_HF_REVISION, str), (
        "LLAMA_3_1_LORA_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )
    try:
        model_path = generate_local_model_path(
            LLAMA_3_1_LORA_HF_REPO_ID, LLAMA_3_1_LORA_HF_REVISION
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {LLAMA_3_1_LORA_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = LLAMA_3_1_LORA_HF_REPO_ID
    return model_path
