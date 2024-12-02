# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The fixtures for all tests in this directory."""

import os
from pathlib import Path

import pytest
from max.pipelines.config import PipelineConfig
from max.pipelines.registry import PIPELINE_REGISTRY
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig


@pytest.fixture(scope="session")
def testdata_directory() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("PIPELINES_TESTDATA")
    assert path is not None
    return Path(path)


@pytest.fixture(scope="session")
def pipeline_model_config(testdata_directory, request):
    """The tiny Llama 3 model that is being served.

    Note: Only one instance of a fixture is cached at a time.
    So we may get multiple invocations of this based on the parameters we are
    invoking it with.
    https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
    """
    config = PipelineConfig(
        weight_path=[Path(testdata_directory / request.param.weight_path)],
        version="3.1",
        max_length=request.param.max_length,
        max_new_tokens=request.param.max_new_tokens,
        quantization_encoding=request.param.encoding,
        device_spec=request.param.device_spec,
    )
    return config


# class DummyTokenizer(PreTrainedPipelineTokenizer[str]):
#     def __init__(self, delegate) -> None:
#         super().__init__(delegate)

#     async def new_context(self, request: TokenGeneratorRequest):
#         return ""


# def identity(x):
#     return x


@pytest.fixture(scope="session")
def app(pipeline_model_config):
    """The FastAPI app used to serve the model."""
    # repo_id = "modularai/llama-3.1"
    # tokenizer = DummyTokenizer(AutoTokenizer.from_pretrained(repo_id))
    # pipeline = TokenGeneratorPipeline(  # type: ignore
    #     TokenGeneratorPipelineConfig.continuous_heterogenous(
    #         tg_batch_size=tinyllama_model.config.max_cache_batch_size,
    #         ce_batch_size=1,
    #     ),
    #     "test",
    #     tokenizer,
    # )

    pipeline_config = PIPELINE_REGISTRY.validate_pipeline_config(
        pipeline_model_config
    )
    tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config,
    )

    assert pipeline_config.huggingface_repo_id is not None
    pipeline_model_name = pipeline_config.huggingface_repo_id

    pipeline_batch_config = (
        TokenGeneratorPipelineConfig.continuous_heterogenous(
            tg_batch_size=pipeline_config.max_cache_batch_size,
            ce_batch_size=1,
        )
    )

    serving_settings = ServingTokenGeneratorSettings(
        model_name=pipeline_model_name,
        model_factory=pipeline_factory,
        pipeline_config=pipeline_batch_config,
        tokenizer=tokenizer,
    )

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    app = fastapi_app(settings, debug_settings, serving_settings)
    return app


# @pytest.fixture(scope="session")
# def tinyllama_model(testdata_directory, request, session):
#     """The tiny Llama 3 model that is being served.

#     Note: Only one instance of a fixture is cached at a time.
#     So we may get multiple invocations of this based on the parameters we are
#     invoking it with.
#     https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
#     """
#     config = PipelineConfig(
#         weight_path=[Path(testdata_directory / request.param.weight_path)],
#         version="3.1",
#         max_length=request.param.max_length,
#         max_new_tokens=request.param.max_new_tokens,
#         quantization_encoding=request.param.encoding,
#         device_spec=request.param.device_spec,
#     )

#     model = Llama3Model(pipeline_config=config, session=session)
#     return model
#     return model
