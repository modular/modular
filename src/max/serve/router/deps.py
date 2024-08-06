# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from fastapi import Depends
from typing_extensions import Annotated

from max.serve.config import RunnerType, Settings, get_settings
from max.serve.engine.pytorch import BasicPyTorchPipeline, BasicPyTorchRunner
from max.serve.engine.runner import Runner
from max.serve.mocks.mock_pytorch_runner import NeuralNetwork


def get_runner(
    settings: Annotated[Settings, Depends(get_settings)], model_name: str
) -> Runner:
    # TODO: Lookup model_name
    match settings.runner_type:
        case RunnerType.PYTORCH:
            return BasicPyTorchRunner(NeuralNetwork())


def get_pipeline(runner: Annotated[Runner, Depends(get_runner)]):
    return BasicPyTorchPipeline(runner, {})
