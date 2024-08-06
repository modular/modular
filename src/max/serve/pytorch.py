# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""
Basic PyTorch runner implementation.
"""

import asyncio
from typing import Optional
import torch

from dataclasses import dataclass, field

from engine.runner import Runner
from scheduler.queues import BatchMultiplexQueue

from mocks.mock_pytorch_runner import NeuralNetwork


@dataclass
class PyTorchContext:
    model_path: str
    max_batch_size: int = 8


@dataclass
class BasicPyTorchRunner(Runner[PyTorchContext, torch.Tensor]):
    model: torch.nn.Module
    _inputs: list[torch.Tensor] = []
    _outputs: list[torch.Tensor] = []

    @property
    def inputs(self) -> list:
        return self._inputs

    @property
    def outputs(self) -> list:
        return self._outputs

    async def init(self, context: PyTorchContext) -> None:
        # TODO: Actually load model at model_path
        self.model = NeuralNetwork()

    async def run(self, context: PyTorchContext) -> None:
        output = self.model(self.inputs)

    async def post_run(self, context: PyTorchContext) -> None:
        inputs = []
        output = {}


@dataclass
class BasicPyTorchPipeline:
    """Simple PyTorch pipeline backed by a queue."""

    runner: BasicPyTorchRunner
    contexts: dict[str, PyTorchContext]
    queue: BatchMultiplexQueue = field(default_factory=BatchMultiplexQueue)

    def __post_init__(self) -> None:
        # TODO: Plumb config values into context.
        self.context = PyTorchContext("")
        self._task: asyncio.Task | None = None

    async def next(self, contexts: dict[str, PyTorchContext]) -> dict[str, str]:
        outputs = {}
        for req_id, ctx in contexts.items():
            await self.runner.pre_run(ctx)
            await self.runner.run(ctx)
            outputs[req_id] = self.runner.outputs
            await self.runner.post_run(ctx)
        return outputs

    def start(self, loop):
        self._task = loop.create_task(
            self.queue.dynamic_batching_worker(
                self.next,
                self.context.max_batch_size,
            )
        )

    def stop(self):
        if self._task:
            self._task.cancel()
        # TODO: also cancel any `queue.get()` tasks
