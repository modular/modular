# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import argparse
import uvicorn
import asyncio
import random
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeAlias, TypeVar

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from typing_extensions import Annotated

from max.serve.scheduler.queues import BatchMultiplexQueue

Context = TypeVar("Context")
EOS = "<eos>"


class TokenGenerator(Generic[Context], Protocol):
    """Interface for LLM token-generator models."""

    async def new_context(self, prompt: str) -> Context:
        ...

    async def next_token(self, contexts: dict[str, Context]) -> dict[str, str]:
        ...


@dataclass
class BatchedTokenModel:
    """The serving layer for an LLM model. Handles continuous batching."""

    model: TokenGenerator
    tokens_queue: BatchMultiplexQueue = field(
        default_factory=BatchMultiplexQueue
    )
    context_queue: BatchMultiplexQueue = field(
        default_factory=BatchMultiplexQueue
    )
    max_batch_size: int = 32
    _background_tasks: set = field(default_factory=set)

    async def generate(self, prompt: str):
        context = await self.model.new_context(prompt)

        # The first token is part of a context-encoding batch.
        # This goes away once we support ragged tensors.
        yield await self.context_queue.submit(context)

        async for token in self.tokens_queue.stream(context):
            if token == EOS:
                break
            yield token
            if (
                token > "10"
            ):  # introduce some randomness to get chunks for streaming
                await asyncio.sleep(0.5)

    def start(self, loop):
        # This can go away once we have ragged tensors
        context_encoder = loop.create_task(
            self.context_queue.dynamic_batching_worker(
                self.model.next_token,
                self.max_batch_size,
            )
        )
        token_generator = loop.create_task(
            self.tokens_queue.continuous_batching_worker(
                self.model.next_token,
                complete=(lambda token: token == EOS),
                max_batch_size=self.max_batch_size,
            )
        )
        self._background_tasks |= {context_encoder, token_generator}

    def stop(self):
        for task in self._background_tasks:
            task.cancel()
        # TODO: also cancel any `queue.get()` tasks


class RandomTokenGenerator:
    # TODO: More configuration, eg. temperature
    async def new_context(self, prompt: str) -> None:
        await asyncio.sleep(0.01)  # pretend we're calling max engine
        return None

    async def next_token(self, contexts: dict[str, None]) -> dict[str, str]:
        await asyncio.sleep(0.01)  # pretend we're calling max engine
        return {
            id: str(rand) if (rand := random.randint(0, 20)) < 20 else EOS
            for id in contexts
        }


class EchoTokenGenerator:
    # TODO: More configuration, eg. temperature
    def __init__(self):
        self.prompt = None
        self.curr_idx = 0

    async def new_context(self, prompt: str) -> None:
        self.prompt = prompt
        self.curr_idx = 0
        await asyncio.sleep(0.01)  # pretend we're calling max engine
        return None

    async def next_token(self, contexts: dict[str, None]) -> dict[str, str]:
        await asyncio.sleep(0.01)  # pretend we're calling max engine
        self.curr_idx = self.curr_idx + 1
        return {
            id: (
                self.prompt[self.curr_idx - 1] if self.curr_idx
                <= len(self.prompt) else EOS
            )
            for id in contexts
        }


# FastAPI dependency injection: shared model state dict
def _models():
    models = {}
    return lambda: models


models = _models()

# Use as `models: Annotated[Models, Depends(models)]` for dependency injection
Models: TypeAlias = dict[str, BatchedTokenModel]


# ------- Fast API interface starts here -------

app = FastAPI()


@app.post("/models/{model_id}/start")
async def start_model(
    model_id: str, models: Annotated[Models, Depends(models)]
):
    # In reality, take some configuration here :)
    if model_id == "random":
        model = BatchedTokenModel(
            RandomTokenGenerator()
        )  # Slow model setup here!
    elif model_id == "echo":
        model = BatchedTokenModel(
            EchoTokenGenerator()
        )  # Slow model setup here!
    else:
        raise HTTPException(
            status_code=404, detail=f"model {model_id} not found"
        )
    model.start(
        asyncio.get_running_loop()
    )  # This starts the serving coroutines
    models[model_id] = model


@app.get("/models/{model_id}/generate")
async def generate(
    model_id: str,
    prompt: str,
    models: Annotated[Models, Depends(models)],
):
    model = models[model_id]
    assert isinstance(model, BatchedTokenModel)
    return EventSourceResponse(model.generate(prompt))


@app.post("/models/{model_id}/stop")
async def stop_model(model_id: str, models: Annotated[Models, Depends(models)]):
    models[model_id].stop()
    del models[model_id]
    # The model will continue executing its current request, at which point
    # references to it are all garbage collected and it can be freed.


if __name__ == "__main__":
    # TODO(kreeger): Add optional args here.
    parser = argparse.ArgumentParser()
    # TODO(kreeger): Need to add more stuff here.
    uvicorn.run(app)
