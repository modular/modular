# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio

import click
from uvicorn import Server

from llama3.llama3 import Llama3Context

from max.serve.api_server import fastapi_app, fastapi_config
from max.serve.config import APIType, Settings

from max import pipelines
from max.pipelines import llama3
from max.pipelines.utils import config_to_flag

from .deps import token_pipeline
from .llm import TokenGenerator, TokenGeneratorPipeline


async def serve_token_generator(model: TokenGenerator):
    pipeline = TokenGeneratorPipeline[Llama3Context](model)
    pipelines = [pipeline]

    settings = Settings(api_types=[APIType.OPENAI])
    app = fastapi_app(settings, pipelines)
    app.dependency_overrides[token_pipeline] = lambda: pipeline

    config = fastapi_config(app=app)
    server = Server(config)
    await server.serve()


@click.command(cls=pipelines.ModelGroup)
def main():
    pass


@main.command(name="llama3")
@config_to_flag(llama3.InferenceConfig)
def serve_llama3(**config_kwargs):
    config = llama3.InferenceConfig(**config_kwargs)
    pipelines.validate_weight_path(config, verify=False)
    model = llama3.Llama3(config)
    print("Beginning text generation serving...")
    asyncio.run(serve_token_generator(model))


if __name__ == "__main__":
    main()
