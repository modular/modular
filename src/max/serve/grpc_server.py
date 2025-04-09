# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# mypy: disable-error-code="import-not-found"

import asyncio
import functools
import os

import click
import max.serve.grpc_serve.grpc_serve as max_grpc
from max.driver import DeviceSpec
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextGenerationPipeline,
    TextTokenizer,
)
from max.pipelines.architectures import register_all_models
from max.serve.pipelines.performance_fake import (
    PerformanceFakingPipelineTokenizer,
    PerformanceFakingTokenGenerator,
)
from transformers import AutoTokenizer


def get_default_replit_config(use_cpu: bool) -> PipelineConfig:
    pipeline_config = PipelineConfig(
        model_path="modularai/replit-code-1.5",
        trust_remote_code=True,
        device_specs=[
            DeviceSpec(id=0, device_type="cpu")
            if use_cpu
            else DeviceSpec(id=0, device_type="gpu")
        ],
        quantization_encoding=SupportedEncoding.float32,
        # save_to_serialized_model_path="/tmp/replit_gpu_16.mef",
        # serialized_model_path="/tmp/replit_gpu_16.mef",
        # pipeline_config.model_config.weight_path = hf_file.download()
    )
    return pipeline_config


def get_default_llama31_config(use_cpu: bool) -> PipelineConfig:
    return PipelineConfig(
        # model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_path="modularai/llama-3.1",
        device_specs=[
            DeviceSpec(id=0, device_type="cpu")
            if use_cpu
            else DeviceSpec(id=0, device_type="gpu")
        ],
        quantization_encoding=SupportedEncoding.bfloat16,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
        # save_to_serialized_model_path="/tmp/llama31_gpu_16.mef",
        # serialized_model_path="/tmp/llama31_gpu_16.mefs",
    )


@click.command()
@click.option(
    "--bypass-serve",
    is_flag=True,
    default=False,
    help="Bypass serving infra to directly run model.",
)
@click.option(
    "--model",
    type=click.Choice(["llama31", "perf-fake", "replit"], case_sensitive=False),
    default="llama31",
)
@click.option(
    "--port",
    type=int,
    default=9090,
)
@click.option(
    "--max-batch-size",
    type=int,
    default=8,
)
@click.option(
    "--use-cpu",
    is_flag=True,
    default=False,
    help="Use the CPU to perform inference.",
)
def serve(
    bypass_serve: bool,
    model: str,
    port: int,
    max_batch_size: int,
    use_cpu: bool,
):
    server_config = max_grpc.GRPCConfig(
        port=port, num_workers=10, max_batch_size=max_batch_size
    )
    if not bypass_serve:
        if model == "perf-fake":
            # Doesn't work!
            model_name = "echo"
            pipeline_config = PipelineConfig(
                model_path="modularai/llama-3.1",
            )
            fake_model_factory = functools.partial(
                PerformanceFakingTokenGenerator,
                ce_baseline=0,
                ce_rate=0,
                ce_padding=False,
                tg_baseline=0,
                tg_rate_no_context=0,
                tg_rate_per_context_token=0,
                tg_padding=False,
                busy_wait=False,
            )
            asyncio.run(
                max_grpc.grpc_serve(
                    server_config,
                    model_name,
                    pipeline_config,
                    fake_model_factory,
                )
            )
        else:
            if model == "llama31":
                # works!
                model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
                pipeline_config = get_default_llama31_config(use_cpu)
            elif model == "replit":
                model_name = "replit/replit-code-v1_5-3b"
                pipeline_config = get_default_replit_config(use_cpu)
            else:
                # TODO arekay - better error handling
                print(f"ERROR invalid model name {model}")
                exit(-1)

            pipeline_config.max_batch_size = max_batch_size
            _, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
                pipeline_config
            )
            asyncio.run(
                max_grpc.grpc_serve(
                    server_config,
                    model_name,
                    pipeline_config,
                    pipeline_factory,
                )
            )
    else:
        if model == "perf-fake":
            # Works
            model_name = "perf-fake"
            fake_tokenizer = PerformanceFakingPipelineTokenizer(
                AutoTokenizer.from_pretrained("modularai/llama-3.1")
            )
            fake_pipeline = PerformanceFakingTokenGenerator(
                ce_baseline=0,
                ce_rate=0,
                ce_padding=False,
                tg_baseline=0,
                tg_rate_no_context=0,
                tg_rate_per_context_token=0,
                tg_padding=False,
                busy_wait=False,
            )
            asyncio.run(
                max_grpc.grpc_serve_direct(
                    server_config,
                    model_name,
                    fake_tokenizer,
                    fake_pipeline,
                )
            )
        elif model == "llama31":
            # Works!
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            pipeline_config = get_default_llama31_config(use_cpu)
            pipeline_config.max_batch_size = max_batch_size
            llama_tokenizer, llama_pipeline = PIPELINE_REGISTRY.retrieve(
                pipeline_config
            )
            assert isinstance(llama_tokenizer, TextTokenizer)
            assert isinstance(llama_pipeline, TextGenerationPipeline)
            asyncio.run(
                max_grpc.grpc_serve_direct(
                    server_config, model_name, llama_tokenizer, llama_pipeline
                )
            )
        elif model == "replit":
            model_name = "replit/replit-code-v1_5-3b"
            pipeline_config = get_default_replit_config(use_cpu)
            pipeline_config.max_batch_size = max_batch_size
            replit_tokenizer, replit_pipeline_factory = (
                PIPELINE_REGISTRY.retrieve_factory(
                    pipeline_config,
                )
            )
            assert isinstance(replit_tokenizer, TextTokenizer)
            assert isinstance(replit_pipeline_factory, TextGenerationPipeline)
            asyncio.run(
                max_grpc.grpc_serve_direct(
                    server_config,
                    model_name,
                    replit_tokenizer,
                    replit_pipeline_factory,
                )
            )
        else:
            raise ValueError(f"invalid model name {model}")


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    register_all_models()
    serve()
