# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# mypy: disable-error-code="import-not-found"

import asyncio
import functools

import click
import max.serve.grpc_serve.grpc_serve as max_grpc
from max.driver import DeviceSpec
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextTokenizer,
    TextGenerationPipeline,
)
from max.pipelines.kv_cache import KVCacheStrategy
from max.serve.pipelines.performance_fake import (
    PerformanceFakingPipelineTokenizer,
    PerformanceFakingTokenGenerator,
)
from transformers import AutoTokenizer


def get_default_replit_config() -> PipelineConfig:
    pipeline_config = PipelineConfig(
        huggingface_repo_id="modularai/replit-code-1.5",
        architecture="MPTForCausalLM",
        trust_remote_code=True,
        save_to_serialized_model_path="/tmp/replit_gpu_16.mef",
        quantization_encoding=SupportedEncoding.float32,
        # serialized_model_path="/tmp/replit_gpu_16.mef",
        # pipeline_config.weight_path = hf_file.download()
    )
    return pipeline_config


def get_default_llama31_config() -> PipelineConfig:
    return PipelineConfig(
        architecture=None,
        version="3.1",
        huggingface_repo_id="modularai/llama-3.1",
        device_spec=DeviceSpec(id=0, device_type="gpu"),
        quantization_encoding=SupportedEncoding.bfloat16,
        max_length=512,
        max_new_tokens=512,
        max_cache_batch_size=16,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
        max_num_steps=1,
        pad_to_multiple_of=2,
        top_k=None,
        # save_to_serialized_model_path="/tmp/llama31_gpu_16.mef",
        # serialized_model_path="/tmp/llama31_gpu_16.mef",
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
def serve(bypass_serve: bool, model: str, port: int):
    server_config = max_grpc.GRPCConfig(port=port, num_workers=10)
    if not bypass_serve:
        if model == "llama31":
            # works!
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            pipeline_config = get_default_llama31_config()
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
        elif model == "perf-fake":
            # Doesn't work!
            model_name = "echo"
            pipeline_config = PipelineConfig(
                huggingface_repo_id="modularai/llama-3.1",
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
        elif model == "replit":
            model_name = "replit/replit-code-v1_5-3b"
            pipeline_config = get_default_replit_config()
            _, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
                pipeline_config,
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
            raise ValueError(f"invalid model name {model}")

    else:
        if model == "llama31":
            # Works!
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            pipeline_config = get_default_llama31_config()
            llama_tokenizer, llama_pipeline = PIPELINE_REGISTRY.retrieve(
                pipeline_config
            )
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
        elif model == "perf-fake":
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
        elif model == "replit":
            model_name = "replit/replit-code-v1_5-3b"
            pipeline_config = get_default_replit_config()
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
    serve()
