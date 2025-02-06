# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs evaluations for the Massive Text Embedding Benchmark.

./bazelw run SDK/integration-test/pipelines/python:pipelines_mteb --\
    --huggingface-repo-id=sentence-transformers/all-mpnet-base-v2 \
    --max-cache-batch-size=1000 \
    --eval-task="STSBenchmark" \
    --eval-output-folder=$PWD/mteb-output

To get results from a reference HuggingFace model, add
  --model-library=mteb

You can also use `--engine=huggingface` (instead of `--model-library=mteb`) to
run a benchmark with huggingface pipeline model
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from functools import cached_property
from typing import Optional, Sequence

import click
import mteb
import numpy as np

# Pipelines
from max.entrypoints.cli import pipeline_config_options
from max.pipelines import (
    PIPELINE_REGISTRY,
    EmbeddingsGenerator,
    PipelineConfig,
    PipelineEngine,
)
from max.pipelines.architectures import register_all_models
from max.pipelines.interfaces import (
    PipelineTask,
    PipelineTokenizer,
    TokenGeneratorRequest,
)


class EmbeddingModel:
    """Implements the MTEB model interface."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        tokenizer: PipelineTokenizer,
        pipeline: EmbeddingsGenerator,
    ):
        self.pipeline_config = pipeline_config
        self.tokenizer = tokenizer
        self.pipeline = pipeline

    @cached_property
    def mteb_model_meta(self) -> mteb.ModelMeta:
        if self.pipeline_config.engine == PipelineEngine.MAX:
            name = f"max_{self.pipeline_config.huggingface_repo_id}"
        elif self.pipeline_config.engine == PipelineEngine.HUGGINGFACE:
            name = f"max_hf_{self.pipeline_config.huggingface_repo_id}"
        else:
            name = f"max_{self.pipeline_config.engine}_{self.pipeline_config.huggingface_repo_id}"

        if meta := mteb.models.MODEL_REGISTRY.get(
            self.pipeline_config.huggingface_repo_id
        ):
            return meta.model_copy(update={"name": name})
        else:
            config = self.pipeline_config.huggingface_config
            return mteb.ModelMeta(
                name=name,
                revision=None,
                release_date=None,
                languages=None,
                n_parameters=None,
                max_tokens=config.max_seq_len,
                embed_dim=config.hidden_size,
                license=None,
                open_weights=True,
                public_training_code=None,
                public_training_data=None,
                framework=[],
                similarity_fn_name=None,
                use_instructions=False,
                training_datasets=None,
            )

    def encode(self, sentences: Sequence[str], **kwargs) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder (unused).

        Returns:
            The encoded sentences.
        """
        if self.pipeline_config.max_batch_size is None:
            logger.warning(
                "Please set --max-cache-batch-size, otherwise the batch size"
                " will automatically be set to 1."
            )
            batch_size = 1
        else:
            batch_size = self.pipeline_config.max_batch_size
        start = 0
        loop = asyncio.get_event_loop()
        results = []
        while start < len(sentences):
            batch = sentences[start : start + batch_size]
            results.extend(loop.run_until_complete(self.batch_encode(batch)))
            start += batch_size
        return np.array(results)

    async def batch_encode(self, sentences: Sequence[str]) -> list[np.ndarray]:
        pipeline_request = {}
        for n, sentence in enumerate(sentences):
            pipeline_request[str(n)] = await self.tokenizer.new_context(
                TokenGeneratorRequest(
                    id=str(n),
                    index=n,
                    prompt=sentence,
                    model_name=self.pipeline_config.huggingface_repo_id,
                )
            )
        response = self.pipeline.encode(pipeline_request)
        results = []
        for n in range(len(sentences)):
            embeddings = response[str(n)].embeddings
            results.append(embeddings)
        return results


logger = logging.getLogger("pipelines_mteb")


@click.command()
@pipeline_config_options
@click.option(
    "--model-library",
    type=click.Choice(["pipeline", "mteb"]),
    default="pipeline",
    help=(
        "Use this to choose how the evaluator loads the model. If 'pipeline' "
        "is chosen, the model will be chosen from the pipeline model registry. "
        " If 'mteb' is chosen, the model's huggingface_repo_id will be passed "
        "to the mteb library. This is separate from the --engine flag, which "
        "is used with --model-library=pipeline (but can be used to load a "
        "HuggingFace model)."
    ),
)
@click.option("--eval-benchmark", type=str)
@click.option("--eval-task", type=str)
@click.option("--eval-output-folder", type=str)
@click.option(
    "--list",
    "list_all",
    type=bool,
    is_flag=True,
    default=False,
    help="List all available benchmarks and tasks.",
)
def main(
    *,
    model_library: str,
    eval_benchmark: Optional[str] = None,
    eval_task: Optional[str] = None,
    eval_output_folder: Optional[str] = None,
    list_all: bool,
    **config_kwargs,
) -> None:
    """Runs a MTEB evaluation benchmark or task on a model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
    )

    if list_all:
        benchmarks = [b.name for b in mteb.get_benchmarks()]
        task_list = [t.metadata.name for t in mteb.get_tasks()]
        print("Available benchmarks:", ", ".join(benchmarks))
        print()
        print("Available tasks:", ", ".join(task_list))

    if not eval_benchmark and not eval_task:
        logger.warning("No benchmark or task selected, exiting.")
        sys.exit(0)

    if eval_benchmark and eval_task:
        logger.error(
            "Both a benchmark and task were requested, please only set one."
        )
        sys.exit(0)

    # orc_rt fix.
    if workspace_dir := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(workspace_dir)

    pipeline_config = PipelineConfig(**config_kwargs)

    model: EmbeddingModel | mteb.encoder_interface.Encoder
    logging.info("Loading model with %s library." % model_library)
    if model_library == "mteb":
        model = mteb.get_model(pipeline_config.huggingface_repo_id)
    else:
        register_all_models()
        tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(
            pipeline_config, task=PipelineTask.EMBEDDINGS_GENERATION
        )
        assert isinstance(pipeline, EmbeddingsGenerator)
        model = EmbeddingModel(pipeline_config, tokenizer, pipeline)

    tasks: mteb.Benchmark | mteb.overview.MTEBTasks
    if eval_benchmark:
        tasks = mteb.get_benchmark(eval_benchmark)
    elif eval_task:
        tasks = mteb.get_tasks(tasks=[eval_task])
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(
        model, output_folder=eval_output_folder, overwrite_results=True
    )


if __name__ == "__main__":
    main()
