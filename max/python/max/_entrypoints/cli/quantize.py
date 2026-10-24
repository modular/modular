# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import logging
import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import requests
from max import tree
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.quant_linear import (
    QuantizedLinear,
)
from max.experimental.tensor import Tensor
from max.nn.quant_config import QuantConfig, WeightSource
from max.pipelines import (
    PIPELINE_REGISTRY,
)
from max.pipelines.context import (
    SamplingParams,
)
from max.pipelines.lib import (
    ModuleV3PipelineModelWithKVCache,
    PipelineArgs,
    PipelineConfig,
)
from max.pipelines.logging_utils import log_basic_config
from max.pipelines.modeling.types import (
    Pipeline,
)
from max.profiler import OneShotCapture, Tracer

from .generate import stream_text_to_console
from .metrics import TextGenerationMetrics

logger = logging.getLogger("max._entrypoints")

MODEL_NAME = "model"


def insert_quantized_linears(
    module: Module[Any, Any],
    targets: Iterable[str],
    quant_config: QuantConfig,
) -> None:
    """Replaces every ``Linear`` matching ``targets`` with a quantized one.

    The config is supplied rather than built here: it is the caller's
    statement of intent.
    """
    count = 0
    for key, value in tree.nodes(module, Linear).items():
        if any(re.compile(pattern).search(key) for pattern in targets):
            wrapped = QuantizedLinear(
                int(value.in_dim),
                int(value.out_dim),
                bias=isinstance(value.bias, Tensor),
                quant_config=quant_config,
            )
            tree.update(
                module,
                {key: wrapped},
                leaf=lambda node, value=value: node is value,
            )
            count += 1
    logger.info(f"Quantizing {count} weight tensors.")


#: The config each ``--quantization-format`` value selects.
#:
#: ``source`` is set here rather than by the caller: the command exists to
#: quantize a native-precision checkpoint, so CHECKPOINT would describe
#: bytes that are not there.
#:
#: The layer tuples are empty because ``QuantConfig`` selects layers by
#: index and this command selects them by name instead:
#: ``insert_quantized_linears`` wraps whatever matches ``--targets``.
#:
#: A row here is necessary but not sufficient -- ``quant_ops`` must have a
#: quantizer registered for the format too, or building the weight raises.
_ON_THE_FLY: dict[str, QuantConfig] = {
    "nvfp4": dataclasses.replace(
        QuantConfig.nvfp4(mlp_quantized_layers=(), attn_quantized_layers=()),
        source=WeightSource.QUANTIZE,
    ),
}


def quant_formats() -> tuple[str, ...]:
    """The values ``--quantization-format`` accepts."""
    return tuple(sorted(_ON_THE_FLY))


def on_the_fly_config(name: str) -> QuantConfig:
    """The config ``--quantization-format`` selects.

    Raises:
        ValueError: if ``name`` is not a format this command can produce.
    """
    try:
        config = _ON_THE_FLY[name]
    except KeyError:
        raise ValueError(
            f"unknown quantization format {name!r}; this command can "
            f"produce {', '.join(quant_formats())}"
        ) from None
    # A copy, because QuantConfig is mutable and the table is shared.
    return dataclasses.replace(config)


def _factory_with_layer_quantizers(
    factory: Callable[[], Any],
    targets: Iterable[str],
    quant_config: QuantConfig,
) -> Callable[[], Any]:
    """Returns a factory that quantizes layers before the module is compiled.

    ``retrieve_factory`` binds the pipeline-model class into the factory.
    The pipeline constructor instantiates that class, and ``load_model``
    compiles whatever ``_instantiate_module`` returns, so the subclass has
    to be substituted before the factory is called.
    """
    if not isinstance(factory, functools.partial):
        raise TypeError(
            "pipeline factory must be a functools.partial so its "
            "pipeline_model class can be replaced"
        )
    keywords = dict(factory.keywords or {})
    model_cls = keywords["pipeline_model"]
    if not issubclass(model_cls, ModuleV3PipelineModelWithKVCache):
        raise TypeError(
            f"{model_cls.__name__} is not a ModuleV3 pipeline model, so "
            "layer quantizers cannot be inserted in _instantiate_module"
        )

    # mypy only accepts a class as a base. The runtime base is the concrete
    # pipeline model retrieved for this checkpoint.
    if TYPE_CHECKING:
        pipeline_model_base = ModuleV3PipelineModelWithKVCache[Any, Any]
    else:
        pipeline_model_base = model_cls

    class QuantizingPipelineModel(pipeline_model_base):
        def _instantiate_module(self, model_config: Any) -> Any:
            nn_model = super()._instantiate_module(model_config)
            insert_quantized_linears(nn_model, targets, quant_config)
            return nn_model

    QuantizingPipelineModel.__name__ = f"Quantizing{model_cls.__name__}"
    QuantizingPipelineModel.__qualname__ = QuantizingPipelineModel.__name__
    keywords["pipeline_model"] = QuantizingPipelineModel
    return functools.partial(factory.func, *factory.args, **keywords)


def quantize_pipeline(
    pipeline_args: PipelineArgs,
    sampling_params: SamplingParams,
    targets: Iterable[str],
    prompt: str,
    quantization_format: str,
    image_urls: Iterable[str] = (),
    num_warmups: int = 0,
    profile: bool = False,
    profile_top_n: int = 15,
) -> None:
    quant_config = on_the_fly_config(quantization_format)
    pipeline_config = PipelineConfig.from_args(pipeline_args)
    retrieved = PIPELINE_REGISTRY.retrieve_factory(pipeline_config)
    tokenizer = retrieved.tokenizer
    pipeline = _factory_with_layer_quantizers(
        retrieved.factory, targets, quant_config
    )()
    assert isinstance(pipeline, Pipeline)
    log_basic_config(pipeline_config, memory_plan=retrieved.memory_plan)
    logger.info("max_batch_size: %d", pipeline.max_batch_size)

    capture = OneShotCapture(top_n=profile_top_n) if profile else None
    try:
        # Run timed run & print results.
        with TextGenerationMetrics(print_report=True) as metrics:
            if image_urls:
                logger.info("Downloading images")
                images = [requests.get(url).content for url in image_urls]
            else:
                images = []

            if num_warmups > 0:
                logger.info("Running warmup")
                warmup_params = dataclasses.replace(
                    sampling_params, max_new_tokens=num_warmups
                )
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        sampling_params=warmup_params,
                        metrics=None,
                        print_tokens=False,
                    )
                )

            # Run and print results.
            logger.info("Beginning text generation")

            with contextlib.ExitStack() as exit_stack:
                if capture is not None:
                    capture.start()
                    # ``Tracer`` adds an NVTX "inference" label visible in
                    # nsys-ui. It must be entered after ``capture.start``
                    # so its NVTX range sits inside the cuda profiler
                    # window.
                    exit_stack.enter_context(
                        Tracer("inference", color="modular_purple")
                    )
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        sampling_params=sampling_params,
                        metrics=metrics,
                        print_tokens=True,
                    )
                )
        # ``TextGenerationMetrics.__exit__`` has printed the report here.
        # End the capture *now*, so nsys's file-writing output follows.
    finally:
        if capture is not None:
            capture.end_and_finalize()
