# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import importlib
import inspect
import multiprocessing as mp
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import torch
from max import pipelines
from max.entrypoints.cli import DevicesOptionType
from max.interfaces import RequestID, SamplingParams, TextGenerationRequest
from max.nn.hooks import PrintHook
from max.pipelines import TextGenerationPipeline
from max.pipelines.lib import PipelineModel
from safetensors.torch import save_file
from test_common.torch_print_hook import TorchPrintHook
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

T = TypeVar("T")

### DEFAULTS ###

MAX_NEW_TOKENS = 1
PROMPT = "What is the meaning of life?"

### Process Utilities ###


def _worker(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    conn: Any,  # mp.connection.Connection
) -> None:
    conn.send(fn(*args, **kwargs))
    conn.close()


def run_in_fresh_process(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker, args=(fn, args, kwargs, child))
    p.start()
    p.join()
    if parent.poll():
        return parent.recv()
    raise RuntimeError("Subprocess failed without returning a result")


### Model Utilities ###


def generate_weights(model_dir: str) -> None:
    path = Path(model_dir)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModel.from_config(cfg, trust_remote_code=True)
    save_file(mdl.state_dict(), path / "model.safetensors")


def maybe_generate_weights(model_path: str) -> None:
    if not Path(model_path).is_dir():
        return
    path = Path(model_path)
    cfg = path / "config.json"
    if not cfg.exists():
        return
    safes = list(path.glob("*.safetensors"))
    if not safes:
        generate_weights(model_path)
        return
    cfg_m = cfg.stat().st_mtime
    newest = max(f.stat().st_mtime for f in safes)
    if cfg_m > newest:
        generate_weights(model_path)


### Tensor Inspector Class ###


class TensorInspector:
    """TensorInspector class for inspecting tensors of a model."""

    def __init__(
        self,
        model_path: str,
        device: str = "gpu",
    ):
        self.model_path = model_path
        self.device = device
        self.prompt = PROMPT
        self.max_new_tokens = MAX_NEW_TOKENS

    @contextmanager
    def print_module_tensors(self, pipe: Any) -> Any:
        """Context manager that adds tensor printing hooks by patching the model class."""

        # Get the module path from the model
        model_module = importlib.import_module(pipe._pipeline_model.__module__)

        # Find the class that inherits from PipelineModel
        model_class = None
        for _, obj in inspect.getmembers(model_module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, PipelineModel)
                and obj != PipelineModel
            ):
                model_class = obj
                break
        hook = PrintHook()
        assert model_class is not None

        # Store the original __init__ in a closure
        def get_wrapped_init(
            original_init: Callable[..., None],
        ) -> Callable[..., None]:
            def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                original_init(self, *args, **kwargs)
                hook.name_layers(self)

            return wrapped_init

        original_init = model_class.__init__
        model_class.__init__ = get_wrapped_init(original_init)  # type: ignore[method-assign]

        try:
            yield
        finally:
            hook.remove()
            model_class.__init__ = original_init  # type: ignore[method-assign]

    def generate_max(self) -> str:
        _, pipe = pipelines.PIPELINE_REGISTRY.retrieve(
            pipelines.PipelineConfig(
                device_specs=DevicesOptionType.device_specs(
                    DevicesOptionType.parse_from_str(self.device)
                ),
                model_path=self.model_path,
                trust_remote_code=True,
                max_num_steps=1,
            )
        )

        # We call the retrieve() again to get a fresh pipeline with the hooks added
        with self.print_module_tensors(pipe):
            tok, pipe = pipelines.PIPELINE_REGISTRY.retrieve(
                pipelines.PipelineConfig(
                    device_specs=DevicesOptionType.device_specs(
                        DevicesOptionType.parse_from_str(self.device)
                    ),
                    model_path=self.model_path,
                    trust_remote_code=True,
                    max_num_steps=-1,
                )
            )
            if not isinstance(pipe, TextGenerationPipeline):
                raise TypeError(
                    f"This tool currently only supports text generation models. "
                    f"Got {type(pipe).__name__} instead."
                )

            out = pipe.generate(
                TextGenerationRequest(
                    request_id=RequestID(),
                    model_name=self.model_path,
                    prompt=self.prompt,
                    sampling_params=SamplingParams(
                        top_k=1, max_new_tokens=self.max_new_tokens
                    ),
                )
            )[0]
            return asyncio.run(tok.decode(out.tokens)).strip()

    def generate_torch(self) -> str:
        torch.set_grad_enabled(False)  # pyright: ignore[reportAttributeAccessIssue]
        torch_device = "cuda" if self.device == "gpu" else "cpu"
        tok = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        ).to(torch_device)

        hook = TorchPrintHook()
        hook.name_layers(model)

        inputs = tok(self.prompt, return_tensors="pt").to(torch_device)

        out = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1] :]

        out = tok.decode(gen, skip_special_tokens=True).strip()
        return out


def print_tensors(
    model_path: str,
    framework: str,
    device: str = "gpu",
) -> str:
    inspector = TensorInspector(model_path, device)
    if framework == "torch":
        return inspector.generate_torch()
    else:
        return inspector.generate_max()
