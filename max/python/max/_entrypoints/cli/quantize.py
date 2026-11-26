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

import contextlib
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from max import tree
from max.driver import CPU, Accelerator, accelerator_count
from max.nn import Module
from max.nn._weight_utils import (
    _ExternalWeight,
    name_weights,
    shard_checkpoint,
)
from max.nn.common_layers.quant_tensor import (
    MXFP4Quantizer,
    MXFP8Quantizer,
    NVFP4TensorQuantizer,
)
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib import (
    PipelineArgs,
    PipelineConfig,
)
from max.pipelines.lib.pipeline_variants.text_generation import (
    TextGenerationPipelineInterface,
)
from max.pipelines.lib.quantization import (
    SOURCE_FORMATS,
    _check_expert_parallel_target,
    _enable_error_prints,
    _graphs_without_compile,
    _retarget_dispatch,
    insert_quantized_linears,
    on_the_fly_config,
    quant_formats,
    quantizing_factory,
)
from max.pipelines.weights.checkpoint_files import (
    ShardWriter,
    _host_tensor,
    _SavedTensor,
    copy_auxiliary_files,
    write_quantization_config,
)
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    nvfp4_quantization_config,
)
from max.tensor import Tensor, compilation

from .quant_error_report import JSON_NAME, write_report

logger = logging.getLogger("max._entrypoints")

# Re-export for callers and tests that still import these from the CLI module.
__all__ = [
    "SOURCE_FORMATS",
    "_check_expert_parallel_target",
    "_enable_error_prints",
    "_graphs_without_compile",
    "_retarget_dispatch",
    "checkpoint_quantization_config",
    "export_quantized_checkpoint",
    "insert_quantized_linears",
    "on_the_fly_config",
    "quant_formats",
    "quantize_pipeline",
    "quantizing_factory",
]


# Leaves each on-the-fly quantizer writes, in checkpoint order. The name is
# the safetensors suffix: ``weight`` becomes ``{path}.weight``.
_QUANTIZER_LEAVES: tuple[tuple[type[Any], tuple[str, ...]], ...] = (
    (
        NVFP4TensorQuantizer,
        ("weight", "weight_scale", "weight_scale_2", "input_scale"),
    ),
    (MXFP8Quantizer, ("weight", "weight_scale")),
    (MXFP4Quantizer, ("weight", "weight_scale")),
)


def quantized_host_tensors(
    path: str, leaves: Mapping[str, Tensor]
) -> list[tuple[str, _SavedTensor]]:
    """The quantized weight and its scale tensors, under ``path``."""
    return [
        (f"{path}.{name}", _host_tensor(tensor))
        for name, tensor in leaves.items()
    ]


def checkpoint_quantization_config(
    fmt: str, ignored: Sequence[str]
) -> dict[str, object]:
    """The ``quantization_config`` block ``fmt``'s loader already reads.

    MXFP8 is modelopt: encoding detection reads ``quantization.quant_algo``,
    and the layer parser reads ``ignore`` from that same nested section.
    MXFP4's parser keys only on ``quant_method`` and quantizes every MLP,
    leaving attention in bf16.
    """
    if fmt == "nvfp4":
        return nvfp4_quantization_config(FP4Format.NVFP4, ignored)
    if fmt == "mxfp8":
        patterns = list(ignored)
        return {
            "quant_method": "modelopt",
            "quantization": {"quant_algo": "MXFP8", "ignore": patterns},
        }
    if fmt == "mxfp4":
        return {"quant_method": "mxfp4"}
    raise ValueError(
        f"unknown quantization format {fmt!r}; this command can "
        f"produce {', '.join(quant_formats())}"
    )


def _ignore_pattern(name: str, quantized_stems: Iterable[str]) -> str | None:
    """The ignore glob for one tensor that was copied unchanged.

    A layer subtree that also holds a quantized weight keeps this tensor's
    own module, so ``mlp.gate*`` does not unquantize the experts beside it.
    A subtree with nothing quantized is ignored whole, as ``self_attn*``.
    """
    match = re.search(r"^(.*\.layers\.\d+\.)([^.]+)", name)
    if match is None:
        return None
    top = f"{match.group(1)}{match.group(2)}"
    if any(stem.startswith(f"{top}.") for stem in quantized_stems):
        return f"{name.rsplit('.', 1)[0]}*"
    return f"{top}*"


def _state_key(path: str) -> str:
    """The checkpoint key for a quantizer found at ``path``."""
    return path if path.endswith(".weight") else f"{path}.weight"


def _source_files(model: Any) -> dict[str, str]:
    """Maps a checkpoint tensor name to the safetensors file holding it."""
    src = Path(model.pipeline_config.model.resolved_weight_paths()[0]).parent
    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file():
        return {}
    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    return weight_map if isinstance(weight_map, dict) else {}


def _release_cached_weight(model: Any, checkpoint_name: str) -> None:
    """Drops one safetensors view from the loader cache."""
    cache = getattr(getattr(model, "weights", None), "_st_weight_map", None)
    if isinstance(cache, dict):
        cache.pop(checkpoint_name, None)


def _module_at_paths(quantizers: Mapping[str, Any]) -> Module[Any, Any]:
    """A module whose parameters sit at the same paths as ``quantizers``."""
    root: Module[Any, Any] = Module()
    for path, quantizer in quantizers.items():
        cursor = root
        parts = path.split(".")
        for part in parts[:-1]:
            child = getattr(cursor, part, None)
            if not isinstance(child, Module):
                child = Module()
                setattr(cursor, part, child)
            cursor = child
        setattr(cursor, parts[-1], quantizer)
    return root


@contextlib.contextmanager
def _place_host_quantizers(module: Module[..., Any]) -> Iterator[None]:
    """Moves host-only quantizers onto a GPU for the body of the block.

    The block-scaled quantize kernel runs on a GPU. MiniMax keeps each
    routed expert on the host until ``stacked()`` places it. With no
    accelerator the weights stay where they are. The wrapped tensor is
    moved back to the host when the block ends.
    """
    placed: list[Any] = []
    gpu: Accelerator | None = None
    if accelerator_count() == 0:
        yield
        return
    for cls, _names in _QUANTIZER_LEAVES:
        for quantizer in tree.nodes(module, cls).values():
            tensor = getattr(quantizer, "_tensor", None)
            if not isinstance(tensor, Tensor) or not all(
                isinstance(device, CPU)
                for device in tensor.mapping.mesh.devices
            ):
                continue
            if gpu is None:
                gpu = Accelerator(0)
            placed.append(quantizer)
            quantizer.to(gpu)
    try:
        yield
    finally:
        for quantizer in placed:
            quantizer.to(CPU())


def _materialize_quantizers(
    module: Module[..., Any],
    state_dict: Mapping[str, Any],
) -> dict[str, dict[str, Tensor]]:
    """Runs the quantizers and returns packed weights and scales.

    The generate trace's quant tensors are graph values with no host bytes,
    so this graph produces the buffers that get written.
    """
    with _place_host_quantizers(module):
        declared = tree.map(
            _ExternalWeight.like, module, leaf=Tensor, shared=True
        )
        name_weights(declared)
        registry = shard_checkpoint(declared, state_dict)

        def export_quants() -> dict[str, dict[str, Tensor]]:
            exported: dict[str, dict[str, Tensor]] = {}
            for cls, names in _QUANTIZER_LEAVES:
                for path, quantizer in tree.nodes(declared, cls).items():
                    quantized = quantizer.quant_tensor
                    exported[path] = {
                        name: getattr(quantized, name) for name in names
                    }
            return exported

        # ``compile`` returns a function of input types. This graph takes none.
        # The session is the process-wide one: the weights carry their devices.
        compiled = compilation.compile(export_quants, weights=registry)()
        try:
            return compiled()
        finally:
            del compiled


def export_quantized_checkpoint(
    model: Any, dest: Path, quantization_format: str
) -> None:
    """Writes a new sharded checkpoint from the quantizers on ``model``.

    Quantizers are grouped by the safetensors file that holds their source
    weight. Each file is quantized, written, and dropped from the loader
    cache before the next file is read. Tensors left unquantized are copied
    under their original names. Anything not on the decoder is copied from
    the source shard unchanged.
    """
    module = model.nn_module
    if not isinstance(module, Module):
        raise TypeError(
            f"{type(module).__name__} is not a ModuleV3 module, so its "
            "quantizers cannot be exported"
        )
    state_dict = dict(model.final_state_dict)
    # ``found`` is what ``ready`` will be, before this file is quantized.
    found: list[tuple[str, Any, str]] = []
    for cls, _names in _QUANTIZER_LEAVES:
        for path, quantizer in tree.nodes(module, cls).items():
            state_key = _state_key(path)
            found.append((path, quantizer, state_dict[state_key].name))
    if not found:
        logger.info("No quantizers inserted, aborting export.")
        return

    files = _source_files(model)
    quantized_keys = {_state_key(path) for path, _, _ in found}
    quantized_stems = [name.removesuffix(".weight") for _, _, name in found]

    def source_file(checkpoint_name: str) -> str:
        return str(files.get(checkpoint_name, ""))

    groups: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for path, quantizer, checkpoint_name in found:
        groups[source_file(checkpoint_name)].append((path, quantizer))
    copied: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for key, value in state_dict.items():
        if key not in quantized_keys:
            copied[source_file(value.name)].append((key, value))

    src = Path(model.pipeline_config.model.resolved_weight_paths()[0]).parent
    shards = set(files.values())
    writer = ShardWriter(dest)
    ignore: set[str] = set()
    exported = 0
    for filename in sorted(set(groups) | set(copied) | shards):
        batch = groups.get(filename, [])
        ready: dict[str, dict[str, Tensor]] = {}
        if batch:
            logger.info(
                "Quantizing %s (%d weights).",
                filename or "checkpoint",
                len(batch),
            )
            ready = _materialize_quantizers(
                _module_at_paths(dict(batch)), state_dict
            )
            exported += len(ready)
        saved: dict[str, _SavedTensor] = {}
        for path, leaves in ready.items():
            source = state_dict[_state_key(path)]
            checkpoint_path = source.name.removesuffix(".weight")
            for name, tensor in quantized_host_tensors(checkpoint_path, leaves):
                saved[name] = tensor
            _release_cached_weight(model, source.name)
        if filename in shards:
            for name in writer.copy_except(src / filename, saved):
                _release_cached_weight(model, name)
        else:
            for name, tensor in saved.items():
                writer.add(name, tensor)
        for key, value in copied.get(filename, []):
            if value.name not in files:
                writer.add(value.name, _host_tensor(value))
            pattern = _ignore_pattern(value.name, quantized_stems)
            if pattern is not None:
                ignore.add(pattern)
            _release_cached_weight(model, value.name)
            del state_dict[key]
        del ready
    writer.close()

    write_quantization_config(
        src,
        dest,
        checkpoint_quantization_config(quantization_format, sorted(ignore)),
    )
    copy_auxiliary_files(src, dest)
    logger.info("Exported %d quantized weights to %s", exported, dest)


def quantize_pipeline(
    pipeline_args: PipelineArgs,
    *,
    export_path: Path,
) -> None:
    """Quantizes the targeted layers and writes a quantized checkpoint.

    On-the-fly settings come from ``pipeline_args.quantization_format`` and
    related fields, assembled into ``PipelineConfig.quant_config``. The
    pipeline is built with ``export_only`` so no generate graph is compiled.
    """
    if pipeline_args.quantization_format is None:
        raise ValueError("quantize needs quantization_format")
    pipeline_config = PipelineConfig.from_args(pipeline_args)
    if pipeline_config.quant_config is None:
        raise ValueError("quantize needs a QuantConfig on the pipeline config")
    pipeline_config = pipeline_config.model_copy(update={"export_only": True})
    retrieved = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config, plan_memory=False
    )
    # Export uses this same pipeline. Its constructor returns once the model
    # exists, before the KV cache and the sampler.
    pipeline = retrieved.factory()
    if not isinstance(pipeline, TextGenerationPipelineInterface):
        raise TypeError(
            f"{type(pipeline).__name__} has no pipeline model to export"
        )
    logger.info(
        "Exporting quantized weights to %s",
        export_path,
    )
    export_quantized_checkpoint(
        pipeline._pipeline_model,
        export_path,
        pipeline_args.quantization_format,
    )
    if pipeline_args.error_report_dir is not None:
        write_report(pipeline_args.error_report_dir)
        logger.info(f"Wrote {pipeline_args.error_report_dir / JSON_NAME}")
