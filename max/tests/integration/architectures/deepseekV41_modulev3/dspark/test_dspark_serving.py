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
"""CPU tests of the ModuleV3 DSpark pairing's host-side wiring.

Everything under test is declaration, module construction and host-side
buffer assembly: the pairing the generic speculative pipeline selects, the
trailing extras the target adapter declares, the trunk and drafter each
building on a data-parallel mesh, and the trunk's batch processor staging
the extras with the mask at the merged width. Nothing here traces the fused
module -- the trunk's fused kernels and the drafter's MXFP4 blocks are
GPU-only.
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from dspark_test_support import TESTDATA
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.deepseekV41_modulev3 import (
    model as model_module,
)
from max.pipelines.architectures.deepseekV41_modulev3 import (
    spec_adapters,
)
from max.pipelines.architectures.deepseekV41_modulev3.arch import (
    deepseekV41_modulev3_arch,
)
from max.pipelines.architectures.deepseekV41_modulev3.batch_processor import (
    DeepseekV41BatchProcessor,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
    extra_input_names,
    extra_inputs,
)
from max.pipelines.architectures.deepseekV41_modulev3.model import (
    DeepseekV41Model,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
    dspark_draft_width,
)
from max.pipelines.architectures.dspark_deepseekV41_modulev3 import (
    DSparkDeepseekV41ModuleV3DraftArchConfig,
    dspark_deepseekV41_modulev3_draft_arch,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.lib import (
    BatchProcessorRuntime,
    KVCacheConfig,
    MAXModelConfig,
    ModelManifest,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.lib.arch_lookup import ARCH_LOOKUP
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
    spec_decode_architecture,
)
from max.pipelines.sampling import SamplingConfig
from max.pipelines.speculative import spec_adapters as registry
from max.pipelines.speculative.config import SpeculativeConfig

BLOCK = 5
"""The checkpoint's draft block width, also the spec-decode step width."""
BATCH = 2
PROMPTS = ([1, 2, 3], [4, 5])
"""The two requests' prompts: five merged prompt rows between them."""
TARGET = "DeepseekV41ForCausalLM_ModuleV3"


def _hf_dict() -> dict[str, Any]:
    """The checkpoint's own config, with every real text field."""
    return cast(dict[str, Any], json.load(open(TESTDATA / "config_full.json")))


def _v41_config() -> V41Config:
    """The trunk's frozen interface view, at the fixture's table extents.

    Every field is the real config's except the Engram table row counts,
    the interface's only unbounded extents (384M rows per table in the
    checkpoint -- sizes no CPU fixture builds). Everything under test reads
    each table's name, never its rows.
    """
    config = V41Config.from_hf(_hf_dict())
    return dataclasses.replace(config, engram_num_embeddings=(4, 4))


def _arch_config(n_devices: int = 1) -> DeepseekV41Config:
    """The architecture config, with the interface's own KV-leaf tree."""
    text = SimpleNamespace(**_hf_dict()["text_config"])
    devices = [DeviceRef.CPU()] * n_devices
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=build_kv_params(
            _v41_config(),
            KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
            devices,
            data_parallel_degree=n_devices,
        ),
        devices=devices,
        max_seq_len=64,
        **DeepseekV41Config.resolve_text_fields(text),
    )


def _pipeline_config() -> PipelineConfig:
    """A config carrying only what the processor reads off it."""
    model = MAXModelConfig.model_construct(
        model_path=str(TESTDATA),
        trust_remote_code=False,
    )
    model._huggingface_config = _hf_dict()
    return PipelineConfig.model_construct(
        models=ModelManifest({"main": model}),
        runtime=PipelineRuntimeConfig(
            max_batch_input_tokens=512, enable_chunked_prefill=True
        ),
        sampling=SamplingConfig(),
    )


def _runtime() -> BatchProcessorRuntime:
    return BatchProcessorRuntime(
        pipeline_config=_pipeline_config(),
        devices=[CPU()],
        return_logits=ReturnLogits.VARIABLE,
        max_batch_size=8,
        max_seq_len=64,
        max_batch_input_tokens=512,
        enable_chunked_prefill=True,
        data_parallel_degree=1,
    )


def _engram_static() -> dict[str, Buffer]:
    """The pinned tables and the token map, one small buffer per name."""
    config = _v41_config()
    static: dict[str, Buffer] = {}
    for ordinal, rows in enumerate(config.engram_num_embeddings):
        static[f"engram_table_{ordinal}_weight"] = Buffer.from_numpy(
            np.zeros((int(rows), 256), np.uint8)
        )
        static[f"engram_table_{ordinal}_scale"] = Buffer.from_numpy(
            np.zeros((int(rows), 8), np.uint8)
        )
    static["engram_token_map"] = Buffer.from_numpy(
        np.zeros(config.vocab_size, np.int32)
    )
    return static


def _processor(
    static: dict[str, Buffer] | None = None, verifies: bool = True
) -> DeepseekV41BatchProcessor:
    """The trunk's processor, as the pairing builds it off the config the
    trunk's build returns."""
    config = _arch_config()
    if static is not None:
        config.engram_static_inputs = static
    if verifies:
        config.dspark_trailing_inputs = extra_inputs(
            _v41_config(), config.devices, verify_rows=True
        )
    return DeepseekV41BatchProcessor(config, _runtime())


def _contexts() -> list[list[TextContext]]:
    """One replica's batch: the two prompts, as fresh contexts."""
    return [
        [
            TextContext(
                tokens=TokenBuffer(np.array(p, np.int64)), max_length=64
            )
            for p in PROMPTS
        ]
    ]


def _kv_inputs() -> KVCacheInputsPerDevice[Buffer, Buffer]:
    """One device's cache values: the six fields every paged leaf carries."""
    return KVCacheInputsPerDevice(
        kv_blocks=Buffer.from_numpy(np.zeros((4, 8, 2), np.float32)),
        cache_lengths=Buffer.from_numpy(np.array([1, 2], np.int32)),
        lookup_table=Buffer.from_numpy(np.zeros((2, 4), np.int32)),
        max_prompt_length=Buffer.from_numpy(np.array([2], np.uint32)),
        max_cache_length=Buffer.from_numpy(np.array([8], np.uint32)),
        page_stride=Buffer.from_numpy(np.array([1], np.int64)),
    )


def _extras(
    processor: DeepseekV41BatchProcessor, return_n_logits: int
) -> dict[str, Buffer]:
    """A prepared batch's trailing values, by the names the graph declares."""
    inputs = processor.prepare_initial_token_inputs(
        _contexts(), _kv_inputs(), return_n_logits=return_n_logits
    )
    return dict(
        zip(
            extra_input_names(processor._v41),
            spec_adapters.batch_trailing_buffers(inputs),
            strict=True,
        )
    )


def test_a_verify_sizes_the_mask_at_the_merged_width() -> None:
    """The extras keep the trunk's order and types, but the mask.

    The verify runs the FFN gate over the merged window, so its mask
    carries the merged rows under the dim the block driver's merge names
    them by; a prompt-width mask would gate the draft rows with another
    row's bit, never an error.
    """
    config = _v41_config()
    refs = [DeviceRef.CPU()]
    trunk = dict(extra_inputs(config, refs))
    verify = dict(extra_inputs(config, refs, verify_rows=True))
    assert list(verify) == list(trunk) == extra_input_names(config)
    for name, dtype in verify.items():
        expected = trunk[name]
        if name == "image_token_mask":
            assert [str(d) for d in dtype.shape] == ["merged_seq_len"]
            assert (dtype.dtype, dtype.device) == (
                expected.dtype,
                expected.device,
            )
        else:
            assert dtype == expected, name


def test_dspark_pairs_the_modulev3_trunk_with_its_modulev3_draft() -> None:
    """``--prefer-module-v3`` with ``dspark`` and no draft model runs the
    generic pairing of the trunk and the checkpoint's ModuleV3 stages."""
    register_all_models()
    paired = spec_decode_architecture(
        "DeepseekV41ForCausalLM", "dspark", None, prefer_module_v3=True
    )
    assert paired is not None
    assert paired.name == f"{TARGET}+dspark"
    assert paired.pipeline_model is not None
    assert issubclass(paired.pipeline_model, SpecDecodePipelineModel)
    assert paired.weight_adapters == deepseekV41_modulev3_arch.weight_adapters
    assert paired.checkpoint_draft_width is dspark_draft_width
    # The trunk captures no device graph, so neither does the pairing.
    assert not paired.supports_device_graph_capture
    assert (
        ARCH_LOOKUP.find("DSparkDeepseekV41DraftModel", prefer_module_v3=True)
        is dspark_deepseekV41_modulev3_draft_arch
    )
    # The serving engine boots the fused graph by its name.
    assert deepseekV41_modulev3_arch.spec_decode_graph_names == {
        "dspark": "UnifiedDSparkDeepseekV41ModuleV3"
    }


def test_the_graph_api_pairing_is_unchanged() -> None:
    """Without ``--prefer-module-v3`` V4.1 pairs its graph-API trunk."""
    register_all_models()
    paired = spec_decode_architecture("DeepseekV41ForCausalLM", "dspark", None)
    assert paired is not None
    assert paired.name == "DeepseekV41ForCausalLM+dspark"


def test_the_modulev3_trunk_has_its_own_target_adapter() -> None:
    """The pairing calls the ModuleV3 trunk through its own adapter, which
    drives the window leaf and prepares every batch."""
    assert registry.primary_kv(TARGET, "dspark") == spec_adapters.WINDOW_LEAF
    assert registry.prepares_every_batch(TARGET)
    assert not registry.relaxed_thinking_acceptance(TARGET)
    config = _arch_config()
    config.dspark_trailing_inputs = extra_inputs(
        _v41_config(), config.devices, verify_rows=True
    )
    names = [name for name, _ in config.dspark_trailing_inputs]
    assert len(registry.trailing_input_types(TARGET, "dspark", config)) == len(
        names
    )
    trailing = [cast(Any, object()) for _ in names]
    extra = registry.extra(TARGET, "dspark", config, trailing)
    assert list(extra) == names
    assert all(extra[n] is v for n, v in zip(names, trailing, strict=True))


def test_prepare_stages_the_extras_in_the_declared_order() -> None:
    """The trunk's processor stages each extra in its declared slot, the
    Engram inputs the trunk's build pinned among them."""
    static = _engram_static()
    processor = _processor(static)
    extras = _extras(processor, BLOCK + 1)
    for name, bound in static.items():
        assert extras[name] is bound, name
    swa = extras["swa_begin"]
    assert swa.dtype == DType.int32
    assert tuple(swa.shape) == (BATCH,)
    dead = extras["engram_lookback_dead"]
    assert dead.dtype == DType.uint8
    assert tuple(dead.shape) == (BATCH, 3)


def test_the_batch_carries_the_pipeline_s_signal_buffers() -> None:
    """The pairing's graph takes the pipeline's signal buffers last, which
    the trunk's batch hands it."""
    processor = _processor(_engram_static())
    inputs = processor.prepare_initial_token_inputs(
        _contexts(), _kv_inputs(), return_n_logits=BLOCK + 1
    )
    assert inputs.signal_buffers == list(processor.runtime.signal_buffers)


def test_the_engram_inputs_must_be_bound_before_the_first_batch() -> None:
    """A trunk built without its tables fails loud, naming the inputs.

    A silent default here would feed the trunk garbage tables and an
    all-zero token map, so the miss is a raise, never a zero-filled
    placeholder.
    """
    with pytest.raises(RuntimeError, match="engram_table"):
        _extras(_processor(), BLOCK + 1)


def test_the_mask_carries_the_merged_rows() -> None:
    """A verify step's mask spans every request's drafts too.

    A prefill verifies nothing: the merged window is the prompt rows
    alone. A decode step merges each request's rows with its draft block,
    the ``return_n_logits - 1`` drafts it verifies. The trunk serving alone
    keeps the prompt width.
    """
    prompt_rows = sum(len(p) for p in PROMPTS)
    processor = _processor(_engram_static())
    prefill = _extras(processor, 1)["image_token_mask"]
    assert prefill.dtype == DType.uint8
    assert tuple(prefill.shape) == (prompt_rows,)
    # V4.1-Flash serves text only, so no position is an image token.
    assert not prefill.to_numpy().any()
    decode = _extras(processor, BLOCK + 1)["image_token_mask"]
    assert tuple(decode.shape) == (prompt_rows + BATCH * BLOCK,)
    alone = _processor(_engram_static(), verifies=False)
    assert tuple(_extras(alone, BLOCK + 1)["image_token_mask"].shape) == (
        prompt_rows,
    )


def _dp_context(monkeypatch: pytest.MonkeyPatch) -> Any:
    """What a DP4/EP4 pairing hands each build, with the checkpoint reads
    and the expert-parallel runtime stubbed out."""
    arch_config = _arch_config(n_devices=4)
    monkeypatch.setattr(
        DeepseekV41Config,
        "initialize",
        classmethod(
            lambda cls, pipeline_config, model_config=None, **kwargs: (
                dataclasses.replace(arch_config)
            )
        ),
    )
    for name, value in {
        "engram_embed_tables": lambda weights, layers: [],
        "validate_engram_tables": lambda tables, config: None,
        "serving_token_map": lambda pipeline_config, config: np.arange(
            config.vocab_size, dtype=np.int32
        ),
        "engram_static_inputs": lambda tables, token_map, device: {},
        "build_ep_config": lambda *args: Mock(),
        "EPBatchManager": lambda ep_config: Mock(),
    }.items():
        monkeypatch.setattr(model_module, name, value)
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            speculative=SpeculativeConfig(
                speculative_method="dspark", num_speculative_tokens=BLOCK
            ),
            model=SimpleNamespace(
                data_parallel_degree=4,
                device_specs=[None] * 4,
                huggingface_config=_hf_dict(),
            ),
            runtime=SimpleNamespace(ep_size=4, enable_chunked_prefill=True),
        ),
        max_seq_len=64,
        target_layer_ids=tuple(arch_config.dspark_target_layer_ids),
        kv_params=None,
        weights={},
        start_ep=lambda ep_config: None,
    )


def test_the_trunk_builds_on_a_data_parallel_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DP4/EP4 serving shape builds the trunk for the pairing.

    The pairing asks for all-row logits, which the trunk refuses under
    data parallelism for its own ``forward``. The fused graph never runs
    that forward -- the verify gathers its rows itself -- so the trunk
    keeps the next-token mode, and its config carries the extras the
    graph takes, the mask at the merged width.
    """
    context = _dp_context(monkeypatch)
    module, config = DeepseekV41Model.build_module(context)
    assert config.return_logits == ReturnLogits.LAST_TOKEN
    assert module.dp == 4
    assert module.mesh.num_devices == 4
    assert [name for name, _ in config.dspark_trailing_inputs] == (
        extra_input_names(V41Config.from_hf(_hf_dict()))
    )
    mask = dict(config.dspark_trailing_inputs)["image_token_mask"]
    assert [str(d) for d in mask.shape] == ["merged_seq_len"]


def test_the_trunk_refuses_other_tap_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stages read their trained taps, never another layer set."""
    context = _dp_context(monkeypatch)
    context.target_layer_ids = (0,)
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        DeepseekV41Model.build_module(context)


def test_the_stages_build_on_a_data_parallel_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every one-device replica holds the whole stage stack, and the
    config the pairing reads names the trunk's taps and the block's fill
    token."""
    context = _dp_context(monkeypatch)
    draft, config = DSparkDeepseekV41ModuleV3DraftArchConfig.build_module(
        context
    )
    assert draft.data_parallel_degree == 4
    assert draft.mesh.num_devices == 4
    assert config.target_layer_ids == config.dspark_target_layer_ids
    assert config.mask_token_id == config.dspark_noise_token_id
