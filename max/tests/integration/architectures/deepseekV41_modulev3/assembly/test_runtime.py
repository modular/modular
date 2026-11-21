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
"""Checks the registered runtime, its weight tree and load lifecycle."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from max import functional as F
from max import tree
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.legacy.graph_groups import BatchContext, StaticBuffers
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.deepseekV41_modulev3 import (
    model as dsv41_model,
)
from max.pipelines.architectures.deepseekV41_modulev3.arch import (
    deepseekV41_modulev3_arch,
)
from max.pipelines.architectures.deepseekV41_modulev3.batch_processor import (
    DeepseekV41BatchProcessor,
)
from max.pipelines.architectures.deepseekV41_modulev3.deepseekV41 import (
    DeepseekV41,
)
from max.pipelines.architectures.deepseekV41_modulev3.dspark_drafter import (
    DSparkDrafter,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
    extra_input_names,
    extra_inputs,
    has_vision_tower,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.mhc import (
    mhc_fused_post_pre,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
)
from max.pipelines.architectures.deepseekV41_modulev3.model import (
    DeepseekV41Model,
    build_ep_config,
    serving_token_map,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
    vision_config,
)
from max.pipelines.architectures.deepseekV41_modulev3.tokenizer import (
    DeepseekV41Tokenizer,
)
from max.pipelines.architectures.deepseekV41_modulev3.weight_adapters import (
    AdapterResult,
    EngramTable,
)
from max.pipelines.context import (
    TextAndVisionContext,
    TextContext,
    TokenBuffer,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib import (
    GraphRole,
    KVCacheConfig,
    ModuleV3PipelineModelWithKVCache,
)
from max.pipelines.lib.config.config import _required_argument_changes
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _driver_call,
    _SpecDecodeModule,
)
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.lib.tokenizer import TextTokenizer
from max.pipelines.modeling.types import InputModality
from max.pipelines.speculative.config import SpeculativeConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from runtime_config import text_config
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast


def test_registered_runtime_takes_text_and_images() -> None:
    assert deepseekV41_modulev3_arch.tokenizer is DeepseekV41Tokenizer
    # The text tokenizer carries the --tokenizer-impl chat encoder.
    assert issubclass(DeepseekV41Tokenizer, TextTokenizer)
    assert deepseekV41_modulev3_arch.context_type is TextAndVisionContext
    assert deepseekV41_modulev3_arch.input_modalities == {
        InputModality.TEXT,
        InputModality.IMAGE,
    }
    assert set(deepseekV41_modulev3_arch.modules) == {
        GraphRole.DECODER,
        GraphRole.VISION,
    }
    assert deepseekV41_modulev3_arch.supports_device_graph_capture
    assert not deepseekV41_modulev3_arch.supports_empty_batches
    # The data/expert-parallel shapes are wired, so a multi-GPU launch
    # passes config validation and serves DP attention + EP MoE.
    assert deepseekV41_modulev3_arch.multi_gpu_supported


def test_real_module_parameter_tree() -> None:
    text = text_config()
    interface = V41Config.from_hf({"text_config": vars(text)})
    kv = build_kv_params(
        interface, KVCacheConfig(kv_cache_page_size=128), [DeviceRef.CPU()]
    )
    config = DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )
    with F.lazy():
        module = DeepseekV41(
            config,
            kv,
            DeviceMesh((CPU(),), (1,), ("tp",)),
            text_config=text,
            token_map=np.arange(128, dtype=np.int32),
        )
    params = dict(module.parameters)
    # ``wo_a`` keeps the checkpoint's own pair, like every dense FP8
    # projection: E4M3 codes plus compact E8M0 block scales, never a
    # bf16 expansion.
    assert params["layers.0.attn.wo_a.weight"].dtype == DType.float8_e4m3fn
    assert params["layers.0.attn.wo_a.scale"].dtype == DType.float8_e8m0fnu
    assert [int(d) for d in params["layers.0.attn.wo_a.scale"].shape] == [
        text.o_groups * text.o_lora_rank // 32,
        text.num_attention_heads * text.head_dim // text.o_groups // 32,
    ]
    for projection in ("w1", "w3", "w2"):
        prefix = f"layers.0.ffn.experts.0.{projection}"
        for leaf in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            assert f"{prefix}.{leaf}" in params
    assert "layers.0.ffn.routed.experts.0.w1.weight" not in params
    # The token map is a graph input the model binds once and the decoder's
    # row count rides its input's extent; this fixture carries no Engram
    # layers, so no table extras.
    assert set(extra_input_names(interface)) == {
        "swa_begin",
        "global_write_floor",
        "image_token_mask",
        "engram_lookback",
        "engram_lookback_dead",
        "engram_token_map",
        "decoder_rows",
    }


def test_load_decoder_binds_static_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(DeepseekV41Model)
    decoder = Mock()
    bind = Mock()
    monkeypatch.setattr(
        ModuleV3PipelineModelWithKVCache,
        "_load_decoder",
        lambda self, session: decoder,
    )
    monkeypatch.setattr(model, "_bind_engram_static_inputs", bind)
    assert model._load_decoder(Mock()) is decoder
    bind.assert_called_once_with()


def test_adapter_result_is_a_weights_mapping() -> None:
    result = AdapterResult({}, [])
    assert isinstance(result, dict)
    assert result.state_dict is result


def test_bind_engram_static_inputs_pins_against_the_device_and_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tables pin against the pipeline's device, then the views drop.

    Page-locked staging needs the accelerator (a ``CPU()`` staging buffer
    is pageable, which the UVA gather must not read), and once the tables
    live in pinned memory the adapter's checkpoint views must not be held
    for the model's lifetime -- at production size that is a second ~98 GB
    mapping per layer.
    """
    model = object.__new__(DeepseekV41Model)
    weight = np.zeros((3, 256), np.uint8)
    scale = np.zeros((3, 8), np.uint8)
    model._engram_tables = [
        EngramTable(layer=1, rows=3, weight_codes=weight, scale_codes=scale)
    ]
    pinned: dict[str, tuple[object, ...]] = {}
    stubs = [Mock(), Mock()]

    def fake_pinned(
        device: object,
        name: str,
        rows: int,
        width: int,
        codes: object,
        dtype: object,
    ) -> object:
        pinned[name] = (device, rows, width, codes, dtype)
        return stubs[len(pinned) - 1]

    monkeypatch.setattr(
        "max.pipelines.architectures.deepseekV41_modulev3.model._pinned_table",
        fake_pinned,
    )
    device = CPU()
    monkeypatch.setattr(model, "devices", [device], raising=False)
    model._token_map = np.arange(7, dtype=np.int32)
    processor = Mock(spec=DeepseekV41BatchProcessor)
    monkeypatch.setattr(model, "_batch_processor", processor, raising=False)
    model._bind_engram_static_inputs()
    assert pinned == {
        "engram_table_0_weight": (
            device,
            3,
            256,
            weight,
            DType.float8_e4m3fn,
        ),
        "engram_table_0_scale": (
            device,
            3,
            8,
            scale,
            DType.float8_e8m0fnu,
        ),
    }
    bound = processor.bind_engram.call_args[0][0]
    assert bound["engram_table_0_weight"] is stubs[0]
    assert bound["engram_table_0_scale"] is stubs[1]
    # The token map rides the first device as the ABI's own input.
    assert bound["engram_token_map"].dtype == DType.int32
    assert bound["engram_token_map"].shape == (7,)
    assert model._engram_tables == []


def test_serving_token_map_uses_the_configured_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The map is read at the configured model revision, never a pinned one.

    Serving from a hub repo id must work: the pipeline resolves the
    checkpoint and its tokenizer files at ``huggingface_model_revision``,
    and the map loader reads that same revision instead of a fixed
    snapshot that only exists if something else pre-cached it.
    """
    delegate = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3, "!": 4},
                unk_token="<unk>",
            )
        ),
        unk_token="<unk>",
    )
    delegate.save_pretrained(tmp_path)
    model = object.__new__(DeepseekV41Model)
    monkeypatch.setattr(
        model,
        "pipeline_config",
        SimpleNamespace(
            model=SimpleNamespace(
                model_path=str(tmp_path),
                huggingface_model_revision="recorded-model-revision",
                trust_remote_code=False,
            )
        ),
        raising=False,
    )
    config = object.__new__(DeepseekV41Config)
    config.vocab_size = len(delegate)
    recorder = Mock(wraps=AutoTokenizer.from_pretrained)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", recorder)
    # The five-token stand-in cannot reproduce the real model's recorded
    # digest, so the loader stops at the digest check -- after resolving
    # the tokenizer the way serving does.
    with pytest.raises(ValueError, match="sha256"):
        serving_token_map(model.pipeline_config, config)
    recorder.assert_called_once_with(
        str(tmp_path),
        revision="recorded-model-revision",
        trust_remote_code=False,
    )


def test_check_engram_host_ram_bounds_the_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each pin must fit what is free, bounded by a fraction of total RAM."""
    gib = 1 << 30
    monkeypatch.setattr(
        "max.pipelines.architectures.deepseekV41_modulev3.model"
        ".host_memory_limit",
        lambda: 100 * gib,
    )
    monkeypatch.setattr(
        "max.pipelines.architectures.deepseekV41_modulev3.model"
        ".available_host_memory",
        lambda: 90 * gib,
    )
    dsv41_model._check_engram_host_ram("engram_table_0_weight", 72 * gib)
    # Over what is free right now, though within 80% of the total.
    with pytest.raises(RuntimeError, match="engram_table_0_weight"):
        dsv41_model._check_engram_host_ram("engram_table_0_weight", 91 * gib)
    # Over the total-RAM bound even with plenty free.
    monkeypatch.setattr(
        "max.pipelines.architectures.deepseekV41_modulev3.model"
        ".available_host_memory",
        lambda: 200 * gib,
    )
    with pytest.raises(RuntimeError, match="budget"):
        dsv41_model._check_engram_host_ram("engram_table_0_weight", 81 * gib)


def test_check_engram_host_ram_skips_when_capacity_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No readable limit and no readable headroom: warn, do not block."""
    monkeypatch.setattr(
        "max.pipelines.architectures.deepseekV41_modulev3.model"
        ".host_memory_limit",
        lambda: None,
    )
    monkeypatch.setattr(
        "max.pipelines.architectures.deepseekV41_modulev3.model"
        ".available_host_memory",
        lambda: None,
    )
    dsv41_model._check_engram_host_ram("engram_table_0_weight", 1 << 50)


@pytest.mark.parametrize(
    "mode", [ReturnLogits.LAST_TOKEN, ReturnLogits.ALL, ReturnLogits.VARIABLE]
)
def test_output_modes_use_ragged_offsets(
    mode: ReturnLogits, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = object.__new__(DeepseekV41)
    config = object.__new__(DeepseekV41Config)
    config.return_logits = mode
    config.return_hidden_states = ReturnHiddenStates.NONE
    model.config = config
    model.dp = 1
    monkeypatch.setattr(model, "head", lambda x: x, raising=False)
    monkeypatch.setattr(model, "norm", lambda x: x, raising=False)
    hidden = Tensor.from_dlpack(np.arange(20, dtype=np.float32).reshape(5, 4))
    offsets = Tensor.from_dlpack(np.array([0, 2, 5], np.uint32))
    count = Tensor.from_dlpack(np.array([2], np.int64))
    output = model._outputs(hidden, offsets, count)
    np.testing.assert_array_equal(
        output.next_token_logits.to_numpy(), hidden.to_numpy()[[1, 4]]
    )
    if mode == ReturnLogits.ALL:
        assert output.logits is not None and output.logit_offsets is not None
        np.testing.assert_array_equal(
            output.logits.to_numpy(), hidden.to_numpy()
        )
        np.testing.assert_array_equal(
            output.logit_offsets.to_numpy(), [0, 2, 5]
        )
    elif mode == ReturnLogits.VARIABLE:
        assert output.logits is not None and output.logit_offsets is not None
        np.testing.assert_array_equal(
            output.logits.to_numpy(), hidden.to_numpy()[[0, 1, 3, 4]]
        )
        np.testing.assert_array_equal(
            output.logit_offsets.to_numpy(), [0, 2, 4]
        )
    else:
        assert output.logits is None


def test_last_hidden_states_are_pre_final_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(DeepseekV41)
    config = object.__new__(DeepseekV41Config)
    config.return_logits = ReturnLogits.LAST_TOKEN
    config.return_hidden_states = ReturnHiddenStates.LAST
    model.config = config
    model.dp = 1
    monkeypatch.setattr(model, "head", lambda x: x, raising=False)
    monkeypatch.setattr(model, "norm", lambda x: 2 * x, raising=False)
    hidden = Tensor.from_dlpack(np.arange(20, dtype=np.float32).reshape(5, 4))
    offsets = Tensor.from_dlpack(np.array([0, 2, 5], np.uint32))
    count = Tensor.from_dlpack(np.array([1], np.int64))
    output = model._outputs(hidden, offsets, count)
    assert output.hidden_states is not None
    raw = hidden.to_numpy()[[1, 4]]
    # LAST returns the pre-final-norm rows, not the normalized ones.
    np.testing.assert_array_equal(output.hidden_states.to_numpy(), raw)
    # The head still consumes the normalized rows.
    np.testing.assert_array_equal(output.next_token_logits.to_numpy(), 2 * raw)


def test_config_selects_modulev3_without_new_shim(tmp_path: Path) -> None:
    register_all_models()
    text = vars(text_config())
    text["rope_scaling"] = None
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "architectures": ["DeepseekV41ForCausalLM"],
                "text_config": text,
            }
        )
    )
    hf = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    architecture = PIPELINE_REGISTRY.retrieve_architecture(
        hf.architectures[0], prefer_module_v3=True
    )
    assert architecture is deepseekV41_modulev3_arch
    assert DeepseekV41Config.get_text_config(hf).num_hidden_layers == 2


@pytest.mark.parametrize("with_vision", [False, True])
def test_the_config_shim_marks_a_vision_tower_only_when_present(
    tmp_path: Path, with_vision: bool
) -> None:
    """The shim gives a config.json without ``vision_config`` an empty one.

    That checkpoint serves text: no encoder, no vision cache, no
    ``gate.bias_vl`` leaves.
    """
    register_all_models()
    text = vars(text_config())
    text["rope_scaling"] = None
    config: dict[str, Any] = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": text,
    }
    if with_vision:
        config["vision_config"] = {
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "patch_size": 14,
            "rope_theta": 10000,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
        }
    (tmp_path / "config.json").write_text(json.dumps(config))
    hf = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert hf.vision_config is not None
    assert has_vision_tower(hf) is with_vision
    assert V41Config.from_hf(hf.to_dict()).has_vision is with_vision
    assert (DeepseekV41Config.estimate_vision_cache_entry_bytes(hf) > 0) is (
        with_vision
    )
    assert (DeepseekV41Config.get_vision_cache_row_spec(hf) is not None) is (
        with_vision
    )
    # A DSpark pairing loads the encoder too, so it fills the cache the
    # hooks size.
    for speculative in (None, SimpleNamespace(speculative_method="dspark")):
        pipeline = SimpleNamespace(
            model=SimpleNamespace(
                huggingface_config=hf, vision_config_overrides={}
            ),
            speculative=speculative,
        )
        assert (vision_config(cast(Any, pipeline)) is not None) is with_vision


def test_tokenizer_rejects_video_before_context_creation() -> None:
    tokenizer = object.__new__(DeepseekV41Tokenizer)
    request = Mock()
    request.videos = [b"video"]
    with pytest.raises(InputError, match="video"):
        asyncio.run(tokenizer.new_context(request))


@pytest.mark.parametrize(
    "all_roles", [False, True], ids=["swa_ratio1", "all_layer_roles"]
)
def test_real_trunk_traces_symbolic_rows_and_batch(
    monkeypatch: pytest.MonkeyPatch, all_roles: bool
) -> None:
    # Shape tracing targets SM100 while running on CPU; no kernel executes.
    monkeypatch.setattr("max.nn._graph_ops._is_sm10x_gpu", lambda: True)
    text = text_config()
    if all_roles:
        text.num_hidden_layers = 12
        text.compress_ratios = [0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        text.kv_source_layer_ids = [2, 5, 7]
        text.index_source_layer_ids = [2, 5, 7, 9, 11]
        text.candidate_source_layer_id = 7
    interface = V41Config.from_hf({"text_config": vars(text)})
    kv = build_kv_params(
        interface, KVCacheConfig(kv_cache_page_size=128), [DeviceRef.CPU()]
    )
    config = DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )
    with F.lazy():
        module = DeepseekV41(
            config,
            kv,
            DeviceMesh((CPU(),), (1,), ("tp",)),
            text_config=text,
            token_map=np.arange(128, dtype=np.int32),
        )
    input_types = list(
        modulev3_ragged_kv_symbolic_inputs(
            kv_params=kv, device_refs=[DeviceRef.CPU()]
        )
    )
    input_types.extend(
        kind for _, kind in extra_inputs(interface, [DeviceRef.CPU()])
    )
    module.trace(*input_types)


def _five_token_tokenizer(
    tmp_path: Path, huggingface_config: SimpleNamespace
) -> DeepseekV41Tokenizer:
    """A tokenizer whose delegate has five ids, built as the pipeline builds
    it."""
    delegate = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {
                    "<unk>": 0,
                    "<eos>": 1,
                    "<think>": 2,
                    "</think>": 3,
                    "hello": 4,
                },
                unk_token="<unk>",
            )
        ),
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<think>", "</think>"],
    )
    delegate.save_pretrained(tmp_path)
    pipeline = Mock()
    pipeline.tokenizer_impl = None
    pipeline.draft_model = None
    pipeline.model.huggingface_config = huggingface_config
    pipeline.model.generation_config = SimpleNamespace(eos_token_id=None)
    return DeepseekV41Tokenizer(str(tmp_path), pipeline, max_length=64)


def test_tokenizer_serves_text_without_a_vision_config(
    tmp_path: Path,
) -> None:
    tokenizer = _five_token_tokenizer(tmp_path, SimpleNamespace(eos_token_id=1))
    assert tokenizer.reasoning_start_token_id == 2
    assert tokenizer.reasoning_end_token_id == 3
    assert tokenizer.image_processor is None
    assert tokenizer.max_images_per_request == 0


def test_a_prompt_id_the_tokenizer_lacks_is_an_input_error(
    tmp_path: Path,
) -> None:
    """The embedding has rows past the tokenizer, but the Engram token map
    has an entry only for each tokenizer id, so the tokenizer's length bounds
    a prompt's ids."""
    tokenizer = _five_token_tokenizer(
        tmp_path, SimpleNamespace(eos_token_id=1, vocab_size=8)
    )
    assert len(tokenizer.delegate) == 5
    assert asyncio.run(tokenizer.encode([4])).tolist() == [4]
    with pytest.raises(InputError, match="out of range"):
        asyncio.run(tokenizer.encode([5]))


def test_registration_disables_prefix_caching() -> None:
    """Serving on defaults must not run prefix caching.

    The batch processor feeds every request a zero window start and write
    floor, so a request resumed from cached pages would read its window
    from position 0 and land compressor and indexer writes in blocks the
    cached prefix shares with other sessions. Until resuming from cached
    pages is supported, the registration forces the KV cache config's
    default (on) off.
    """
    assert deepseekV41_modulev3_arch.required_arguments == {
        "enable_prefix_caching": False
    }
    # The enforcement config construction runs: a cache config that
    # leaves prefix caching on is rebuilt with it off.
    assert _required_argument_changes(
        deepseekV41_modulev3_arch, "KVCacheConfig", KVCacheConfig()
    ) == {"enable_prefix_caching": False}


def test_trunk_gates_read_the_checkpoint_norm_eps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every mHC gate boundary uses the checkpoint's ``rms_norm_eps``.

    ``mhc_fused_post_pre`` defaults ``norm_eps`` to ``1e-20``; a
    checkpoint revision with a different ``config.json`` epsilon must move
    the gates with the norms and the Engram gate, not leave them behind.
    """
    monkeypatch.setattr("max.nn._graph_ops._is_sm10x_gpu", lambda: True)
    text = text_config()
    text.rms_norm_eps = 1e-6
    interface = V41Config.from_hf({"text_config": vars(text)})
    kv = build_kv_params(
        interface, KVCacheConfig(kv_cache_page_size=128), [DeviceRef.CPU()]
    )
    config = DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )
    boundaries = Mock(wraps=mhc_fused_post_pre)
    monkeypatch.setattr(
        "max.pipelines.architectures.deepseekV41_modulev3.layers.block"
        ".mhc_fused_post_pre",
        boundaries,
    )
    with F.lazy():
        module = DeepseekV41(
            config,
            kv,
            DeviceMesh((CPU(),), (1,), ("tp",)),
            text_config=text,
            token_map=np.arange(128, dtype=np.int32),
        )
    assert module.norm.eps == 1e-6
    assert all(
        layer.attn_norm.eps == 1e-6 and layer.ffn_norm.eps == 1e-6
        for layer in module.layers
    )
    input_types = list(
        modulev3_ragged_kv_symbolic_inputs(
            kv_params=kv, device_refs=[DeviceRef.CPU()]
        )
    )
    input_types.extend(
        kind for _, kind in extra_inputs(interface, [DeviceRef.CPU()])
    )
    module.trace(*input_types)
    # Two boundaries per layer, each carrying the checkpoint's epsilon.
    assert [call.kwargs["norm_eps"] for call in boundaries.call_args_list] == [
        1e-6
    ] * (2 * config.num_hidden_layers)


def _context(tokens: list[int], processed: int) -> TextContext:
    context = TextContext(
        max_length=1000,
        tokens=TokenBuffer(np.asarray(tokens, dtype=np.int64)),
    )
    if processed:
        context.tokens.skip_processing(processed)
    return context


def _processor(monkeypatch: pytest.MonkeyPatch) -> DeepseekV41BatchProcessor:
    processor = object.__new__(DeepseekV41BatchProcessor)
    processor._v41 = V41Config.from_hf({"text_config": vars(text_config())})
    processor._engram_static = {}
    processor._verifies = False
    processor._dp_degree = 1
    monkeypatch.setattr(
        processor, "runtime", SimpleNamespace(devices=[CPU()]), raising=False
    )
    # The extras' static tail (here only the token map; this config
    # declares no Engram tables) is the model's to bind, so this test
    # binds it directly.
    processor.bind_engram(
        {"engram_token_map": Buffer.from_numpy(np.zeros(1, np.int32))}
    )
    return processor


def test_extras_read_real_lookback_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continuations see their real predecessors, fresh prefills see -1.

    The lookback feeds the Engram hash, which would otherwise bind every
    decode and chunked-prefill row to the sequence-start sentinel. The
    shared legacy builder produces them at the ABI's int32.
    """
    processor = _processor(monkeypatch)
    extras = processor._extras(
        [
            # Chunked-prefill continuation (5 of 6 prompt tokens
            # processed), a decode step (the 9 active, 7 and 8 behind
            # it), and a fresh prefill, batched together.
            _context([10, 11, 12, 13, 14, 15], 5),
            _context([7, 8, 9], 2),
            _context([4, 5, 6], 0),
        ],
        np.array([0, 6, 9, 12]),
    )
    lookback = extras[
        extra_input_names(processor._v41).index("engram_lookback")
    ]
    got = lookback.to_numpy()
    assert got.dtype == np.int32
    np.testing.assert_array_equal(
        got,
        [
            [14, 13, 12],
            [8, 7, -1],
            [-1, -1, -1],
        ],
    )


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("history", [False, True])
def test_real_trunk_traces_tensor_parallel_virtual_sm100(
    tp: int, raw: bool, history: bool
) -> None:
    """Stages real attention/MoE sharding without compiling or executing."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(4)
    try:
        devices = tuple(Accelerator(i) for i in range(tp))
        refs = [DeviceRef.from_device(device) for device in devices]
        mesh = DeviceMesh(devices, (tp,), ("tp",))
        text = text_config(
            num_hidden_layers=3,
            compress_ratios=[0, 1, 0],
            num_attention_heads=64,
            o_groups=8,
            moe_intermediate_size=128,
            engram_lookback_history=history,
            decoder_replay=history,
        )
        interface = V41Config.from_hf({"text_config": vars(text)})
        kv = build_kv_params(
            interface, KVCacheConfig(kv_cache_page_size=128), refs
        )
        config = DeepseekV41Config(
            dtype=DType.bfloat16,
            kv_params=kv,
            devices=refs,
            max_seq_len=256,
            raw_checkpoint_weights=raw,
            **DeepseekV41Config.resolve_text_fields(text),
        )
        with F.lazy():
            module = DeepseekV41(
                config,
                kv,
                mesh,
                text_config=text,
                token_map=np.arange(128, dtype=np.int32),
            )
        attn = module.layers[0].attn
        assert (
            int(attn.wq_b.weight.local_shards[0].shape[0])
            == text.num_attention_heads * text.head_dim // tp
        )
        assert (
            int(attn.attn_sink.local_shards[0].shape[0])
            == text.num_attention_heads // tp
        )
        assert (
            int(attn.wo_a.weight.local_shards[0].shape[0])
            == text.o_groups * text.o_lora_rank // tp
        )
        assert (
            int(attn.wo_b.weight.local_shards[0].shape[1])
            == text.o_groups * text.o_lora_rank // tp
        )
        inputs = list(
            modulev3_ragged_kv_symbolic_inputs(kv_params=kv, device_refs=refs)
        )
        inputs.extend(
            kind for _, kind in extra_inputs(interface, refs, artifact=raw)
        )
        staged = str(module.trace(*inputs))
        assert "window_kv_store" in staged
    finally:
        set_virtual_device_count(0)


@pytest.mark.parametrize(
    ("devices_count", "dp"), [(2, 1), (4, 1), (4, 4), (8, 8)]
)
def test_real_fused_dspark_named_history_virtual_sm100(
    devices_count: int, dp: int
) -> None:
    """The actual fused builder stages named arena/history/replay inputs."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(8)
    try:
        register_all_models()
        devices = tuple(Accelerator(i) for i in range(devices_count))
        refs = [DeviceRef.from_device(d) for d in devices]
        mesh = DeviceMesh(
            devices, (devices_count,), ("dp" if dp > 1 else "tp",)
        )
        text = text_config(
            num_hidden_layers=3,
            compress_ratios=[0, 1, 0, 0],
            num_attention_heads=64,
            o_groups=8,
            moe_intermediate_size=128,
            n_mtp_layers=1,
            dspark_target_layer_ids=[2],
            dspark_noise_token_id=0,
            dspark_n_routed_experts=8,
            dspark_num_experts_per_tok=2,
            dspark_markov_rank=32,
            engram_lookback_history=True,
            decoder_replay=True,
        )
        interface = V41Config.from_hf(
            {"text_config": vars(text)}, artifact_export=True
        )
        kv = build_kv_params(
            interface,
            KVCacheConfig(kv_cache_page_size=128),
            refs,
            data_parallel_degree=dp,
            num_draft_tokens=5,
        )
        declarations = extra_inputs(
            interface,
            refs,
            dp,
            verify_rows=True,
            artifact=True,
        )
        config = DeepseekV41Config(
            dtype=DType.bfloat16,
            kv_params=kv,
            devices=refs,
            max_seq_len=256,
            max_batch_rows=74,
            raw_checkpoint_weights=True,
            artifact_export=True,
            data_parallel_degree=dp,
            # As the pipeline config derives it from the per-replica draft
            # head below.
            replicate_spec_head=dp > 1,
            dspark_trailing_inputs=declarations,
            **DeepseekV41Config.resolve_text_fields(text),
        )
        speculative = SpeculativeConfig.model_construct(
            speculative_method="dspark",
            num_speculative_tokens=5,
            draft_proposal="sampled",
            use_greedy_acceptance=False,
            draft_probs_layout="replica" if dp > 1 else "device0",
            draft_head_parallelism="replica" if dp > 1 else "device0",
        )
        pipeline: Any = SimpleNamespace(
            models=SimpleNamespace(
                main_architecture_name="DeepseekV41ForCausalLM"
            ),
            model=SimpleNamespace(
                data_parallel_degree=dp,
                pipeline_stage=None,
                huggingface_config=SimpleNamespace(
                    get_text_config=lambda: text
                ),
            ),
            draft_model=None,
            speculative=speculative,
            needs_bitmask_constraints=False,
            runtime=SimpleNamespace(
                prefer_module_v3=True,
                reasoning_parser=None,
                ep_size=dp,
                max_batch_size=2,
                max_batch_input_tokens=64,
            ),
        )
        ep_config = (
            build_ep_config(
                pipeline, config, devices_count, config.routed_expert_format
            )
            if dp > 1
            else None
        )
        with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
            target = DeepseekV41(
                config,
                kv,
                mesh,
                text_config=text,
                token_map=np.arange(128, dtype=np.int32),
                data_parallel_degree=dp,
                ep_config=ep_config,
            )
            draft = DSparkDrafter(
                config,
                mesh,
                rope=RopeParams.from_text_config(text),
                data_parallel_degree=dp,
            )
            fused = _SpecDecodeModule(
                target,
                draft,
                partial(
                    _driver_call,
                    pipeline,
                    MultiKVCacheParams.from_params({"target": kv}),
                    refs,
                    include_signal_buffers=True,
                    target_config=config,
                    draft_config=config,
                    hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
                    trace_owns_signal_buffers=True,
                ),
            )
        call = fused.driver_call()
        staged = fused.trace(*call.arguments)
        assert "SpecDecodeOutputs" in str(staged.out_tree)
        probabilities = [
            layout for layout in staged.out_layouts if len(layout.shape) == 3
        ]
        assert len(probabilities) == max(1, dp)
        assert all(
            int(layout.shape[1]) == 5 and int(layout.shape[2]) == 128
            for layout in probabilities
        )
        assert call.input_tree is not None
        assert len(tree.leaves(call.input_tree)) == len(call.input_types)
        fields = {
            key
            for group in tree.nodes(call.input_tree, BatchContext).values()
            for key in group
        }
        # Data parallelism adds the canonical per-replica context lengths.
        assert fields - {"length", "total_length"} == {
            "window_start",
            "write_floor",
            "image_token_mask",
            "lookback_tokens",
            "lookback_source",
            "lookback_mask",
            "window_rows",
            "replay_skip",
            "replay_row_indices",
            "replay_offsets",
            "replay_longest",
        }
        # The kept rows arrive one input per replica, none of them on the host.
        for group in tree.nodes(call.input_tree, BatchContext).values():
            for key in (
                "replay_row_indices",
                "replay_offsets",
                "replay_longest",
            ):
                if key in group:
                    assert len(group[key]) == max(1, dp), key
        # The Engram tables reach the ops as the fp8 they read, not bytes.
        constants = {
            key: spec.dtype
            for group in tree.nodes(call.input_tree, StaticBuffers).values()
            for key, spec in group.items()
        }
        assert constants == {
            f"engram_table_{i}_{part}": dtype
            for i in range(len(text.engram_num_embeddings))
            for part, dtype in (
                ("weight", DType.float8_e4m3fn),
                ("scale", DType.float8_e8m0fnu),
            )
        }
    finally:
        set_virtual_device_count(0)
