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
"""Checks Gemma4 role weights, optional vision and late cache attachment."""

import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import tree
from max._artifact.value_types import treedef_to_json
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightData
from max.nn import Module
from max.nn.legacy.graph_groups import GraphGroup, vision_outputs
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.gemma4 import Gemma3_MultiModalModel
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.gemma4.vision_model.preprocess_graph import (
    Gemma4ImagePreprocess,
)
from max.pipelines.architectures.gemma4.vision_model.vision_model import (
    Gemma4VisionModel,
)
from max.pipelines.architectures.gemma4_modulev3 import (
    gemma4_modulev3_arch,
    gemma4_unified_modulev3_arch,
    spec_adapters,
)
from max.pipelines.architectures.gemma4_modulev3.gemma4 import Gemma4TextModel
from max.pipelines.architectures.gemma4_modulev3.model_config import (
    Gemma4Config,
)
from max.pipelines.architectures.gemma4_modulev3.modules import (
    Gemma4Decoder,
    Gemma4Vision,
)
from max.pipelines.architectures.gemma4_modulev3.registered_batch_processor import (
    _Gemma4Inputs,
)
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.module_registration import (
    ModuleV3DecoderPipelineModel,
    RegisteredDecoderBuilder,
    _registered_weight_naming,
    _VisionGraph,
)
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _SpecDecodeModuleV3VisionCachePipelineModel,
)
from max.pipelines.lib.registration_runtime import role_weights
from max.pipelines.lib.role_contracts import VisionInputs
from max.pipelines.lib.weight_transform import CheckpointRole
from max.pipelines.speculative.driver import SequentialBatch
from max.tensor import Tensor


def test_raw_checkpoint_names_are_preserved_until_role_weights_are_prepared() -> (
    None
):
    value = NonCallableMock(spec=WeightData, dtype=DType.float32)
    raw = {
        "model.language_model.layers.0.input_layernorm.weight": value,
        "model.vision_tower.patch_embedder.weight": value,
        "model.embed_vision.embedding_projection.weight": value,
    }
    config = Gemma4Config.__new__(Gemma4Config)
    assert config.preserve_checkpoint_keys
    prepared = config.prepare_weights(raw)
    assert set(prepared) == {
        "decoder.language_model.layers.0.input_layernorm.weight",
        "vision.patch_embedder.weight",
        "vision.embed_vision.embedding_projection.weight",
    }
    assert all(weight is value for weight in prepared.values())
    assert "model.language_model.layers.0.input_layernorm.weight" in raw


def test_the_exported_weight_entries_name_what_each_role_binds() -> None:
    """An exported artifact names each graph's weights from the checkpoint,
    so a serving engine binds every weight the role's module declares under
    the name serving gives it. A graph skips the names it doesn't declare."""
    raw = {
        name: NonCallableMock(spec=WeightData, dtype=DType.float32)
        for name in (
            "model.language_model.layers.0.input_layernorm.weight",
            "model.vision_tower.patch_embedder.weight",
            "model.embed_vision.embedding_projection.weight",
        )
    }
    config = Gemma4Config.__new__(Gemma4Config)
    prepared = config.prepare_weights(raw)
    pipeline_model = gemma4_modulev3_arch.pipeline_model
    assert isinstance(pipeline_model, type)
    assert issubclass(pipeline_model, ModuleV3DecoderPipelineModel)
    naming = _registered_weight_naming(pipeline_model, None)
    assert naming.keys() == {GraphRole.DECODER, GraphRole.VISION}
    for role, entries in naming.items():
        (entry,) = entries
        assert entry.checkpoint == CheckpointRole.TARGET
        named = entry.transform.apply_data(raw)
        bound = role_weights(prepared, config, role)
        assert bound
        assert {name: named[name] for name in bound} == bound


def _planning_text_config(**overrides: Any) -> Gemma4TextConfig:
    """A one-layer text config; ``overrides`` replaces any of its fields."""
    fields: dict[str, Any] = dict(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        hidden_activation="gelu_tanh",
        max_position_embeddings=8,
        max_seq_len=8,
        rms_norm_eps=1e-6,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        query_pre_attn_scalar=8,
        sliding_window=8,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_local_base_freq=10000.0,
        sliding_window_pattern=-1,
        dtype=DType.bfloat16,
        devices=[DeviceRef.CPU()],
        interleaved_rope_weights=False,
        kv_params=NonCallableMock(),
        layer_types=["sliding_attention"],
    )
    fields.update(overrides)
    return Gemma4TextConfig(**fields)


def _finalize(
    planning: Gemma4TextConfig,
    monkeypatch: pytest.MonkeyPatch,
    n_devices: int = 1,
) -> Gemma4Config:
    """Finalizes a config around ``planning``; checkpoint parsing is stubbed."""
    monkeypatch.setattr(
        Gemma4ForConditionalGenerationConfig,
        "finalize",
        lambda self, huggingface_config, state_dict, return_logits: None,
    )
    config = Gemma4Config.__new__(Gemma4Config)
    config.devices = [DeviceRef.CPU()] * n_devices
    config.text_config = planning
    config.finalize(
        NonCallableMock(),
        {},
        ModelOutputConfig(
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        ),
    )
    return config


def test_finalize_completes_a_copy_of_the_text_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The text config's rope_theta raises when read, as a full copy reads it.
    planning = _planning_text_config()
    config = _finalize(planning, monkeypatch)
    assert config.text_config is not planning
    assert (
        config.text_config.return_hidden_states
        == ReturnHiddenStates.ALL_NORMALIZED
    )
    assert planning.return_hidden_states == ReturnHiddenStates.NONE


def _moe_text_config(**overrides: Any) -> Gemma4TextConfig:
    """The 26B-A4B MoE widths, with KV heads that split over 8 devices."""
    fields: dict[str, Any] = dict(
        num_key_value_heads=8,
        num_global_key_value_heads=8,
        enable_moe_block=True,
        num_experts=128,
        top_k_experts=8,
        moe_intermediate_size=704,
    )
    fields.update(overrides)
    return _planning_text_config(**fields)


_NVFP4 = QuantConfig.nvfp4(mlp_quantized_layers=(), attn_quantized_layers=())


@pytest.mark.parametrize(
    ("quant_config", "n_devices", "ok"),
    [
        (None, 8, True),
        (_NVFP4, 1, True),
        (_NVFP4, 4, True),
        # 704 / 8 = 88, not a multiple of the 16-wide NVFP4 scale block.
        (_NVFP4, 8, False),
    ],
    ids=["bf16_tp8", "nvfp4_tp1", "nvfp4_tp4", "nvfp4_tp8"],
)
def test_finalize_checks_each_device_gets_whole_expert_scale_blocks(
    quant_config: QuantConfig | None,
    n_devices: int,
    ok: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning = _moe_text_config(quant_config=quant_config)
    if ok:
        _finalize(planning, monkeypatch, n_devices)
        return
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        _finalize(planning, monkeypatch, n_devices)


def test_unified_checkpoint_skips_only_the_vision_tower() -> None:
    config: Any = SimpleNamespace(vision_config=None)
    assert not Gemma4Vision.is_enabled(config)
    config.vision_config = object()
    assert Gemma4Vision.is_enabled(config)


def _pipeline_config(checkpoint: Path) -> Any:
    """Stands in for a pipeline config of a checkpoint with the ids
    `google/gemma-4-31B-it`'s `config.json` states and a processor config
    that takes the processors' defaults."""
    (checkpoint / "processor_config.json").write_text(
        json.dumps({"image_processor": {}, "video_processor": {}})
    )
    return SimpleNamespace(
        model=SimpleNamespace(
            model_path=str(checkpoint),
            huggingface_model_revision=None,
            huggingface_config=SimpleNamespace(
                vision_config=SimpleNamespace(
                    patch_size=16, pooling_kernel_size=3
                ),
                image_token_id=258_880,
                video_token_id=258_884,
                boi_token_id=255_999,
                eoi_token_id=258_882,
            ),
        )
    )


def _registered(arch: Any) -> type[ModuleV3DecoderPipelineModel]:
    """Returns the pipeline model a registered architecture serves with."""
    pipeline_model = arch.pipeline_model
    assert isinstance(pipeline_model, type)
    assert issubclass(pipeline_model, ModuleV3DecoderPipelineModel)
    return pipeline_model


def _vision_model_config(
    monkeypatch: pytest.MonkeyPatch, vision_config: object | None
) -> Any:
    """A finalized config whose checkpoint's vision tower is float16.

    The dtype isn't the default, so a hook that reads it shows it read the
    finalized config.
    """
    config: Any = _finalize(_planning_text_config(), monkeypatch)
    config.vision_config = vision_config
    config.unquantized_dtype = DType.float16
    return config


def test_export_records_the_graph_api_vision_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`max export` reads the vision metadata and the image preprocessing
    graph off the pipeline model, for the config its VISION module is built
    with. They match the graph-API Gemma 4's, so a serving engine can't tell
    the two exports apart."""
    pipeline_config = _pipeline_config(tmp_path)
    config = _vision_model_config(monkeypatch, NonCallableMock())
    pipeline_model = _registered(gemma4_modulev3_arch)
    graph_api_config: Any = Gemma4ForConditionalGenerationConfig.__new__(
        Gemma4ForConditionalGenerationConfig
    )
    graph_api_config.vision_config = object()
    graph_api_config.unquantized_dtype = DType.float16

    spec = pipeline_model.vision_preprocess_spec_for(pipeline_config, config)
    expected = Gemma3_MultiModalModel.vision_preprocess_spec_for(
        pipeline_config, graph_api_config
    )
    assert spec is not None and expected is not None
    assert spec.name == expected.name == "gemma4_vision_preprocess"
    assert spec.input_types == expected.input_types
    assert spec.weights == {}
    assert isinstance(spec.module, Gemma4ImagePreprocess)
    assert isinstance(expected.module, Gemma4ImagePreprocess)
    assert spec.module.dtype == expected.module.dtype == DType.float16
    metadata = pipeline_model.vision_metadata_for(pipeline_config, config)
    assert metadata is not None
    assert metadata == Gemma3_MultiModalModel.vision_metadata_for(
        pipeline_config, graph_api_config
    )


def test_the_built_pipeline_model_reads_its_vision_modules_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A built pipeline model answers for the config its VISION module was
    built with, and answers nothing before it builds one."""
    pipeline_model = _registered(gemma4_modulev3_arch)
    model: Any = SimpleNamespace(
        pipeline_config=_pipeline_config(tmp_path),
        _vision_config=None,
        vision_metadata_for=pipeline_model.vision_metadata_for,
        vision_preprocess_spec_for=pipeline_model.vision_preprocess_spec_for,
    )
    assert pipeline_model.vision_metadata(model) is None
    assert pipeline_model.vision_preprocess_spec(model) is None

    model._vision_config = _vision_model_config(monkeypatch, NonCallableMock())
    assert pipeline_model.vision_metadata(model) is not None
    spec = pipeline_model.vision_preprocess_spec(model)
    assert spec is not None
    assert isinstance(spec.module, Gemma4ImagePreprocess)
    assert spec.module.dtype == DType.float16


def test_a_fused_speculative_graph_exports_its_targets_vision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fused speculative-decoding pipeline never builds its target's
    pipeline model, so it reads the target's vision metadata and image
    preprocessing off the target's class, for the config the target was
    built with. A ModuleV3 target's match what it exports alone."""
    pipeline_config = _pipeline_config(tmp_path)
    config = _vision_model_config(monkeypatch, NonCallableMock())
    fused: Any = SimpleNamespace(
        pipeline_config=pipeline_config,
        pairing=SimpleNamespace(target=gemma4_modulev3_arch),
        arch_config_as=lambda cls: SimpleNamespace(target=config),
    )
    alone = _registered(gemma4_modulev3_arch)

    fused_cls = _SpecDecodeModuleV3VisionCachePipelineModel
    metadata = fused_cls.vision_metadata(fused)
    assert metadata is not None
    assert metadata == alone.vision_metadata_for(pipeline_config, config)
    spec = fused_cls.vision_preprocess_spec(fused)
    assert spec is not None
    assert isinstance(spec.module, Gemma4ImagePreprocess)
    assert spec.module.dtype == DType.float16
    assert fused_cls.vision_preprocess_video_spec(fused) is None


def test_the_fused_vision_graph_names_its_weights_as_an_export_alone(
    tmp_path: Path,
) -> None:
    """A fused speculative-decoding pipeline names its target's vision
    weights with the entries the target's own export writes, since both
    compile the vision graph from the same module and checkpoint."""
    pipeline_model = _registered(gemma4_modulev3_arch)
    context: Any = SimpleNamespace(adapter=None)
    fused = RegisteredDecoderBuilder(pipeline_model).vision_weight_entries(
        context
    )
    alone = _registered_weight_naming(pipeline_model, None)[GraphRole.VISION]
    assert fused == list(alone)
    assert fused == list(Gemma4Config.weight_entries()[GraphRole.VISION])


def _vision_config(device: DeviceRef) -> Any:
    return SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=16),
        devices=[device],
        unquantized_dtype=DType.bfloat16,
    )


def _fields(
    groups: OrderedDict[str, object],
) -> list[tuple[str, str | None, str, tuple[TensorType, ...]]]:
    """Lists each group's fields as a serving engine reads them: by group
    kind and field name, with one type per device."""
    fields = []
    for name, group in groups.items():
        assert isinstance(group, GraphGroup)
        for field, types in group.items():
            per_device = types if isinstance(types, tuple) else (types,)
            fields.append((name, group.kind, field, per_device))
    return fields


def test_the_vision_inputs_are_named_as_the_graph_api_names_them() -> None:
    """The serving engine plans the encoder from its input tree's groups and
    fills each field from the preprocessing graph's output of that name,
    which both exports share."""
    vision: Any = object.__new__(Gemma4Vision)
    vision.config = _vision_config(DeviceRef.GPU())
    graph_api: Any = object.__new__(Gemma4VisionModel)
    graph_api.config = vision.config
    assert _fields(vision.input_types()) == _fields(graph_api.input_tree())


class _Echo(Module[..., Tensor]):
    """Takes the Gemma 4 encoder's arguments and returns its patches."""

    def forward(
        self,
        patches_flat: Tensor,
        pixel_position_ids: Tensor,
        cu_seqlens: Tensor,
        pool_gather_index: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        return patches_flat


def test_the_vision_graph_records_its_named_trees() -> None:
    """An exported vision graph takes the named tree as its one argument and
    returns the embeddings in the group the graph-API Gemma 4 returns."""
    vision: Any = object.__new__(Gemma4Vision)
    vision.config = _vision_config(DeviceRef.CPU())
    types = vision.input_types()
    staged, _ = _VisionGraph(_Echo()).stage(types)

    assert treedef_to_json(staged.in_tree) == treedef_to_json(
        tree.flatten(((types,), {}))[1]
    )
    assert treedef_to_json(staged.out_tree) == treedef_to_json(
        tree.flatten(vision_outputs([object()]))[1]
    )


def test_a_text_only_registration_exports_no_vision_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The text-only line builds no VISION module, so it records none."""
    pipeline_config = _pipeline_config(tmp_path)
    text_only = _vision_model_config(monkeypatch, None)
    for arch in (gemma4_modulev3_arch, gemma4_unified_modulev3_arch):
        pipeline_model = _registered(arch)
        assert (
            pipeline_model.vision_preprocess_spec_for(
                pipeline_config, text_only
            )
            is None
        )
        assert (
            pipeline_model.vision_metadata_for(pipeline_config, text_only)
            is None
        )


def test_late_cache_attachments_pack_only_device_zero_vision() -> None:
    buffers = [Buffer.from_numpy(np.array([i], np.int64)) for i in range(7)]
    _, structure = tree.flatten(
        OrderedDict(
            tokens="tokens",
            vision=VisionInputs(
                embeddings="embeddings", scatter_indices="indices"
            ),
        )
    )
    inputs = _Gemma4Inputs(
        tokens=buffers[0],
        input_row_offsets=buffers[1],
        return_n_logits=buffers[2],
        input_tree_def=structure,
    )
    inputs.vision_embeddings = buffers[3:5]
    inputs.vision_scatter_indices = buffers[5:7]
    assert inputs.buffers == (buffers[0], buffers[3], buffers[5])


def test_the_spec_target_verifies_the_wrapped_language_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(spec_adapters, "tensor_compat_verify", lambda f: f)
    monkeypatch.setattr(
        spec_adapters,
        "_verify",
        lambda target, batch: calls.append((target, batch)),
    )
    language_model = object.__new__(Gemma4TextModel)
    decoder: Any = object.__new__(Gemma4Decoder)
    decoder.model = SimpleNamespace(language_model=language_model)
    verify = spec_adapters.verify(
        decoder, None, ReturnHiddenStates.ALL_NORMALIZED, None
    )
    batch = NonCallableMock(spec=SequentialBatch)
    verify(batch)
    assert calls == [(language_model, batch)]
