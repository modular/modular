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
"""Layer substitution, and the format ``--quantization-format`` resolves to."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pytest
from max import functional as F
from max import tree
from max._entrypoints.cli.quantize import (
    _enable_error_prints,
    _graphs_without_compile,
    _ignore_pattern,
    checkpoint_quantization_config,
    export_quantized_checkpoint,
    insert_quantized_linears,
    on_the_fly_config,
    quant_formats,
)
from max.driver import accelerator_count
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn import Linear, Module
from max.nn.common_layers.linear import QKVLinear
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.moe import MoE
from max.nn.common_layers.quant_linear import (
    QuantizedLinear,
    QuantizedMLP,
    QuantizedQKVLinear,
)
from max.nn.common_layers.quant_moe import QuantizedMoE
from max.nn.common_layers.quant_tensor import (
    MXFP4Quantizer,
    MXFP8Quantizer,
    NVFP4TensorQuantizer,
)
from max.nn.quant_config import QuantFormat, WeightSource
from max.pipelines.weights.checkpoint_files import (
    ShardWriter,
    _host_tensor,
    _read_saved_tensor,
)
from max.tensor import Tensor


class _Block(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = Linear(32, 64, bias=True)
        self.k_proj = Linear(32, 64, bias=False)
        self.mlp = Linear(32, 128, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return x


class _OwnLinear(Module[[Tensor], Tensor]):
    """A projection that is not :class:`Linear`."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = Tensor.zeros((64, 32), dtype=DType.bfloat16)

    def forward(self, x: Tensor) -> Tensor:
        return x


def test_insert_replaces_a_models_own_linear(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A plain rank-2 projection becomes a QuantizedLinear, like any other."""

    class Holder(Module[[Tensor], Tensor]):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = _OwnLinear()
            self.down_proj = Linear(32, 64, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return x

    with F.lazy():
        block = Holder()
        with caplog.at_level(logging.WARNING):
            insert_quantized_linears(
                block,
                [r"gate_proj", r"down_proj"],
                on_the_fly_config("nvfp4"),
            )

        assert type(block.gate_proj) is QuantizedLinear
        assert type(block.down_proj) is QuantizedLinear
        assert isinstance(block.gate_proj.weight, NVFP4TensorQuantizer)
        assert list(block.gate_proj.weight.shape) == [64, 32]
        assert (
            "Replaced a custom _OwnLinear with a common QuantizedLinear. "
            "This should be ok for weights-only export but is likely "
            "to break on-the-fly quantization."
        ) in caplog.text


def test_insert_quantized_linears_wraps_only_matching_layers() -> None:
    """Matching linears become quant containers."""
    with F.lazy():
        block = _Block()
        insert_quantized_linears(
            block, [r"q_proj", r"k_proj"], on_the_fly_config("nvfp4")
        )

        assert type(block.q_proj) is QuantizedLinear
        assert type(block.k_proj) is QuantizedLinear
        assert type(block.mlp) is Linear
        assert isinstance(block.q_proj.weight, NVFP4TensorQuantizer)

        for layer, out_dim, has_bias in (
            (block.q_proj, 64, True),
            (block.k_proj, 64, False),
        ):
            assert layer.in_dim == 32
            assert layer.out_dim == out_dim
            assert isinstance(layer.bias, Tensor) is has_bias
            assert isinstance(layer.weight, NVFP4TensorQuantizer)
            assert layer.weight.block_size == (1, 16)
            assert list(layer.weight.shape) == [out_dim, 32]


@pytest.mark.parametrize("name", quant_formats())
def test_every_offered_format_builds_a_quantize_on_load_config(
    name: str,
) -> None:
    """``--quantization-format`` accepts exactly what it can deliver.

    The source is the point: a config this command hands out must say the
    bytes arrive in native precision, or the weight loader would look for
    packed data that the checkpoint does not hold.
    """
    config = on_the_fly_config(name)
    assert config.source is WeightSource.QUANTIZE
    assert config.quantizes_on_load
    assert config.format.value == name


@pytest.mark.parametrize(
    ("name", "quantizer_cls", "block_k"),
    [
        ("nvfp4", NVFP4TensorQuantizer, 16),
        ("mxfp8", MXFP8Quantizer, 32),
        ("mxfp4", MXFP4Quantizer, 32),
    ],
)
def test_insert_builds_the_formats_quantizer(
    name: str, quantizer_cls: type[object], block_k: int
) -> None:
    """Each offered format wraps the linear in that format's quantizer."""
    with F.lazy():
        block = _Block()
        insert_quantized_linears(block, [r"mlp"], on_the_fly_config(name))

    weight = cast(
        NVFP4TensorQuantizer | MXFP8Quantizer | MXFP4Quantizer, block.mlp.weight
    )
    assert isinstance(weight, quantizer_cls)
    assert weight.block_size == (1, block_k)


def test_export_configs_use_the_keys_the_loaders_read() -> None:
    """MXFP8 nests ``ignore`` where the modelopt parser looks for it."""
    ignored = ["model.layers.0.self_attn*"]
    mxfp8 = checkpoint_quantization_config("mxfp8", ignored)
    assert mxfp8["quant_method"] == "modelopt"
    section = mxfp8["quantization"]
    assert isinstance(section, dict)
    assert section["quant_algo"] == "MXFP8"
    assert section["ignore"] == ignored

    assert checkpoint_quantization_config("mxfp4", ignored) == {
        "quant_method": "mxfp4"
    }

    nvfp4 = checkpoint_quantization_config("nvfp4", ignored)
    assert nvfp4["quant_method"] == "modelopt"
    assert nvfp4["quant_algo"] == "NVFP4"
    assert nvfp4["ignore"] == ignored


def test_an_unknown_format_names_the_ones_that_work() -> None:
    """A typo should not surface as a KeyError from a private dict."""
    with pytest.raises(ValueError, match="unknown quantization format") as exc:
        on_the_fly_config("fp4-but-faster")
    for name in quant_formats():
        assert name in str(exc.value)


def test_insert_swaps_fused_mlp_and_qkv() -> None:
    """A full fused group becomes the quant fuser, members and all."""
    with F.lazy():
        config = on_the_fly_config("nvfp4")
        source = MLP(32, 64, activation_function="gelu_tanh")
        activation = source.activation_function
        mlp = insert_quantized_linears(
            source,
            [r"^gate_proj$", r"^up_proj$", r"^down_proj$"],
            config,
        )
        qkv = insert_quantized_linears(
            QKVLinear(32, 32, 32),
            [r"^q_proj$", r"^k_proj$", r"^v_proj$"],
            config,
        )

        assert type(mlp) is QuantizedMLP
        assert mlp.activation_function is activation
        assert type(qkv) is QuantizedQKVLinear
        for layer in (
            mlp.gate_proj,
            mlp.up_proj,
            mlp.down_proj,
            qkv.q_proj,
            qkv.k_proj,
            qkv.v_proj,
        ):
            assert isinstance(layer.weight, NVFP4TensorQuantizer)


def test_insert_leaves_down_proj_on_the_plain_mlp() -> None:
    """``down_proj`` is its own matmul, so the MLP fuser stays put."""
    with F.lazy():
        mlp = MLP(32, 64)
        insert_quantized_linears(
            mlp, [r"^down_proj$"], on_the_fly_config("nvfp4")
        )

        assert type(mlp) is MLP
        assert type(mlp.down_proj) is QuantizedLinear
        assert isinstance(mlp.down_proj.weight, NVFP4TensorQuantizer)
        assert type(mlp.gate_proj.weight) is Tensor
        assert type(mlp.up_proj.weight) is Tensor


def test_insert_verbose_prints_the_patched_module(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--verbose`` prints the module after the swap, before the count log."""
    with F.lazy():
        mlp = MLP(32, 64)
        insert_quantized_linears(
            mlp, [r"^down_proj$"], on_the_fly_config("nvfp4"), verbose=True
        )

    assert "down_proj" in capsys.readouterr().out


def test_quantizer_path_omits_the_weight_segment() -> None:
    """The packed weight is transparent, so the quantizer's path has no ``.weight``."""

    class Holder(Module[[Tensor], Tensor]):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = QuantizedLinear(
                32, 64, bias=False, quant_config=on_the_fly_config("nvfp4")
            )

        def forward(self, x: Tensor) -> Tensor:
            return x

    with F.lazy():
        module = Holder()
    assert list(tree.nodes(module, NVFP4TensorQuantizer)) == ["down_proj"]


def _moe() -> MoE:
    return MoE(
        hidden_dim=32,
        num_experts=2,
        num_experts_per_token=1,
        moe_dim=64,
        has_shared_experts=True,
        shared_experts_dim=64,
    )


class _MoeBlock(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _moe()

    def forward(self, x: Tensor) -> Tensor:
        return x


def test_insert_swaps_a_moe_whose_experts_all_match() -> None:
    """Every routed expert weight is wrapped, and the MoE becomes quantized."""
    with F.lazy():
        block = _MoeBlock()
        insert_quantized_linears(
            block,
            [r"^mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
            on_the_fly_config("nvfp4"),
        )

        assert type(block.mlp) is QuantizedMoE
        expert = block.mlp.experts[0]
        assert isinstance(expert.gate_proj.weight, NVFP4TensorQuantizer)
        assert isinstance(expert.down_proj.weight, NVFP4TensorQuantizer)
        # The shared expert was not targeted, so its weight stays plain.
        assert type(block.mlp.shared_experts.gate_proj.weight) is Tensor


def test_insert_leaves_a_moe_when_only_its_name_matches() -> None:
    """``mlp.gate_proj`` is not an expert projection, so the MoE stays put."""
    with F.lazy():
        block = _MoeBlock()
        insert_quantized_linears(
            block,
            [r"^mlp\.(gate_proj|up_proj)$"],
            on_the_fly_config("nvfp4"),
        )

        assert type(block.mlp) is MoE
        assert all(type(expert) is MLP for expert in block.mlp.experts)


def test_insert_rejects_quantizing_some_experts() -> None:
    """A QuantizedMoE stacks every expert, so a subset cannot be swapped."""
    with F.lazy():
        block = _MoeBlock()
        with pytest.raises(ValueError, match="all experts"):
            insert_quantized_linears(
                block,
                [r"^mlp\.experts\.0\.(gate_proj|up_proj)$"],
                on_the_fly_config("nvfp4"),
            )
        assert type(block.mlp) is MoE


def test_insert_rejects_half_of_a_fused_group() -> None:
    """One side of a fused matmul cannot pick a global scale by itself."""
    with F.lazy():
        mlp = MLP(32, 64)
        qkv = QKVLinear(32, 32, 32)
        config = on_the_fly_config("nvfp4")
        with pytest.raises(ValueError, match="one matmul"):
            insert_quantized_linears(mlp, [r"^gate_proj$"], config)
        with pytest.raises(ValueError, match="one matmul"):
            insert_quantized_linears(qkv, [r"^q_proj$", r"^k_proj$"], config)
        assert type(mlp) is MLP
        assert type(qkv) is QKVLinear


def _reports_error(layer: Any) -> bool:
    """Whether ``layer``'s weight is a quantizer that reports its error."""
    assert isinstance(layer.weight, NVFP4TensorQuantizer)
    return layer.weight.reports_error


def _sweeps(layer: Any) -> bool:
    """Whether ``layer``'s weight is an NVFP4 quantizer that sweeps the clip."""
    assert isinstance(layer.weight, NVFP4TensorQuantizer)
    return layer.weight._mse_sweep_scales


def test_report_error_builds_reporting_quantizers() -> None:
    """With ``report_error`` every inserted quantizer reports; others do not."""
    with F.lazy():
        block = _Block()
        insert_quantized_linears(
            block,
            [r"q_proj", r"k_proj"],
            on_the_fly_config("nvfp4", report_error=True),
        )

        assert _reports_error(block.q_proj)
        assert _reports_error(block.k_proj)
        assert isinstance(block.mlp, Linear)


def test_mse_sweep_scales_reaches_the_quantizer() -> None:
    """The CLI flag is the quantizer's sweep, and a rebuild keeps it."""
    with F.lazy():
        default_block = _Block()
        insert_quantized_linears(
            default_block, [r"q_proj"], on_the_fly_config("nvfp4")
        )
        on_block = _Block()
        insert_quantized_linears(
            on_block,
            [r"q_proj"],
            on_the_fly_config("nvfp4", mse_sweep_scales=True),
        )
        rebuilt = tree.map(lambda tensor: tensor, on_block, leaf=Tensor)

        assert not _sweeps(default_block.q_proj)
        assert _sweeps(on_block.q_proj)
        assert _sweeps(rebuilt.q_proj)


def test_error_prints_drop_stale_dumps(tmp_path: Path) -> None:
    """Arming the print directory clears the previous run's dumps."""
    stale = tmp_path / "quant_error.old.weight.max"
    stale.write_bytes(b"stale")
    calls: dict[str, Any] = {}

    class _Session:
        def set_debug_print_options(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    session = cast(Any, _Session())
    _enable_error_prints(session, tmp_path)

    assert not stale.exists()
    assert calls["output_directory"] == tmp_path
    _enable_error_prints(session, None)
    assert calls["output_directory"] == tmp_path


def test_report_error_is_off_by_default() -> None:
    with F.lazy():
        block = _Block()
        insert_quantized_linears(block, [r"q_proj"], on_the_fly_config("nvfp4"))
        assert not _reports_error(block.q_proj)


def test_report_error_covers_the_members_of_fused_groups() -> None:
    """Gate and up are one matmul but each member still reports."""
    with F.lazy():
        mlp = insert_quantized_linears(
            MLP(32, 64),
            [r"^gate_proj$", r"^up_proj$", r"^down_proj$"],
            on_the_fly_config("nvfp4", report_error=True),
        )

        assert type(mlp) is QuantizedMLP
        assert _reports_error(mlp.gate_proj)
        assert _reports_error(mlp.up_proj)
        assert _reports_error(mlp.down_proj)


def test_report_error_survives_the_compile_rebuild() -> None:
    """``compile`` rebuilds the module by tree-mapping it; the flag rides along."""
    with F.lazy():
        block = _Block()
        insert_quantized_linears(
            block, [r"q_proj"], on_the_fly_config("nvfp4", report_error=True)
        )
        rebuilt = tree.map(lambda tensor: tensor, block, leaf=Tensor)

        assert _reports_error(rebuilt.q_proj)


def test_report_error_rejects_a_format_that_cannot_measure() -> None:
    """A format without a measuring hook fails before compile, by name."""
    config = on_the_fly_config("nvfp4", report_error=True)
    config.format = QuantFormat.BLOCKSCALED_FP8
    with F.lazy():
        block = _Block()
        with pytest.raises(ValueError, match="only implemented for"):
            insert_quantized_linears(block, [r"q_proj"], config)


def test_report_error_reaches_layers_built_from_the_config() -> None:
    """A layer built straight from ``quant_config`` reports, with no wiring."""
    config = on_the_fly_config("nvfp4", report_error=True)
    with F.lazy():
        mlp = QuantizedMLP(32, 64, quant_config=config)

        assert _reports_error(mlp.gate_proj)
        assert _reports_error(mlp.up_proj)
        assert _reports_error(mlp.down_proj)


def test_report_error_needs_a_weight_quantized_on_load() -> None:
    config = on_the_fly_config("nvfp4", report_error=True)
    config.source = WeightSource.CHECKPOINT
    with F.lazy(), pytest.raises(ValueError, match="quantized on load"):
        QuantizedLinear(32, 64, quant_config=config)


def test_ignore_pattern_keeps_a_quantized_sibling() -> None:
    """A copied router does not mark the experts beside it unquantized."""
    stems = ("model.layers.3.mlp.experts.0.gate_proj",)
    assert (
        _ignore_pattern("model.layers.3.mlp.gate.weight", stems)
        == "model.layers.3.mlp.gate*"
    )
    assert (
        _ignore_pattern("model.layers.3.self_attn.q_a_proj.weight", stems)
        == "model.layers.3.self_attn*"
    )
    assert _ignore_pattern("model.embed_tokens.weight", stems) is None


def test_shard_writer_flushes_a_full_shard_before_the_next(
    tmp_path: Path,
) -> None:
    """A shard is a real file once it fills, and close adds the of-count."""
    dest = tmp_path / "out"
    writer = ShardWriter(dest, shard_bytes=8)
    blobs = (
        ("a", (DType.uint8, [6], b"aaaaaa")),
        ("b", (DType.uint8, [6], b"bbbbbb")),
        ("c", (DType.uint8, [6], b"cccccc")),
    )
    for index, (name, blob) in enumerate(blobs, start=1):
        writer.add(name, blob)
        assert len(list(dest.glob("model-?????.safetensors"))) == index - 1
    writer.close()
    assert sorted(path.name for path in dest.glob("*.safetensors")) == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    assert list(dest.glob("model-?????.safetensors")) == []


def test_export_keeps_weights_from_every_graph() -> None:
    """The export mapping is the decoder. Other graphs are still built."""
    vision = object()
    decoder = object()

    class _Spec:
        def __init__(self, module: object, weights: dict[str, object]) -> None:
            self.module = module
            self.weights = weights

    class _Model:
        def _load_state_dict(self) -> dict[str, Any]:
            return {}

        def _create_model_config(self, state_dict: dict[str, Any]) -> object:
            del state_dict
            return object()

        def graph_specs(self, model_config: object) -> dict[str, _Spec]:
            del model_config
            return {
                "vision": _Spec(
                    "vision-module",
                    {"patch_embedder.input_proj.weight": vision},
                ),
                "decoder": _Spec(
                    "decoder-module",
                    {"language_model.embed_tokens.weight": decoder},
                ),
            }

    model = _Model()
    assert _graphs_without_compile(model, cast(Any, None)) == {}
    assert model._export_state_dict == {  # type: ignore[attr-defined]
        "language_model.embed_tokens.weight": decoder,
    }
    assert model.compiled_modules == {  # type: ignore[attr-defined]
        "vision": "vision-module",
        "decoder": "decoder-module",
    }


@pytest.mark.skipif(accelerator_count() == 0, reason="requires a GPU")
def test_export_quantizes_one_weight_and_copies_the_rest(
    tmp_path: Path,
) -> None:
    """One quantizer is packed. The untouched weights are copied beside it."""
    import json

    class Holder(Module[[Tensor], Tensor]):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = QuantizedLinear(
                32, 64, bias=True, quant_config=on_the_fly_config("nvfp4")
            )
            self.norm = Linear(32, 32, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return x

    def entry(tensor: Tensor, name: str) -> WeightData:
        return WeightData(
            tensor.driver_tensor, name, tensor.dtype, tensor.shape
        )

    with F.lazy():
        module = Holder()
    weight = Tensor.zeros((64, 32), dtype=DType.bfloat16)
    bias = Tensor.zeros((64,), dtype=DType.bfloat16)
    norm = Tensor.ones((32,), dtype=DType.bfloat16)
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "test"}))
    (src / "tokenizer_config.json").write_text("{}")
    source_raw = b"\x00\x00\x80\x3f"
    source_name = "vision_tower.vision_model.embeddings.patch_embedding.weight"
    source_writer = ShardWriter(src)
    source_writer.add(
        "model.layers.0.mlp.down_proj.weight", _host_tensor(weight)
    )
    source_writer.add("model.layers.0.mlp.down_proj.bias", _host_tensor(bias))
    source_writer.add("model.norm.weight", _host_tensor(norm))
    source_writer.add(source_name, (DType.float32, [1], source_raw))
    source_writer.close()

    class _Paths:
        def resolved_weight_paths(self) -> list[Path]:
            return [src / "model.safetensors"]

    class _Pipeline:
        model = _Paths()

    class _Host:
        nn_module = module
        final_state_dict = {
            "down_proj.weight": entry(
                weight, "model.layers.0.mlp.down_proj.weight"
            ),
            "down_proj.bias": entry(bias, "model.layers.0.mlp.down_proj.bias"),
            "norm.weight": entry(norm, "model.norm.weight"),
        }
        pipeline_config = _Pipeline()
        weights = None

    dest = tmp_path / "out"
    export_quantized_checkpoint(_Host(), dest, "nvfp4")

    index = json.loads((dest / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"])
    assert "model.layers.0.mlp.down_proj.weight" in names
    assert "model.layers.0.mlp.down_proj.weight_scale" in names
    assert "model.layers.0.mlp.down_proj.weight_scale_2" in names
    assert "model.layers.0.mlp.down_proj.input_scale" in names
    assert "model.layers.0.mlp.down_proj.bias" in names
    assert "model.norm.weight" in names
    assert source_name in names
    assert "vision_model.embeddings.patch_embedding.weight" not in names
    restored = _read_saved_tensor(
        dest / index["weight_map"][source_name], source_name
    )
    assert restored[2] == source_raw
    config = json.loads((dest / "config.json").read_text())
    assert config["quantization_config"]["quant_algo"] == "NVFP4"
    assert "model.layers.0.mlp*" not in config["quantization_config"]["ignore"]
    assert (dest / "tokenizer_config.json").is_file()
