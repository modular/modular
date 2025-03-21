# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import enum
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Sequence, cast, get_type_hints

import click
import pytest
from click.testing import CliRunner
from max.driver import Device, DeviceSpec, Tensor, load_devices
from max.dtype import DType
from max.engine import GPUProfilingMode, InferenceSession, Model
from max.entrypoints.cli import (
    get_default,
    get_field_type,
    is_flag,
    is_multiple,
    is_optional,
    pipeline_config_options,
    validate_field_type,
)
from max.graph import Graph, TensorType
from max.graph.weights import WeightsFormat
from max.pipelines import (
    PIPELINE_REGISTRY,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    PipelineTask,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
    upper_bounded_default,
)
from max.pipelines.context import InputContext
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.pipeline import KVCacheMixin
from transformers import AutoConfig


class TestEnum(str, enum.Enum):
    DEFAULT = "default"
    ALT = "alt"


@dataclass
class TestConfig:
    bool_field: bool = False
    enum_field: TestEnum = TestEnum.DEFAULT
    path_sequence_field: list[Path] = field(default_factory=list)
    device_specs_field: list[DeviceSpec] = field(
        default_factory=lambda: [DeviceSpec.cpu()]
    )
    optional_str_field: Optional[str] = None
    optional_enum_field: Optional[TestEnum] = None


@dataclass
class Output:
    default: Any
    field_type: Any
    flag: bool
    multiple: bool
    optional: bool


class DummyModelInputs(ModelInputs):
    input1: Tensor | None = None
    input2: Tensor | None = None
    input3: Tensor | None = None
    input4: Tensor | None = None

    def __init__(
        self,
        input1: Tensor | None = None,
        input2: Tensor | None = None,
        input3: Tensor | None = None,
        input4: Tensor | None = None,
        kv_cache_inputs: KVCacheInputs | None = None,
    ):
        self.input1 = input1
        self.input2 = input2
        self.input3 = input3
        self.input4 = input4
        self.kv_cache_inputs = kv_cache_inputs


class DummyPipelineModel(PipelineModel, KVCacheMixin):
    """A pipeline model with setup, input preparation and execution methods."""

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        """Runs the graph."""
        model_inputs = cast(DummyModelInputs, model_inputs)
        return ModelOutputs(next_token_logits=model_inputs.input1)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        raise NotImplementedError("calculate_max_seq_len is not implemented")

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[InputContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> DummyModelInputs:
        """Prepares the initial inputs to be passed to `.execute()`.

        The inputs and functionality of this method can vary per model.
        For example, the model inputs could include:
        - Encoded tensors
        - A unique IDs for each tensor if this model uses a KV Cache manager.

        This function would batch the encoded tensors, claim a slot in the kv
        cache if the ID hasn't been seen before, and return the inputs and
        caches as a list of tensors."""
        return DummyModelInputs(
            input1=Tensor.zeros((0, 0), DType.float32),
            input2=Tensor.zeros((0, 0), DType.float32),
            input3=Tensor.zeros((0, 0), DType.float32),
            input4=Tensor.zeros((0, 0), DType.float32),
            kv_cache_inputs=None,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> DummyModelInputs:
        """Prepares the secondary inputs to be passed to `.execute()`.

        While `prepare_initial_token_inputs` is responsible for managing the initial inputs.
        This function is responsible for updating the inputs, for each step in a multi-step execution pattern.
        """
        return DummyModelInputs(
            input1=Tensor.zeros((0, 0), DType.float32),
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    @classmethod
    def _get_num_kv_heads(cls, hf_config: Any) -> int:
        if hasattr(hf_config, "num_key_value_heads"):
            return hf_config.num_key_value_heads
        elif hasattr(hf_config, "num_attention_heads"):
            return hf_config.num_attention_heads
        elif hasattr(hf_config, "n_heads"):
            return hf_config.n_heads
        else:
            raise ValueError(
                "num_key_value_heads or num_attention_heads or n_heads not found in huggingface_config"
            )

    @classmethod
    def _get_hidden_size(cls, hf_config: Any) -> int:
        if hasattr(hf_config, "hidden_size"):
            return hf_config.hidden_size
        elif hasattr(hf_config, "d_model"):
            return hf_config.d_model
        else:
            raise ValueError(
                "hidden_size or d_model not found in huggingface_config"
            )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        if hasattr(huggingface_config, "num_hidden_layers"):
            return huggingface_config.num_hidden_layers
        elif hasattr(huggingface_config, "num_layers"):
            return huggingface_config.num_layers
        elif hasattr(huggingface_config, "n_layers"):
            return huggingface_config.n_layers
        else:
            raise ValueError(
                "num_hidden_layers or num_layers or n_layers not found in huggingface_config"
            )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        num_kv_heads = cls._get_num_kv_heads(huggingface_config)
        hidden_size = cls._get_hidden_size(huggingface_config)

        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=num_kv_heads,
            head_dim=hidden_size // num_kv_heads,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            page_size=kv_cache_config.kv_cache_page_size,
            n_devices=n_devices,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int | None,
    ) -> KVCacheManager:
        """Provided a PipelineConfig and InferenceSession, load the kv manager."""
        assert available_cache_memory is not None
        num_layers = self.get_num_layers(self.pipeline_config)
        devices = load_devices(self.pipeline_config.model_config.device_specs)

        return load_kv_manager(
            params=self.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.pipeline_config.model_config.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, self.huggingface_config
            ),
            num_layers=num_layers,
            devices=devices,
            available_cache_memory=available_cache_memory,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int | None,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        assert available_cache_memory is not None
        assert pipeline_config.max_length is not None
        num_layers = cls.get_num_layers(huggingface_config=huggingface_config)

        return estimate_kv_cache_size(
            params=cls.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=pipeline_config.max_length,
            num_layers=num_layers,
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        """Provided a PipelineConfig and InferenceSession, build and load the model graph."""
        kv_inputs = self.kv_manager.input_symbols()[0]
        with Graph(
            "dummy",
            input_types=[
                TensorType(DType.int64, shape=["batch_size"]),
                *kv_inputs,
            ],
        ) as graph:
            tokens, kv_inputs_value = graph.inputs
            graph.output(tokens)
            return session.load(graph)


class DummyLlamaPipelineModel(DummyPipelineModel):
    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for DummyModel, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e


DUMMY_ARCH = SupportedArchitecture(
    name="LlamaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["modularai/llama-3.1"],
    default_encoding=SupportedEncoding.float32,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.q6_k: [KVCacheStrategy.CONTINUOUS],
    },
    pipeline_model=DummyLlamaPipelineModel,
    tokenizer=TextTokenizer,
    multi_gpu_supported=True,
    default_weights_format=WeightsFormat.gguf,
)

PIPELINE_REGISTRY.reset()
PIPELINE_REGISTRY.register(DUMMY_ARCH)

VALID_RESULTS = {
    "bool_field": Output(
        default=False,
        field_type=bool,
        flag=True,
        multiple=False,
        optional=False,
    ),
    "enum_field": Output(
        default=TestEnum.DEFAULT,
        field_type=click.Choice([e for e in TestEnum]),
        flag=False,
        multiple=False,
        optional=False,
    ),
    "path_sequence_field": Output(
        default=[],
        field_type=click.Path(path_type=Path),
        flag=False,
        multiple=True,
        optional=False,
    ),
    "device_specs_field": Output(
        default=[DeviceSpec.cpu()],
        field_type=DeviceSpec,
        flag=False,
        multiple=True,
        optional=False,
    ),
    "optional_str_field": Output(
        default=None,
        field_type=str,
        flag=False,
        multiple=False,
        optional=True,
    ),
    "optional_enum_field": Output(
        default=None,
        field_type=click.Choice([e for e in TestEnum]),
        flag=False,
        multiple=False,
        optional=True,
    ),
}


def test_pipeline_config_cli_parsing():
    field_types = get_type_hints(PipelineConfig)
    for config_field in fields(PipelineConfig):
        if not config_field.name.startswith("_"):
            validate_field_type(field_types[config_field.name])


def test_cli__get_default():
    for config_field in fields(TestConfig):
        assert (
            get_default(config_field)
            == VALID_RESULTS[config_field.name].default
        )


def test_cli__get_field_type():
    field_types = get_type_hints(TestConfig)
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            field_type = get_field_type(field_types[config_field.name])

            # If optional, ensure the underlying value has the correct type
            expected_type = VALID_RESULTS[config_field.name].field_type
            if not isinstance(expected_type, type):
                assert type(expected_type) is type(field_type)
            else:
                assert expected_type == field_type


def test_cli__option_is_flag():
    field_types = get_type_hints(TestConfig)
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            flag = is_flag(field_types[config_field.name])
            assert flag == VALID_RESULTS[config_field.name].flag, (
                f"failed test_is_flag for {config_field.name}"
            )


def test_cli__option_is_multiple():
    field_types = get_type_hints(TestConfig)
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            multiple = is_multiple(field_types[config_field.name])
            assert multiple == VALID_RESULTS[config_field.name].multiple


def test_cli__option_is_optional():
    field_types = get_type_hints(TestConfig)
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            optional = is_optional(field_types[config_field.name])
            assert optional == VALID_RESULTS[config_field.name].optional


@click.group()
def cli():
    pass


@dataclass
class TestCommand:
    args: list[str]
    expected: dict[str, Any]
    valid: bool


# TODO(AITLIB-277): Add back test coverage for kv_cache_config. These had to be removed
# as the kv cache strategy changes within config right after the config is
# initialized, which may not be what is provided in the command line.
TEST_COMMANDS = [
    TestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--cache-strategy",
            "naive",
            "--devices",
            "gpu",
        ],
        expected={
            "model_config": {
                "trust_remote_code": False,
            },
        },
        valid=True,
    ),
    TestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--devices",
            "gpu",
            "--gpu-profiling",
            "on",
        ],
        expected={
            "profiling_config": {
                "gpu_profiling": GPUProfilingMode.ON,
            },
        },
        valid=True,
    ),
    TestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--devices",
            "gpu",
            "--gpu-profiling",
            "detailed",
        ],
        expected={
            "profiling_config": {
                "gpu_profiling": GPUProfilingMode.DETAILED,
            },
        },
        valid=True,
    ),
    TestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--devices",
            "gpu",
            "--gpu-profiling",
            "invalid",
        ],
        expected={
            "profiling_config": {
                "gpu_profiling": GPUProfilingMode.DETAILED,
            },
        },
        valid=False,
    ),
    # TODO(AITLIB-278): Add back test coverage for weight_path.
    # TestCommand(
    #     args=[
    #         "--model-path",
    #         "modularai/llama-3.1",
    #         "--weight-path",
    #         "model1.safetensors",
    #         "--weight-path",
    #         "model2.safetensors",
    #         "--weight-path",
    #         "model3.safetensors",
    #     ],
    #     expected={
    #         "model_config": {
    #             "model_path": "modularai/llama-3.1",
    #             "trust_remote_code": False,
    #             "weight_path": [
    #                 Path("model1.safetensors"),
    #                 Path("model2.safetensors"),
    #                 Path("model3.safetensors"),
    #             ],
    #         },
    #     },
    #     valid=True,
    # ),
    # TestCommand(
    #     args=[
    #         "--model-path",
    #         "modularai/llama-3.1",
    #         "--weight-path",
    #         "model1.safetensors",
    #         "--weight-path",
    #         "model2.safetensors",
    #         "--weight-path",
    #         "model3.safetensors",
    #         "--devices",
    #         "gpu:0",
    #     ],
    #     expected={
    #         "model_config": {
    #             "model_path": "modularai/llama-3.1",
    #             "trust_remote_code": False,
    #             "weight_path": [
    #                 Path("model1.safetensors"),
    #                 Path("model2.safetensors"),
    #                 Path("model3.safetensors"),
    #             ],
    #             "device_specs": [DeviceSpec.accelerator(id=0)],
    #         },
    #     },
    #     valid=True,
    # ),
]


@cli.command(name="testing")
@pipeline_config_options
@click.option(
    "--idx",
    type=int,
)
def testing(
    idx,
    **config_kwargs,
):
    # Retrieve test command.
    test_command = TEST_COMMANDS[idx]

    if test_command.valid:
        # Initialize Pipeline Config.
        pipeline_config = PipelineConfig(**config_kwargs)

        for attr_name, expected_value in test_command.expected.items():
            assert hasattr(pipeline_config, attr_name)
            test_value = getattr(pipeline_config, attr_name)

            if isinstance(expected_value, dict):
                # Recursively check nested dictionaries
                for key, value in expected_value.items():
                    assert hasattr(test_value, key)
                    nested_test_value = getattr(test_value, key)

                    if key == "kv_cache_config":
                        # Handle nested kv_cache_config
                        for kv_key, kv_value in value.items():
                            assert hasattr(nested_test_value, kv_key)
                            kv_test_value = getattr(nested_test_value, kv_key)
                            assert kv_test_value == kv_value
                    else:
                        assert nested_test_value == value
            else:
                assert test_value == expected_value

    else:
        with pytest.raises(Exception):
            # Initialize Pipeline Config.
            pipeline_config = PipelineConfig(**config_kwargs)

            for attr_name, expected_value in test_command.expected.items():
                assert hasattr(pipeline_config, attr_name)
                test_value = getattr(pipeline_config, attr_name)

                if isinstance(expected_value, dict):
                    # Recursively check nested dictionaries
                    for key, value in expected_value.items():
                        assert hasattr(test_value, key)
                        nested_test_value = getattr(test_value, key)

                        if key == "kv_cache_config":
                            # Handle nested kv_cache_config
                            for kv_key, kv_value in value.items():
                                assert hasattr(nested_test_value, kv_key)
                                kv_test_value = getattr(
                                    nested_test_value, kv_key
                                )
                                assert kv_test_value == kv_value
                        else:
                            assert nested_test_value == value
                else:
                    assert test_value == expected_value


def test_cli__terminal_commands():
    runner = CliRunner()
    for idx, command in enumerate(TEST_COMMANDS):
        command.args.extend(["--idx", str(idx)])
        print(f"full_args: {command.args}")
        result = runner.invoke(testing, command.args)

        if command.valid:
            assert result.exit_code == 0
        else:
            assert result.exit_code != 0
