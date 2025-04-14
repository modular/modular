# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from max.driver import DeviceSpec
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
)
from test_common.pipeline_config import (
    mock_pipeline_config_hf_dependencies,
)
from test_common.pipeline_model_dummy import (
    DUMMY_ARCH,
    prepare_registry,
)


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_config__validates_supported_device():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # Valid device/encoding combinations.
    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=SupportedEncoding.float32,
        max_length=1,
    )

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device_specs=[DeviceSpec.accelerator()],
        quantization_encoding=SupportedEncoding.bfloat16,
        max_length=1,
    )
