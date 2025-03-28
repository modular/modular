# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.driver import DeviceSpec
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.config_enums import SupportedEncoding
from test_common.pipeline_model import DUMMY_GPTQ_ARCH, prepare_registry


@prepare_registry
def test_config__raises_with_unsupported_GPTQ_format():
    PIPELINE_REGISTRY.register(DUMMY_GPTQ_ARCH)
    # this should work
    config = PipelineConfig(
        model_path="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        quantization_encoding=SupportedEncoding.gptq,
        device_specs=[DeviceSpec.accelerator()],
    )

    # We expect this to fail.
    with pytest.raises(ValueError):
        unsupported_config = PipelineConfig(
            model_path="jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4",
            quantization_encoding=SupportedEncoding.gptq,
            device_specs=[DeviceSpec.accelerator()],
        )
