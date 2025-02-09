# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: rm -f max-engine-telemetry-test-telemetry
# RUN: export MODULAR_TELEMETRY_EXPORTERS_LOGS_FILE_PATH=max-engine-telemetry-test-telemetry
# RUN: %mojo -debug-level full %s %S/../../Inputs/relu3x100x100.torchscript
# RUN: rm -f max-engine-telemetry-test-telemetry

from pathlib import Path
from sys import argv

from max.engine import InferenceSession, InputSpec
from max.tensor import TensorSpec
from testing import assert_true


fn test_model_execute() raises:
    var file_name = "max-engine-telemetry-test-telemetry"

    var args = argv()
    var model_path = args[1]

    var session = InferenceSession()
    var model = session.load(
        Path(model_path),
        input_specs=List[InputSpec](TensorSpec(DType.float32, 1, 3, 100, 100)),
    )

    var file_path = Path(file_name)
    var text = file_path.read_text()
    var expected_msg = "max.engine.api.language: mojo"
    assert_true(expected_msg in text)


fn main() raises:
    test_model_execute()
