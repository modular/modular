# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import tempfile
from pathlib import Path

import numpy as np
from max.engine import InferenceSession


def test_api_source(mo_model_path: Path):
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        filepath = temp_file.name
    os.environ["MODULAR_TELEMETRY_EXPORTERS_LOGS_FILE_PATH"] = filepath
    session = InferenceSession()
    model = session.load(mo_model_path)
    _ = model.execute(np.ones(5, dtype=np.float32))
    expected_line = "max.engine.api.language: python"

    with open(filepath) as file:
        lines = [line.strip() for line in file.readlines()]

    assert expected_line in lines
