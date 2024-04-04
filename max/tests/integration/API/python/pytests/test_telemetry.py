# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import tempfile
from pathlib import Path
from subprocess import run

import max.engine as me
import numpy as np


def test_api_source(mo_model_path: Path):
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        filepath = temp_file.name
    os.environ["MODULAR_TELEMETRY_EXPORTERS_LOGS_FILE_PATH"] = filepath
    session = me.InferenceSession()
    model = session.load(mo_model_path)
    output = model.execute(input=np.ones((5)))
    expected_line = "max.engine.api.language: python"

    with open(filepath, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    assert expected_line in lines
