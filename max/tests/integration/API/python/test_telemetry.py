# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import os
import tempfile
from pathlib import Path

import numpy as np
from max.driver import CPU
from max.engine import InferenceSession


def test_api_source(mo_model_path: Path) -> None:
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        filepath = temp_file.name
    os.environ["MODULAR_TELEMETRY_EXPORTERS_LOGS_FILE_PATH"] = filepath
    session = InferenceSession(devices=[CPU()])
    model = session.load(mo_model_path)
    _ = model.execute(np.ones(5, dtype=np.float32))
    expected_line = "max.engine.api.language: python"

    with open(filepath) as file:
        lines = [line.strip() for line in file]

    assert expected_line in lines
