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

import os
import subprocess
import sys
from pathlib import Path


def test_native_session_reports_malformed_config(tmp_path: Path) -> None:
    (tmp_path / "modular.cfg").write_text("missing equals sign\n")
    # Config is cached on first use, so start a fresh interpreter.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pytest\n"
            "from max._core.engine import InferenceSession\n"
            "with pytest.raises(ValueError, match='malformed line'):\n"
            "    InferenceSession(devices=[], custom_extensions=[])",
        ],
        env={
            **os.environ,
            "MODULAR_HOME": str(tmp_path),
            "PYTHONPATH": os.pathsep.join(sys.path),
        },
        check=True,
        timeout=30,
    )
