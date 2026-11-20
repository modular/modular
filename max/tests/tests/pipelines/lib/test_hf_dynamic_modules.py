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

import hashlib
from pathlib import Path

import pytest
from transformers.dynamic_module_utils import get_class_from_dynamic_module


@pytest.mark.parametrize("symlinked", [False, True])
def test_local_dynamic_module_tracks_relative_imports(
    tmp_path: Path, symlinked: bool
) -> None:
    snapshot = tmp_path / "snapshots" / "revision"
    blobs = tmp_path / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()

    def write_source(name: str, source: str) -> None:
        path = snapshot / name
        if symlinked:
            blob = blobs / hashlib.sha256(source.encode()).hexdigest()
            blob.write_text(source)
            path.unlink(missing_ok=True)
            path.symlink_to(blob)
        else:
            path.write_text(source)

    write_source(
        "tokenization.py",
        "from .encoding import VALUE\nclass Tokenizer:\n    value = VALUE\n",
    )
    write_source("encoding.py", "from .constants import VALUE\n")
    for value in [1, 2]:
        write_source("constants.py", f"VALUE = {value}\n")
        tokenizer = get_class_from_dynamic_module(
            "tokenization.Tokenizer", str(snapshot), local_files_only=True
        )
        assert tokenizer.value == value
