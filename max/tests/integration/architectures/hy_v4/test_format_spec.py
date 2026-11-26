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
"""Checks that Hy4 artifacts carry the FormatSpec Mach parses output with."""

import json
from pathlib import Path

import pytest
from max._artifact.export import checkpoint_format_spec
from max._artifact.format_spec import FormatSpec
from max._artifact.model_artifact import ModelArtifact
from max._artifact.serialization import deserialize, serialize
from max.pipelines.architectures.hy_v4 import arch
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.weights.hf_utils import HuggingFaceRepo


@pytest.mark.parametrize(
    "architecture", [arch.hy_v4_arch, arch.hy_v4_vl_arch], ids=lambda a: a.name
)
def test_both_archs_ship_the_hy4_format_spec_in_the_artifact(
    tmp_path: Path, architecture: SupportedArchitecture
) -> None:
    # Without the section, Mach runs no reasoning parser and returns the
    # thinking in content.
    document = checkpoint_format_spec(
        architecture, HuggingFaceRepo(str(tmp_path))
    )
    assert document == Path(arch.__file__).parent / "formatspec.json"

    dest = serialize(
        ModelArtifact(format_spec=FormatSpec(document)), tmp_path / "hy4"
    )
    sections = json.loads((dest / "artifact.json").read_text())
    assert sections["format_spec"] == {
        "document": {"$file": "format_spec.json"}
    }
    loaded = deserialize(dest).format_spec
    assert loaded is not None
    assert loaded.document.read_bytes() == document.read_bytes()
