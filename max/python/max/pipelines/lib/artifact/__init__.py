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
"""Exports a pipeline to, and loads it from, a ``.m`` model artifact.

An artifact is a directory holding ``metadata.json`` (provenance and file
checksums), ``artifact.json`` (one section per pipeline component), and any
files the components reference, such as the compiled MEF.
"""

from .metadata import ArtifactMetadata
from .model_artifact import ModelArtifact
from .serialization import deserialize, serialize
from .types import (
    ArtifactComponent,
    ArtifactError,
    ArtifactFile,
    ArtifactType,
    HuggingFaceRef,
)

__all__ = [
    "ArtifactComponent",
    "ArtifactError",
    "ArtifactFile",
    "ArtifactMetadata",
    "ArtifactType",
    "HuggingFaceRef",
    "ModelArtifact",
    "deserialize",
    "serialize",
]
