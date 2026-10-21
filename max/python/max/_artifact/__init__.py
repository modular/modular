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
"""Exports a pipeline to, and loads it from, a ``.m`` pipeline artifact.

An artifact is a directory holding ``metadata.json`` (provenance and file
checksums), ``artifact.json`` (one section per pipeline component), and any
files the components reference, such as the compiled MEFs.

The package's own import exposes only the value types, which
:mod:`max.experimental` uses, so importing it never imports the pipeline
stack. The writer and reader are in :mod:`.serialization` and
:mod:`.model_artifact`, and ``max export`` is in :mod:`.export`.
"""

from .value_types import (
    ArtifactComponent,
    ArtifactError,
    ArtifactFile,
    ArtifactType,
    FileWriter,
    HuggingFaceRef,
)

__all__ = [
    "ArtifactComponent",
    "ArtifactError",
    "ArtifactFile",
    "ArtifactType",
    "FileWriter",
    "HuggingFaceRef",
]
