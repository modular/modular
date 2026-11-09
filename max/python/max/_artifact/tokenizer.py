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
"""The ``tokenizer`` component: a checkpoint's tokenizer files and ids."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self

from .value_types import (
    ArtifactFile,
    ArtifactType,
    HuggingFaceRef,
    decode_field,
    expect,
)


@dataclass(frozen=True)
class Tokenizer:
    """The tokenizer the exported checkpoint is served with.

    Args:
        family: The orchestrator tokenizer family, or ``None`` to have the
            orchestrator pick it by model name.
        flavor: The family's flavor, or ``None`` for its only one.
        files: The checkpoint's tokenizer files, keyed by file name, each
            referenced at the checkpoint's revision rather than copied.
        chat_templates: Each chat template's text, or the path of its file,
            keyed by template name.
        vocab_size: The model's embedding row count.
        sampleable_vocab_size: How many token ids the tokenizer defines, at
            most ``vocab_size``; the rows past it are padding.
        eos_token_ids: The token ids that end generation.
    """

    family: str | None
    flavor: str | None
    files: Mapping[str, HuggingFaceRef]
    chat_templates: Mapping[str, str | Path]
    vocab_size: int
    sampleable_vocab_size: int
    eos_token_ids: tuple[int, ...]

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the ids, the file references and the templates."""
        return {
            "family": self.family,
            "flavor": self.flavor,
            "files": dict(self.files),
            "chat_templates": {
                name: ArtifactFile(
                    f"tokenizer/chat_templates/{name}.jinja", template
                )
                for name, template in self.chat_templates.items()
            },
            "vocab_size": self.vocab_size,
            "sampleable_vocab_size": self.sampleable_vocab_size,
            "eos_token_ids": list(self.eos_token_ids),
        }

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        """Rebuilds the component written by :meth:`__serialize__`.

        Raises:
            ArtifactError: If a field is missing or malformed.
        """
        return cls(
            family=decode_field(
                value, "family", lambda v: None if v is None else expect(v, str)
            ),
            flavor=decode_field(
                value, "flavor", lambda v: None if v is None else expect(v, str)
            ),
            files=decode_field(value, "files", _read_refs),
            chat_templates=decode_field(value, "chat_templates", _read_files),
            vocab_size=decode_field(
                value, "vocab_size", lambda v: expect(v, int)
            ),
            sampleable_vocab_size=decode_field(
                value, "sampleable_vocab_size", lambda v: expect(v, int)
            ),
            eos_token_ids=decode_field(
                value,
                "eos_token_ids",
                lambda v: tuple(expect(i, int) for i in expect(v, list)),
            ),
        )


def _read_files(value: ArtifactType) -> dict[str, Path]:
    return {
        name: expect(expect(file, ArtifactFile).content, Path)
        for name, file in expect(value, dict).items()
    }


def _read_refs(value: ArtifactType) -> dict[str, HuggingFaceRef]:
    return {
        name: expect(ref, HuggingFaceRef)
        for name, ref in expect(value, dict).items()
    }
