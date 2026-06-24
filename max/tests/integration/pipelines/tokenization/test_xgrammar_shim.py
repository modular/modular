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

"""Parity/integration test for the ``max._xgrammar`` shim.

Validates that the shim restores the upstream xgrammar Python-layer surface MAX's
structured-output backend depends on -- over the nanobind binding --
including the structural-tag bridge (vendored pydantic models ->
``model_dump_json`` -> the C++ structural-tag compiler) used by the tool-call
path.
"""

import importlib.util
import sys

import numpy as np
from max import _xgrammar as xgr

_VOCAB = [
    "{",
    "}",
    "[",
    "]",
    '"',
    ":",
    ",",
    " ",
    "a",
    "b",
    "1",
    "true",
    "null",
    "<eos>",
]


def _compiler() -> xgr.GrammarCompiler:
    tokenizer_info = xgr.TokenizerInfo(
        _VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[len(_VOCAB) - 1],
    )
    return xgr.GrammarCompiler(tokenizer_info)


def test_shim_is_torch_free() -> None:
    assert "torch" not in sys.modules
    assert importlib.util.find_spec("torch") is None


def test_shim_surface_present() -> None:
    assert hasattr(xgr.TokenizerInfo, "from_huggingface")
    assert callable(xgr.get_builtin_structural_tag)
    for name in (
        "TokenizerInfo",
        "VocabType",
        "GrammarCompiler",
        "CompiledGrammar",
        "GrammarMatcher",
        "StructuralTag",
        "StructuralTagItem",
        "allocate_token_bitmask",
    ):
        assert hasattr(xgr, name)


def test_allocate_token_bitmask() -> None:
    bitmask = xgr.allocate_token_bitmask(2, 100)
    assert bitmask.shape == (2, (100 + 31) // 32)
    assert bitmask.dtype == np.int32
    assert (bitmask == -1).all()


def test_json_schema_path_through_shim() -> None:
    compiled = _compiler().compile_json_schema('{"type": "object"}')
    assert isinstance(compiled, xgr.CompiledGrammar)
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = np.full((xgr.get_bitmask_size(len(_VOCAB)),), -1, dtype=np.int32)
    assert matcher.fill_next_token_bitmask(bitmask) is True
    assert int(bitmask[0]) != -1


def test_structural_tag_bridge() -> None:
    tag = xgr.StructuralTag.from_legacy_structural_tag(
        [xgr.StructuralTagItem(begin="a", schema={"type": "object"}, end="b")],
        triggers=["a"],
    )
    roundtripped = xgr.StructuralTag.model_validate_json(tag.model_dump_json())
    assert isinstance(roundtripped, xgr.StructuralTag)

    compiled = _compiler().compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)
