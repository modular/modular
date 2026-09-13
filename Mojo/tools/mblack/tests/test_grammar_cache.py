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

"""Tests for grammar cache keying by content hash."""

import os
import shutil
import sys
import tempfile

import pytest

from mblib2to3.pgen2.driver import (
    _generate_pickle_name,
    load_packaged_grammar,
)


class TestGeneratePickleName:
    def test_no_content_hash(self):
        name = _generate_pickle_name("Grammar.txt")
        assert name.endswith(".pickle")
        assert "Grammar" in name
        # Should not contain any hash segment beyond the Python version
        version_str = ".".join(map(str, sys.version_info))
        assert version_str in name

    def test_with_content_hash(self):
        name = _generate_pickle_name("Grammar.txt", content_hash="abc123")
        assert ".abc123." in name
        assert name.endswith(".pickle")

    def test_different_hashes_produce_different_names(self):
        name_a = _generate_pickle_name(
            "Grammar.txt", content_hash="aaa"
        )
        name_b = _generate_pickle_name(
            "Grammar.txt", content_hash="bbb"
        )
        assert name_a != name_b

    def test_cache_dir_with_content_hash(self):
        name = _generate_pickle_name(
            "/some/path/Grammar.txt",
            cache_dir="/cache",
            content_hash="deadbeef",
        )
        assert name.startswith("/cache/")
        assert ".deadbeef." in name
        # Should only contain basename, not full path
        assert "/some/path/" not in name

    def test_no_content_hash_backward_compat(self):
        """Without a content hash the name is unchanged from the original."""
        name = _generate_pickle_name("Grammar.txt")
        version_str = ".".join(map(str, sys.version_info))
        expected = f"Grammar{version_str}.pickle"
        assert name == expected


class TestLoadPackagedGrammarCacheIsolation:
    """Verify that different grammar file contents produce separate caches."""

    @pytest.fixture()
    def grammar_env(self, tmp_path):
        """Create two grammar files with different content and a shared cache."""
        # Copy the real Grammar.txt as the base
        real_grammar = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "mblib2to3",
            "Grammar.txt",
        )

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # "env A" — the real grammar
        env_a = tmp_path / "env_a"
        env_a.mkdir()
        shutil.copy(real_grammar, env_a / "Grammar.txt")

        # "env B" — same grammar with a trivial addition (extra blank line)
        env_b = tmp_path / "env_b"
        env_b.mkdir()
        grammar_b = env_b / "Grammar.txt"
        grammar_b.write_text(
            (env_a / "Grammar.txt").read_text() + "\n# extra\n"
        )

        return env_a, env_b, cache_dir

    def test_different_grammars_use_different_pickles(self, grammar_env):
        env_a, env_b, cache_dir = grammar_env

        load_packaged_grammar(
            "mblib2to3", str(env_a / "Grammar.txt"), cache_dir=str(cache_dir)
        )
        pickles_after_a = set(os.listdir(cache_dir))

        load_packaged_grammar(
            "mblib2to3", str(env_b / "Grammar.txt"), cache_dir=str(cache_dir)
        )
        pickles_after_b = set(os.listdir(cache_dir))

        # A second, different pickle must have been created.
        new_pickles = pickles_after_b - pickles_after_a
        assert len(new_pickles) == 1, (
            f"Expected a new pickle for the different grammar, "
            f"but found: {new_pickles}"
        )

    def test_same_grammar_reuses_pickle(self, grammar_env):
        env_a, _, cache_dir = grammar_env

        load_packaged_grammar(
            "mblib2to3", str(env_a / "Grammar.txt"), cache_dir=str(cache_dir)
        )
        pickles_first = set(os.listdir(cache_dir))

        # Load the same grammar again.
        load_packaged_grammar(
            "mblib2to3", str(env_a / "Grammar.txt"), cache_dir=str(cache_dir)
        )
        pickles_second = set(os.listdir(cache_dir))

        assert pickles_first == pickles_second
