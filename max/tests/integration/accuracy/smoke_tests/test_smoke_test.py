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

from hf_repo_lock import load_db
from smoke_tests.smoke_test import MODEL_ALIASES


def test_hf_repo_lock_tsv_reachable() -> None:
    assert len(load_db()) > 0, "hf-repo-lock.tsv not found or empty"


def test_model_aliases_are_valid() -> None:
    for alias, config in MODEL_ALIASES.items():
        assert "max_serve_args" in config, (
            f"Model alias {alias!r} must contain 'max_serve_args'"
        )


def test_model_aliases_contain_exactly_one_double_underscore() -> None:
    for alias in MODEL_ALIASES:
        count = alias.count("__")
        assert count == 1, (
            f"Model alias {alias!r} must contain exactly one '__'"
            f" (found {count})"
        )


def test_all_alias_hf_model_paths_in_hf_repo_lock() -> None:
    """Every MODEL_ALIASES key's hf_model_path prefix must be pinned in hf-repo-lock.tsv."""
    lock = load_db()
    missing = [
        alias for alias in MODEL_ALIASES if alias.rsplit("__", 1)[0] not in lock
    ]
    assert not missing, (
        f"MODEL_ALIASES hf_model_path prefixes missing from hf-repo-lock.tsv: {missing}"
    )


def test_model_aliases_lookup_is_case_insensitive() -> None:
    for key in MODEL_ALIASES:
        assert MODEL_ALIASES.get(key.lower()) is not None
        assert MODEL_ALIASES.get(key.upper()) is not None
