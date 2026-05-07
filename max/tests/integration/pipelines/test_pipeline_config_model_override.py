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

"""Regression test for ``--model-override`` revision sequencing.

Under ``HF_HUB_OFFLINE`` the cache is keyed by revision, so any HF lookup
that uses the dataclass-default revision ``"main"`` will fail when only
the pinned SHA is pre-downloaded. ``main.*`` and ``draft.*`` overrides
must therefore reach ``MAXModelConfig`` construction before
``HuggingFaceRepo.__post_init__`` resolves the offline cache.
"""

from unittest.mock import patch

from max.pipelines import PipelineConfig

GENERATE_LOCAL_PATH = "max.pipelines.lib.hf_utils.generate_local_model_path"
HF_OFFLINE = "huggingface_hub.constants.HF_HUB_OFFLINE"

PINNED_SHA = "abcdef123"


def test_main_override_revision_used_for_offline_cache_lookup() -> None:
    revisions_seen: list[str] = []

    def stub(repo_id: str, revision: str) -> str:
        revisions_seen.append(revision)
        return f"/fake/cache/{repo_id}__{revision}"

    with patch(HF_OFFLINE, True), patch(GENERATE_LOCAL_PATH, side_effect=stub):
        # Construction may raise downstream once it tries to load weights
        # from the fake path; what we care about is which revisions were
        # asked for during HuggingFaceRepo construction.
        try:
            PipelineConfig(  # type: ignore[call-arg]
                model_path="some/repo",
                model_override=[
                    f"main.huggingface_model_revision={PINNED_SHA}",
                    f"main.huggingface_weight_revision={PINNED_SHA}",
                ],
            )
        except Exception:
            pass

    assert set(revisions_seen) == {PINNED_SHA}, (
        f"Expected every offline lookup to use the pinned SHA "
        f"{PINNED_SHA!r}; got {revisions_seen}"
    )
