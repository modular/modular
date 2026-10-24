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

"""The KV tree a target and its draft share.

The pool sizes one tree for both models, so the draft must be built on the
target's devices, page size and dtype, and every leaf must carry the width the
driver verifies at. A wrong tree is a silent wrong answer at serve time.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _kv_tree,
    _Pairing,
)

PAGE_SIZE = 32
WIDTH = 3


def _leaf(num_draft_tokens: int = 0) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=[DeviceRef.CPU()],
        n_kv_heads=1,
        head_dim=16,
        num_layers=2,
        page_size=PAGE_SIZE,
        num_draft_tokens=num_draft_tokens,
    )


def _pairing(
    draft_kv: KVCacheParams | None, seen: list[dict[str, Any]]
) -> _Pairing:
    """Returns a pairing whose draft builds ``draft_kv`` and records what
    it was given."""

    class _DraftConfig:
        @staticmethod
        def construct_kv_params(
            huggingface_config: object,
            pipeline_config: object,
            devices: list[DeviceRef],
            kv_cache_config: KVCacheConfig,
            cache_dtype: DType,
        ) -> KVCacheParams | None:
            seen.append(
                {
                    "devices": devices,
                    "page_size": kv_cache_config.kv_cache_page_size,
                    "dtype": cache_dtype,
                }
            )
            return draft_kv

    draft = SimpleNamespace(pipeline_model=None, config=_DraftConfig)
    return _Pairing("Target", cast(Any, SimpleNamespace()), cast(Any, draft))


def _tree(
    draft_kv: KVCacheParams | None,
) -> tuple[MultiKVCacheParams, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    pipeline_config = SimpleNamespace(
        speculative=SimpleNamespace(draft_width=WIDTH),
        model=SimpleNamespace(kv_cache=KVCacheConfig()),
        draft_model=SimpleNamespace(huggingface_config=SimpleNamespace()),
    )
    tree = _kv_tree(
        cast(PipelineConfig, pipeline_config),
        _pairing(draft_kv, seen),
        _leaf(),
    )
    return tree, seen


def test_the_draft_is_built_on_the_targets_devices_page_size_and_dtype() -> (
    None
):
    _, seen = _tree(_leaf())
    assert seen == [
        {
            "devices": [DeviceRef.CPU()],
            "page_size": PAGE_SIZE,
            "dtype": DType.bfloat16,
        }
    ]


def test_both_leaves_carry_the_width_the_driver_verifies_at() -> None:
    tree, _ = _tree(_leaf())
    assert set(tree.children) == {"target", "draft"}
    for child in tree.children.values():
        assert isinstance(child, KVCacheParams)
        assert child.num_draft_tokens == WIDTH


def test_a_draft_without_a_cache_has_no_draft_child() -> None:
    """A draft whose config has no KV params reads and writes the target's
    leaves, so it adds none."""
    tree, _ = _tree(None)
    assert set(tree.children) == {"target"}
    target = tree.children["target"]
    assert isinstance(target, KVCacheParams)
    assert target.num_draft_tokens == WIDTH


def test_a_draft_that_writes_more_slots_than_it_verifies_widens_the_tree() -> (
    None
):
    """A block draft's own leaf declares the wider count it writes; the target
    leaf is stamped to match, so the pool reserves the same slots for both."""
    tree, _ = _tree(_leaf(num_draft_tokens=WIDTH + 4))
    for child in tree.children.values():
        assert isinstance(child, KVCacheParams)
        assert child.num_draft_tokens == WIDTH + 4
