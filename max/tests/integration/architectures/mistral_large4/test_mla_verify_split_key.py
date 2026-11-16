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
"""The verify's dispatch split, resolved through the op that publishes it.

``test_decode_capture_safe.py`` pins that the cache tree the pipeline resolves
carries the row-invariant pin exactly where the sparse op takes it. This pins
what the dispatch op does with that opt-in, which no host-only test can see:
the split it returns is the pin's shape-pure count, unchanged as the cache
grows, where the same cache without the opt-in takes main's fitted record and
moves with the live cache length.

The numbers are the ones ``test_mla_sparse_verify_rows_graph`` pins for the
kernel's own dispatch key at this shape (16 opted, 1 fitted at a 126-key
cache), so a drift on either side of the Mojo binding fails one of the two.
"""

from __future__ import annotations

import dataclasses

import pytest
from max.driver import accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAAttnKey, MLAKVCacheParams, MultiKVCacheParams
from max.nn.kv_cache.utils import MultiAttnKey
from max.pipelines.architectures.mistral_large4_modulev3.config import (
    INDEXER_GROUP,
    NOPE_MLA_GROUP,
    ROPE_MLA_GROUP,
    build_kv_params,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers.sparse_mla import (
    sparse_selection_width,
)
from max.pipelines.architectures.mistral_large4_modulev3.mistral_large4 import (
    _pin_verify_split,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    ML4_DSPARK_VERIFY_ROWS,
)
from max.pipelines.lib import KVCacheConfig

ROPE_LAYERS = [True, False]
"""One rope flag per layer: layer 0 owns a rope latent and layer 1 does not,
so the tree holds both latent groups plus the indexer."""
TP8 = 8
"""The deployment's rank count: 8 heads per rank, 32 rows over a verify."""

INDEX_TOPK = 2048
"""The selection width the launch passes as its sparse index stride."""

INDEX_HEAD_DIM = 128
"""The width of one indexer key row: the indexer's own cache row."""

SHORT_CACHE = 126
"""A cache under the top-k, where the fitted record collapses to one split."""

LONG_CACHE = 4096
"""A cache past the stride, where the fitted record splits."""

PINNED_SPLIT = 16
"""The pin's count at this shape; ``sparse_decode_partitions_for_rows``' own."""

FITTED_SHORT_SPLIT = 1
"""Main's record at the short cache: one page per split is not worth it."""


def _built_tree() -> MultiKVCacheParams:
    """The cache tree the config builds for the deployment, before the pin."""
    tree = build_kv_params(
        rope_layers=list(ROPE_LAYERS),
        kv_source_layers=[0, 1],
        indexer_types=["full", "shared"],
        num_attention_heads=64,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        devices=[DeviceRef.GPU(0)] * TP8,
        kv_cache_config=KVCacheConfig(kv_cache_page_size=128),
        cache_dtype=DType.float8_e4m3fn,
        speculative_method="dspark",
        num_draft_tokens=ML4_DSPARK_VERIFY_ROWS - 1,
    )
    assert isinstance(tree, MultiKVCacheParams)
    return tree


def _served_tree() -> MultiKVCacheParams:
    """The tree the served model keys its captures from: built, then pinned.

    The pin lands on the model's own tree in ``MistralLarge4.__init__``, after
    the builder returns, which is why it is applied here rather than passed in.
    """
    tree = _built_tree()
    _pin_verify_split(tree, ROPE_LAYERS)
    return tree


def _rope_leaf() -> MLAKVCacheParams:
    """The rope group's latent leaf, exactly as the tree builder keys it.

    No field is overridden: the op passes the selection's width as the sparse
    index stride and declares the indices physical, and the builder sets both
    on the leaf for that reason. A leaf that kept the dense defaults would key
    every cache length separately while the launch took one split.
    """
    leaf = _built_tree().children[ROPE_MLA_GROUP]
    assert isinstance(leaf, MLAKVCacheParams)
    return leaf


def _split(leaf: MLAKVCacheParams, cache_length: int) -> int:
    """The split the dispatch op returns for a verify of ``cache_length``."""
    key = leaf.resolve_attn_key(1, ML4_DSPARK_VERIFY_ROWS, cache_length)
    assert isinstance(key, MLAAttnKey)
    return int(key.num_partitions)


def _tree_key(
    tree: MultiKVCacheParams, query_rows: int, cache_length: int
) -> MultiAttnKey:
    """The composite key the runner captures a step of this shape under."""
    key = tree.resolve_attn_key(1, query_rows, cache_length)
    assert isinstance(key, MultiAttnKey)
    return key


def test_the_leaf_carries_the_dispatch_fields_the_launch_passes() -> None:
    """The builder's leaf states what the sparse op passes, so the key agrees.

    The op hands the kernel ``physical_indices=True`` and the selection's width
    as the index stride; the leaf the key is resolved from has to say the same,
    which is what makes the two views in
    ``test_decode_capture_safe.test_the_capture_key_takes_the_split_the_launch_takes``
    one decision rather than two.
    """
    leaf = _rope_leaf()
    assert leaf.sparse_indices_stride == sparse_selection_width(
        leaf, INDEX_TOPK
    )
    assert leaf.sparse_physical_indices
    assert not leaf.sparse_row_invariant_verify


def test_the_indexer_key_is_the_indexers_own() -> None:
    """Only the indexer's selection width is set; its form stays its own.

    The leaf's stride is what keeps its key off the live cache; its dispatch
    form and row width are what keep that key on the quantized record, which
    sizes the split from the stride alone. Borrowing a latent's form instead --
    ``joining_sparse_dispatch_of``, which other architectures' indexer leaves
    use -- reroutes it through the latent's record, and that one reads the
    cache length.
    """
    tree = _built_tree()
    indexer = tree.children[INDEXER_GROUP]
    assert isinstance(indexer, MLAKVCacheParams)
    assert indexer.sparse_indices_stride == sparse_selection_width(
        indexer, INDEX_TOPK
    )
    rope = _rope_leaf()
    assert indexer.sparse_kv_form is None
    assert indexer.resolved_sparse_kv_form != rope.resolved_sparse_kv_form
    assert indexer.sparse_dispatch_head_dim is None
    assert indexer.resolved_sparse_dispatch_head_dim == INDEX_HEAD_DIM


@pytest.mark.skipif(
    accelerator_count() == 0, reason="resolving a key materializes a device"
)
def test_the_pinned_verify_keys_the_split_the_kernel_pins() -> None:
    """The opt-in reaches the dispatcher and returns the pin's own count."""
    leaf = dataclasses.replace(_rope_leaf(), sparse_row_invariant_verify=True)
    assert _split(leaf, SHORT_CACHE) == PINNED_SPLIT
    assert _split(leaf, LONG_CACHE) == PINNED_SPLIT


@pytest.mark.skipif(
    accelerator_count() == 0, reason="resolving a key materializes a device"
)
def test_without_the_opt_in_the_key_keeps_mains_fitted_record() -> None:
    """Nothing else moved: an un-opted cache keys as main keys it."""
    leaf = _rope_leaf()
    assert not leaf.sparse_row_invariant_verify
    assert _split(leaf, SHORT_CACHE) == FITTED_SHORT_SPLIT
    assert _split(leaf, LONG_CACHE) != _split(leaf, SHORT_CACHE)


@pytest.mark.skipif(
    accelerator_count() == 0, reason="resolving a key materializes a device"
)
@pytest.mark.parametrize(
    "query_rows",
    [1, ML4_DSPARK_VERIFY_ROWS],
    ids=["decode", "verify"],
)
def test_the_served_tree_keys_one_way_however_long_the_cache(
    query_rows: int,
) -> None:
    """The served tree's whole key is a property of the shape, not the cache.

    The runner captures a step under ``MultiKVCacheParams.resolve_attn_key``,
    which folds every attention child -- both latents and the indexer -- so one
    cache-tracking child is enough to capture the step again at every cache
    length. Comparing keys rather than leaf counts is the only way to see that:
    a per-leaf assertion passes while a sibling still moves.

    Both widths matter because an MLA leaf keys differently on a one-row decode
    and on the four-row verify, so a field only one of them reads is invisible
    to the other.
    """
    tree = _served_tree()
    assert {
        name for name, _ in _tree_key(tree, query_rows, SHORT_CACHE).children
    } == {
        ROPE_MLA_GROUP,
        NOPE_MLA_GROUP,
        INDEXER_GROUP,
    }
    assert _tree_key(tree, query_rows, SHORT_CACHE) == _tree_key(
        tree, query_rows, LONG_CACHE
    )
