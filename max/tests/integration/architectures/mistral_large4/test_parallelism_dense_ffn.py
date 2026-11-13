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
"""The dense layer-0 FFN shards only where every rank holds every token.

``w2`` is row parallel, so a TP split of the dense FFN yields a partial sum on
every rank. The split is legal exactly when every rank holds every token:
under ``TP_TP`` (single device or pure TP), and under TP attention beside EP's
allreduce backend, which all-reduces after the FFN. Under EP's reduce-scatter
backend each rank holds a disjoint row subset instead, and ``post_mlp`` only
all-gathers rows, so a split there would leave the sum incomplete -- silently
wrong logits on more than one device. A data-parallel replica holds only its
own rows for the same reason, so the allreduce backend does not make the split
legal there either.

These tests are CPU only and pin that decision table. The numerical consequence
is structural: with the split the graph gains a real all-reduce after the dense
FFN, and without it no collective follows it at all.
"""

from __future__ import annotations

from max.pipelines.architectures.deepseekV3_modulev3.layers.transformer_block import (
    ParallelismMode,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers.transformer_block import (
    _shards_dense_ffn,
)


def test_tp_tp_shards_the_dense_ffn() -> None:
    """Pure tensor parallelism replicates every token, so the split is legal."""
    assert _shards_dense_ffn(ParallelismMode.TP_TP, None)


def test_tp_ep_with_allreduce_shards_the_dense_ffn() -> None:
    """The allreduce EP backend is the arm ``gen_mef --ep-use-allreduce`` takes."""
    assert _shards_dense_ffn(ParallelismMode.TP_EP, True)


def test_tp_ep_with_reduce_scatter_keeps_the_dense_ffn_whole() -> None:
    """Row-parallel tokens make a sharded down projection silently partial."""
    assert not _shards_dense_ffn(ParallelismMode.TP_EP, False)


def test_data_parallel_never_shards_the_dense_ffn() -> None:
    """A DP replica holds its own rows, so there is nothing to shard."""
    assert not _shards_dense_ffn(ParallelismMode.DP_EP, None)
    assert not _shards_dense_ffn(ParallelismMode.DP_EP, False)
    assert not _shards_dense_ffn(ParallelismMode.DP_EP, True)
