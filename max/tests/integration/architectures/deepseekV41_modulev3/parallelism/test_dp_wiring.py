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
"""The DP/EP wiring contracts: input declarations, guards, the EP config.

Every test here is CPU-only: they check what the pipeline declares and
refuses before any graph is built, plus the routed-experts leaf contract
the EP dispatch quantizer reads and the host-resident row count the
grouped kernels' tile-regime estimate derives from. The DP4+EP4 graph
itself is staged by ``test_dp4_graph``.
"""

from __future__ import annotations

import random
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.legacy.kv_cache import KVCacheInputs
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV41_modulev3.batch_processor import (
    DeepseekV41BatchProcessor,
)
from max.pipelines.architectures.deepseekV41_modulev3.deepseekV41 import (
    DeepseekV41,
    _host_total_rows_op,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
    extra_inputs,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.moe import (
    RoutedExperts,
)
from max.pipelines.architectures.deepseekV41_modulev3.model import (
    DeepseekV41Model,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh
from runtime_config import text_config
from test_common.context_utils import create_text_context


def _interface_config() -> V41Config:
    return V41Config.from_hf({"text_config": vars(text_config())})


def _model_config(
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
) -> DeepseekV41Config:
    text = text_config()
    kv = build_kv_params(
        _interface_config(),
        KVCacheConfig(kv_cache_page_size=128),
        [DeviceRef.CPU()],
    )
    config = DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )
    config.return_logits = return_logits
    config.return_hidden_states = return_hidden_states
    return config


def _batch_processor(dp: int) -> DeepseekV41BatchProcessor:
    runtime = Mock()
    runtime.pipeline_config.model.data_parallel_degree = dp
    runtime.pipeline_config.model.huggingface_config = {
        "text_config": vars(text_config())
    }
    return DeepseekV41BatchProcessor(Mock(), runtime)


def _ep_manager() -> EPBatchManager:
    """The EP batch manager the pipeline model builds at ep_size 4."""
    model = _model(dp=4, ep_size=4)
    return EPBatchManager(model._build_ep_config(_model_config(), n_devices=4))


def test_extra_inputs_name_global_rows_under_dp() -> None:
    config = _interface_config()
    refs = [DeviceRef.GPU(0)]
    per_request = dict(extra_inputs(config, refs, data_parallel_degree=4))
    for name in ("swa_begin", "engram_lookback", "engram_lookback_dead"):
        assert per_request[name].shape[0] == "global_batch_rows", name
    # The tensor-parallel ABI keeps the per-replica batch symbol, which
    # unifies the rows with each replica's KV metadata.
    assert dict(extra_inputs(config, refs))["swa_begin"].shape[0] == (
        "replica_0_batch_size"
    )


def test_batch_processor_declares_the_dp_pair() -> None:
    processor = _batch_processor(dp=4)
    kv_params = build_kv_params(
        _interface_config(),
        KVCacheConfig(kv_cache_page_size=128),
        [DeviceRef.GPU(i) for i in range(4)],
        data_parallel_degree=4,
    )
    inputs = processor.get_symbolic_inputs(
        kv_params=kv_params,
        device_refs=[DeviceRef.GPU(i) for i in range(4)],
    )
    # The CPU split pair sits between the row offsets and the KV leaves,
    # matching the forward's variadic order.
    assert inputs[3] == TensorType(
        DType.int64, shape=[5], device=DeviceRef.CPU()
    )
    assert inputs[4] == TensorType(
        DType.int64, shape=["input_row_offsets_len"], device=DeviceRef.CPU()
    )


def test_batch_processor_declares_no_pair_at_dp_one() -> None:
    processor = _batch_processor(dp=1)
    kv_params = build_kv_params(
        _interface_config(),
        KVCacheConfig(kv_cache_page_size=128),
        [DeviceRef.GPU(0)],
    )
    inputs = processor.get_symbolic_inputs(
        kv_params=kv_params, device_refs=[DeviceRef.GPU(0)]
    )
    splits = TensorType(DType.int64, shape=[2], device=DeviceRef.CPU())
    assert splits not in inputs


def test_prepare_inputs_flattens_replica_batches_at_dp() -> None:
    """Under data parallelism the replicas pack as one global batch.

    The requests keep their order in the flattened batch, the CPU split
    boundaries carry the cumulative request counts, and the int64
    row-offset mirror matches the flattened offsets. An empty replica
    contributes a zero-width boundary, not a gap; the per-request extras
    are sized by the global request count the graph slices per rank.
    """
    processor = _batch_processor(dp=2)
    processor.runtime.devices = [CPU()]
    processor.runtime.signal_buffers = []
    # The static tail of the ABI (the token map) is fed from whatever the
    # model bound; this test binds it directly instead.
    processor.bind_engram(
        {"engram_token_map": Buffer.from_numpy(np.zeros(1, np.int32))}
    )
    replica_batches = [
        [
            create_text_context(np.array([1, 2, 3], dtype=np.int64)),
            create_text_context(np.array([4], dtype=np.int64)),
        ],
        [],
    ]
    inputs = processor.prepare_initial_token_inputs(
        replica_batches,
        kv_cache_inputs=cast(KVCacheInputs[Buffer, Buffer], Mock()),
    )
    assert inputs.data_parallel_splits is not None
    assert inputs.data_parallel_splits.to_numpy().tolist() == [0, 2, 2]
    assert inputs.input_row_offsets_i64 is not None
    assert inputs.input_row_offsets_i64.to_numpy().tolist() == [0, 3, 4]
    # The leading per-request extras, in ABI order; the static tail (the
    # token map and the decoder row extent) follows them.
    _swa_begin, _floor, mask, lookback, _dead, *_tail = inputs.extras
    assert mask.to_numpy().shape == (4,)
    assert lookback.to_numpy().shape == (2, 3)


def _model(
    dp: int,
    ep_size: int,
    n_devices: int = 4,
    *,
    chunked_prefill: bool = True,
) -> DeepseekV41Model:
    model = object.__new__(DeepseekV41Model)
    pipeline_config = Mock()
    pipeline_config.runtime.ep_size = ep_size
    pipeline_config.runtime.max_batch_input_tokens = 512
    pipeline_config.runtime.enable_chunked_prefill = chunked_prefill
    pipeline_config.model.data_parallel_degree = dp
    pipeline_config.model.device_specs = [Mock() for _ in range(n_devices)]
    model.pipeline_config = pipeline_config
    model.devices = [Mock() for _ in range(n_devices)]
    return model


def test_model_builds_the_checkpoint_ep_config() -> None:
    model = _model(dp=4, ep_size=4)
    config = _model_config()
    ep_config = model._build_ep_config(config, n_devices=4)
    assert ep_config.dispatch_dtype == DType.uint8
    assert ep_config.combine_dtype == DType.bfloat16
    assert ep_config.hidden_size == config.hidden_size
    assert ep_config.n_experts == config.n_routed_experts
    assert ep_config.top_k == config.num_experts_per_tok
    assert ep_config.moe_dim == config.moe_intermediate_size
    assert ep_config.n_gpus_per_node == 4
    assert not ep_config.use_allreduce
    assert ep_config.dispatch_quant_config is not None
    assert ep_config.dispatch_quant_config.is_nvfp4


def test_model_refuses_multi_device_without_dp_attention() -> None:
    # The arch registration lets every multi-GPU layout through, but the
    # trunk wires one: plain TP would die deep in graph staging, so the
    # runtime must refuse it before any weights load.
    model = _model(dp=1, ep_size=1)
    with pytest.raises(ValueError, match="only as"):
        model._init_distributed_runtime(_model_config())


def test_model_refuses_multi_device_partial_dp() -> None:
    # A proper divisor still refuses: the module runs TP=1 attention per
    # rank, so every device is a DP replica.
    model = _model(dp=2, ep_size=4)
    with pytest.raises(ValueError, match="only as"):
        model._init_distributed_runtime(_model_config())


def test_model_refuses_multi_device_partial_ep() -> None:
    model = _model(dp=4, ep_size=2)
    with pytest.raises(ValueError, match="only as"):
        model._init_distributed_runtime(_model_config())


def test_model_ep_runtime_requires_ep_size_equal_devices() -> None:
    model = _model(dp=1, ep_size=2, n_devices=1)
    with pytest.raises(ValueError, match="must equal the device count"):
        model._init_distributed_runtime(_model_config())


def test_model_ep_runtime_requires_chunked_prefill() -> None:
    # Without chunking the batch token budget is a soft limit, so a
    # prompt longer than max_batch_input_tokens overflows the EP dispatch
    # buffers, which are sized to it.
    model = _model(dp=4, ep_size=4, chunked_prefill=False)
    with pytest.raises(ValueError, match="requires chunked prefill"):
        model._init_distributed_runtime(_model_config())


def test_model_ep_runtime_binds_the_batch_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The export path (virtual devices) stages the wiring the engine
    # replays: the batch manager and its input tail, with the symmetric
    # heap left to the engine's handshake.
    monkeypatch.setattr(
        "max.pipelines.architectures.deepseekV41_modulev3.model"
        ".is_virtual_device_mode",
        lambda: True,
    )
    model = _model(dp=4, ep_size=4)
    model._init_distributed_runtime(_model_config())
    manager = model._ep_batch_manager
    assert isinstance(manager, EPBatchManager)
    # The EP tail the compile input list appends after the extras: the
    # per-device counter and symmetric-heap buffers.
    assert model._modulev3_extra_input_types == manager.input_types()
    assert model._modulev3_extra_input_types
    assert model.ep_comm_initializer is None


def test_batch_processor_binds_the_ep_tail() -> None:
    """The processor carries the initializer's buffers as the input tail.

    The model's load hands the ``EPCommInitializer`` to the processor
    (the base ``load_graphs`` binding), and every forward's inputs must
    end with the buffers it produced, in ``EPBatchManager.input_types``
    order -- or the in-process forward fails on input arity.
    """
    processor = _batch_processor(dp=4)
    assert processor._ep_inputs() == ()
    initializer = Mock()
    buffers = [Mock(), Mock()]
    initializer.model_inputs.return_value = buffers
    processor.bind_ep_comm_initializer(initializer)
    assert processor._ep_inputs() == tuple(buffers)
    # The no-EP path unbinds: an unbound processor stages no tail.
    processor.bind_ep_comm_initializer(None)
    assert processor._ep_inputs() == ()


def test_model_declares_p2p_dispatch_layout() -> None:
    # gen_mef's EP backend check reads this: with DP attention each rank's
    # dispatch tokens are its own rows, which only the P2P backend serves.
    assert DeepseekV41Model.ep_dispatch_token_layout == "tp_sharded"


def test_module_dp_degree_must_match_mesh() -> None:
    # The guards under test fire before any layer is built, so a simulated
    # mesh (one CPU device in every slot) is enough.
    config = _model_config()
    kv = build_kv_params(
        _interface_config(),
        KVCacheConfig(kv_cache_page_size=128),
        [DeviceRef.CPU()],
    )
    mesh = DeviceMesh((CPU(),) * 4, (4,), ("dp",))
    with pytest.raises(ValueError, match="must equal the mesh size"):
        DeepseekV41(
            config,
            kv,
            mesh,
            text_config=text_config(),
            token_map=np.arange(128, dtype=np.int32),
            data_parallel_degree=2,
            ep_batch_manager=_ep_manager(),
        )


def test_module_requires_ep_with_dp() -> None:
    config = _model_config()
    kv = build_kv_params(
        _interface_config(),
        KVCacheConfig(kv_cache_page_size=128),
        [DeviceRef.CPU()],
    )
    mesh = DeviceMesh((CPU(),) * 4, (4,), ("dp",))
    with pytest.raises(ValueError, match="expert-parallel"):
        DeepseekV41(
            config,
            kv,
            mesh,
            text_config=text_config(),
            token_map=np.arange(128, dtype=np.int32),
            data_parallel_degree=4,
            ep_batch_manager=None,
        )


def test_module_dp_serves_next_token_logits_only() -> None:
    config = _model_config(return_logits=ReturnLogits.ALL)
    kv = build_kv_params(
        _interface_config(),
        KVCacheConfig(kv_cache_page_size=128),
        [DeviceRef.CPU()],
    )
    mesh = DeviceMesh((CPU(),) * 4, (4,), ("dp",))
    with pytest.raises(ValueError, match="next-token logits only"):
        DeepseekV41(
            config,
            kv,
            mesh,
            text_config=text_config(),
            token_map=np.arange(128, dtype=np.int32),
            data_parallel_degree=4,
            ep_batch_manager=_ep_manager(),
        )


def test_routed_experts_leaf_declares_the_dispatch_scale() -> None:
    """The module leaf's consumer contract: one ``[1]`` fp32 shard per rank.

    ``ep_forward`` reads the leaf's per-rank shards and broadcasts each to
    ``[num_experts]``; the dispatch kernel divides it in. A leaf of any
    other shape or dtype would change that expansion, so the module
    declares it ``[1]`` fp32 with one shard per rank in mesh order -- the
    derived ``ffn.global_input_scale`` weight loads into.
    """
    mesh = DeviceMesh((CPU(),) * 2, (2,), ("tp",))
    with F.lazy(), default_device(mesh):
        experts = RoutedExperts(64, 64, 8, mesh=mesh)
    leaf = experts.global_input_scale
    assert leaf.dtype == DType.float32
    assert [int(d) for d in leaf.shape] == [1]
    shards = list(leaf.local_shards)
    assert len(shards) == mesh.num_devices
    expanded = F.broadcast_to(shards[0], [experts.num_experts])
    assert [int(d) for d in expanded.shape] == [experts.num_experts]
    assert expanded.dtype == DType.float32


def _int64(values: list[int]) -> Tensor:
    """A host int64 tensor, as the batch processor stages the DP pair."""
    return Tensor(values, dtype=DType.int64, device=CPU())


def _estimate(total_rows: Tensor, top_k: int, n_gpus_per_node: int) -> int:
    """The expression ``ep_forward`` runs, on the host-resident count."""
    return int(
        (total_rows * top_k // n_gpus_per_node).cast(DType.uint32).item()
    )


@pytest.mark.parametrize("seed", range(8))
def test_host_rows_equal_the_per_rank_sum(seed: int) -> None:
    """The host-resident row count is the sum the graph would compute.

    Randomized ragged batches over a DP4 split: the per-rank row counts
    (each rank's slice of the offsets) sum to the last request's end row,
    which is what ``_host_total_rows`` reads off the CPU pair. Any
    divergence here would mis-size every grouped GEMM the estimate feeds.
    """
    rng = random.Random(seed)
    dp = 4
    n_req = rng.randint(dp, 40)
    interior = sorted(rng.sample(range(1, n_req), dp - 1))
    splits = [0, *interior, n_req]
    bounds = [0]
    for _ in range(n_req):
        bounds.append(bounds[-1] + rng.randint(0, 64))

    host = _host_total_rows_op(_int64(splits), _int64(bounds), last=dp)
    per_rank = sum(bounds[splits[r + 1]] - bounds[splits[r]] for r in range(dp))
    assert int(host.item()) == per_rank == bounds[-1]


@pytest.mark.parametrize(
    ("requests_per_rank", "expected_estimate"),
    [(16, 576), (24, 864)],
)
def test_estimate_over_uniform_batches(
    requests_per_rank: int, expected_estimate: int
) -> None:
    """The estimate over a uniform batch: total rows times ``top_k``.

    With every request the same width, the estimate lands at
    ``M * top_k`` per rank -- the average received pairs, which is what
    the tile-regime selection keys on, not a receive bound.
    """
    dp, rows_per_request, top_k = 4, 6, 6
    bounds = [rows_per_request * i for i in range(dp * requests_per_rank + 1)]
    splits = [requests_per_rank * rank for rank in range(dp + 1)]
    total_rows = _host_total_rows_op(_int64(splits), _int64(bounds), last=dp)
    assert _estimate(total_rows, top_k, dp) == expected_estimate


def test_estimate_tracks_the_live_row_count() -> None:
    """The estimate follows the live count the host staged, not a bucket.

    A step with 9 live requests on one rank and 24 on the rest: the
    estimate reads the staged total, so it stays exact for as long as the
    engine stages live counts.
    """
    dp, top_k, width = 4, 6, 6
    live = [9, 24, 24, 24]
    splits, bounds = [0], [0]
    for count in live:
        for _ in range(count):
            bounds.append(bounds[-1] + width)
        splits.append(splits[-1] + count)
    total_rows = _host_total_rows_op(_int64(splits), _int64(bounds), last=dp)
    live_rows = sum(count * width for count in live)
    assert int(total_rows.item()) == live_rows
    assert _estimate(total_rows, top_k, dp) == live_rows * top_k // dp


if __name__ == "__main__":
    raise SystemExit("run via bazel or pytest")
