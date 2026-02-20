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

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.attention.multi_latent_attention import (
    DataParallelLatentAttentionWithRope,
)
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
    RaggedKVCacheInputs,
)
from max.nn.legacy.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import DeepseekV2Config


def _single_gpu_baseline(
    *,
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    use_prefill: bool,
    prefill_buffer_size: int = 16384,
) -> torch.Tensor:
    """Runs DataParallelLatentAttentionWithRope on a single GPU as a numerical baseline."""
    attention_weights = dict(attention_weights)

    device0 = Accelerator(0)
    session = InferenceSession(devices=[device0])
    session.set_debug_print_options(style=PrintStyle.COMPACT)

    assert config.rope_scaling is not None
    scaling_params = DeepseekYarnRopeScalingParams(
        scaling_factor=config.rope_scaling["factor"],
        original_max_position_embeddings=config.rope_scaling[
            "original_max_position_embeddings"
        ],
        beta_fast=config.rope_scaling["beta_fast"],
        beta_slow=config.rope_scaling["beta_slow"],
        mscale=config.rope_scaling["mscale"],
        mscale_all_dim=config.rope_scaling["mscale_all_dim"],
    )
    rope = DeepseekYarnRotaryEmbedding(
        dim=config.qk_rope_head_dim,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_seq_len=config.max_position_embeddings,
        scaling_params=scaling_params,
    )

    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        num_layers=config.num_hidden_layers,
        n_kv_heads=1,
        head_dim=576,
        cache_strategy="paged",
        devices=[DeviceRef.GPU()],
        page_size=128,
        is_mla=True,
    )

    attn = DataParallelLatentAttentionWithRope(
        rope=rope,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_size=config.hidden_size,
        kv_params=kv_params,
        dtype=DType.bfloat16,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        devices=[DeviceRef.GPU()],
        buffer_size=prefill_buffer_size,
    )
    attn.load_state_dict(attention_weights)

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
    )

    hidden_state_type = TensorType(
        DType.bfloat16, ["total_seq_len", config.hidden_size], DeviceRef.GPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], DeviceRef.GPU()
    )

    def construct() -> Graph:
        with Graph(
            "DataParallelLatentAttentionWithRope_single_baseline",
            input_types=(
                hidden_state_type,
                input_row_offsets_type,
                *kv_params.get_symbolic_inputs()[0],
            ),
        ) as graph:
            hidden_states = graph.inputs[0].tensor
            input_row_offsets = graph.inputs[1].tensor
            kv_collection = PagedCacheValues(
                kv_blocks=graph.inputs[2].buffer,
                cache_lengths=graph.inputs[3].tensor,
                lookup_table=graph.inputs[4].tensor,
                max_lengths=graph.inputs[5].tensor,
            )
            out_list = attn(
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                xs=[hidden_states],
                signal_buffers=[],  # DP parity, unused
                kv_collections=[kv_collection],
                freqs_cis=[rope.freqs_cis],
                input_row_offsets=[input_row_offsets],
            )
            graph.output(out_list[0])
        return graph

    g = construct()
    compiled = session.load(g, weights_registry=attn.state_dict())

    batch_size = 1
    total_tokens = input_tensor.shape[1]
    prompt_lens = [total_tokens] if use_prefill else [1]

    # Reserve 1 request
    batch = []
    ctx = create_text_context(np.empty(prompt_lens[0]))
    kv_manager.claim(ctx.request_id, replica_idx=0)
    kv_manager.alloc(ctx, replica_idx=0, num_steps=1)
    batch.append(ctx)

    # Row offsets on host to avoid GPU __setitem__
    row_off = Buffer(DType.uint32, [batch_size + 1])  # host
    row_off[0] = 0
    row_off[1] = prompt_lens[0]

    if use_prefill:
        kv_inputs = kv_manager.get_runtime_inputs([batch])[0]
        inp = (
            Buffer.from_numpy(input_tensor[0, :, :].view(torch.float16).numpy())
            .view(DType.bfloat16)
            .to(device0)
        )
        out = compiled.execute(inp, row_off.to(device0), *kv_inputs)
        return from_dlpack(out[0]).to(torch.bfloat16).to("cpu")[None, :, :]

    # decode path (one step at a time)
    outs = []
    for tok_idx in range(total_tokens):
        for ctx in batch:
            kv_manager.alloc(ctx, replica_idx=0, num_steps=1)
        kv_inputs = kv_manager.get_runtime_inputs([batch])[0]
        tok = (
            Buffer.from_numpy(
                input_tensor[:, tok_idx, :].view(torch.float16).numpy()
            )
            .view(DType.bfloat16)
            .to(device0)
        )
        out = compiled.execute(tok, row_off.to(device0), *kv_inputs)

        ctx.update(42)
        kv_manager.step([batch])

        outs.append(
            from_dlpack(out[0]).to(torch.bfloat16).to("cpu")[:, None, :]
        )
    return torch.concat(outs, dim=1)


def _build_scaling_and_rope(
    config: DeepseekV2Config,
) -> DeepseekYarnRotaryEmbedding:
    assert config.rope_scaling is not None
    scaling = DeepseekYarnRopeScalingParams(
        scaling_factor=config.rope_scaling["factor"],
        original_max_position_embeddings=config.rope_scaling[
            "original_max_position_embeddings"
        ],
        beta_fast=config.rope_scaling["beta_fast"],
        beta_slow=config.rope_scaling["beta_slow"],
        mscale=config.rope_scaling["mscale"],
        mscale_all_dim=config.rope_scaling["mscale_all_dim"],
    )
    return DeepseekYarnRotaryEmbedding(
        dim=config.qk_rope_head_dim,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_seq_len=config.max_position_embeddings,
        scaling_params=scaling,
    )


def _build_kv_params(config: DeepseekV2Config, dp_degree: int) -> KVCacheParams:
    return KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=576,
        num_layers=config.num_hidden_layers,
        cache_strategy="paged",
        devices=[DeviceRef.GPU(i) for i in range(dp_degree)],
        page_size=128,
        data_parallel_degree=dp_degree,
        is_mla=True,
    )


def _build_dp_attention(
    *,
    config: DeepseekV2Config,
    rope: DeepseekYarnRotaryEmbedding,
    devices_ref: list[DeviceRef],
    attention_weights: dict[str, torch.Tensor],
) -> DataParallelLatentAttentionWithRope:
    attn = DataParallelLatentAttentionWithRope(
        rope=rope,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_size=config.hidden_size,
        kv_params=_build_kv_params(config, dp_degree=len(devices_ref)),
        dtype=DType.bfloat16,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        devices=devices_ref,
        buffer_size=16384,
    )
    attn.load_state_dict(attention_weights)
    return attn


def _build_graph_and_compile(
    *,
    config: DeepseekV2Config,
    session: InferenceSession,
    attn: DataParallelLatentAttentionWithRope,
    rope: DeepseekYarnRotaryEmbedding,
    kv_manager: PagedKVCacheManager,
    devices: list[Accelerator],
) -> tuple:
    """Builds a per-device inputs graph and compiles it."""
    input_types: list[TensorType | BufferType] = []
    for d in devices:
        devref = DeviceRef.from_device(d)
        input_types.append(
            TensorType(
                DType.bfloat16, ["total_seq_len", config.hidden_size], devref
            )
        )  # hidden
        input_types.append(
            TensorType(DType.uint32, ["input_row_offsets_len"], devref)
        )  # offsets

    # KV symbols across all devices
    kv_syms = kv_manager.params.get_symbolic_inputs()
    for tup in kv_syms:
        input_types.extend(tup)

    def construct() -> Graph:
        with Graph(
            "DataParallelLatentAttentionWithRope_MultiGPU",
            input_types=tuple(input_types),
        ) as graph:
            n = len(devices)
            xs = []
            input_row_offsets_list = []
            idx = 0
            for _ in range(n):
                xs.append(graph.inputs[idx].tensor)
                input_row_offsets_list.append(graph.inputs[idx + 1].tensor)
                idx += 2

            kv_collections = []
            # Each device contributes 4 KV inputs: blocks, lengths, lookup, max_lengths
            for i in range(n):
                base = 2 * n + 4 * i
                kv_collections.append(
                    PagedCacheValues(
                        kv_blocks=graph.inputs[base + 0].buffer,
                        cache_lengths=graph.inputs[base + 1].tensor,
                        lookup_table=graph.inputs[base + 2].tensor,
                        max_lengths=graph.inputs[base + 3].tensor,
                    )
                )

            outs = attn(
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                xs=xs,
                signal_buffers=[],  # DP parity, unused
                kv_collections=kv_collections,
                freqs_cis=[rope.freqs_cis for _ in range(n)],
                input_row_offsets=input_row_offsets_list,
            )
            # For comparison we return replica 0's output (others may be identical)
            graph.output(outs[0])
        return graph

    g = construct()
    compiled = session.load(g, weights_registry=attn.state_dict())
    return compiled, g


def _flatten_kv_kv_inputs(fetch_list: Sequence[RaggedKVCacheInputs]) -> list:
    flat: list = []
    for f in fetch_list:
        flat.extend([f.blocks, f.cache_lengths, f.lookup_table, f.max_lengths])
    return flat


def _run_distributed_dp(
    *,
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dp_degree: int,
    use_prefill: bool,
) -> torch.Tensor:
    """Runs DataParallelLatentAttentionWithRope in multi-GPU data-parallel mode and returns replica-0 output."""
    devices = [Accelerator(i) for i in range(dp_degree)]
    devices_ref = [DeviceRef.from_device(d) for d in devices]
    session = InferenceSession(devices=devices)
    session.set_debug_print_options(style=PrintStyle.COMPACT)

    rope = _build_scaling_and_rope(config)
    attn = _build_dp_attention(
        config=config,
        rope=rope,
        devices_ref=devices_ref,
        attention_weights=attention_weights,
    )

    kv_manager = PagedKVCacheManager(
        params=_build_kv_params(config, dp_degree),
        total_num_pages=8,
        session=session,
    )

    compiled, _ = _build_graph_and_compile(
        config=config,
        session=session,
        attn=attn,
        rope=rope,
        kv_manager=kv_manager,
        devices=devices,
    )

    total_tokens = input_tensor.shape[1]

    # Claim one request per replica
    batch = []
    seq_len = total_tokens if use_prefill else 1
    for replica_idx in range(dp_degree):
        ctx = create_text_context(np.empty(seq_len))
        kv_manager.claim(ctx.request_id, replica_idx=replica_idx)
        kv_manager.alloc(ctx, replica_idx=replica_idx, num_steps=1)
        batch.append(ctx)
    batches_by_replica = [[ctx] for ctx in batch]

    if use_prefill:
        # Single execute covering full prompt lengths (identical prompt on each replica).
        fetch_list = kv_manager.get_runtime_inputs(batches_by_replica)
        kv_args = _flatten_kv_kv_inputs(fetch_list)

        # Per-device inputs: full sequence + [0, T] row offsets (built on host)
        args = []
        for dev in devices:
            inp = (
                Buffer.from_numpy(
                    input_tensor[0, :, :].view(torch.float16).numpy()
                )
                .view(DType.bfloat16)
                .to(dev)
            )
            ro_host = Buffer(DType.uint32, [2])  # host
            ro_host[0] = 0
            ro_host[1] = total_tokens
            ro = ro_host.to(dev)
            args.extend([inp, ro])

        out = compiled.execute(*args, *kv_args)
        return from_dlpack(out[0]).to(torch.bfloat16).to("cpu")[None, :, :]

    # decode: loop tokens, one step per execute
    outs = []
    for tok_idx in range(total_tokens):
        for ctx in batch:
            kv_manager.alloc(ctx, replica_idx=replica_idx, num_steps=1)
        fetch_list = kv_manager.get_runtime_inputs(batches_by_replica)
        kv_args = _flatten_kv_kv_inputs(fetch_list)

        step_args = []
        for dev in devices:
            tok_np = input_tensor[:, tok_idx, :].view(torch.float16).numpy()
            h = Buffer.from_numpy(tok_np).view(DType.bfloat16).to(dev)

            ro_host = Buffer(DType.uint32, [2])  # host
            ro_host[0] = 0
            ro_host[1] = 1
            ro = ro_host.to(dev)

            step_args.extend([h, ro])

        out = compiled.execute(*step_args, *kv_args)

        # Advance contexts
        for ctx in batch:
            ctx.update(42)
        kv_manager.step(batches_by_replica)

        outs.append(
            from_dlpack(out[0]).to(torch.bfloat16).to("cpu")[:, None, :]
        )

    return torch.concat(outs, dim=1)


# -----------------
# Actual test cases
# -----------------


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_data_parallel_latent_attention_decode_matches_single_multi(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,  # unused; parity with other tests
    attention_weights: dict[str, torch.Tensor],
) -> None:
    single = _single_gpu_baseline(
        config=config,
        input_tensor=input_tensor,
        attention_weights=attention_weights,
        use_prefill=False,
    )
    dp_multi = _run_distributed_dp(
        config=config,
        input_tensor=input_tensor,
        attention_weights=attention_weights,
        dp_degree=4,
        use_prefill=False,
    )
    torch.testing.assert_close(single, dp_multi, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_data_parallel_latent_attention_prefill_matches_single_multi(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,  # unused; parity with other tests
    attention_weights: dict[str, torch.Tensor],
) -> None:
    single = _single_gpu_baseline(
        config=config,
        input_tensor=input_tensor,
        attention_weights=attention_weights,
        use_prefill=True,
    )
    dp_multi = _run_distributed_dp(
        config=config,
        input_tensor=input_tensor,
        attention_weights=attention_weights,
        dp_degree=4,
        use_prefill=True,
    )
    torch.testing.assert_close(single, dp_multi, rtol=5e-4, atol=5e-4)
