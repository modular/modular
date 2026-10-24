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
"""Runs MAX's MiMo-V2 on a checkpoint directory, teacher-forced.

Shared by ``check_reference_fixtures.py``, which scores the real checkpoint
and the production-width fixtures by hand, and by
``test_serving_numerics_gpu.py``, which scores the tiny fixture in CI.
"""

from __future__ import annotations

import dataclasses
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import CPU, Accelerator, Buffer, batch_inplace_copy
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import SafetensorWeights
from max.nn.comm import Signals
from max.nn.kv_cache import (
    BatchCharacteristics,
    MHAKVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.mimo_v2.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2.model_config import (
    FULL,
    SLIDING,
    MiMoV2Config,
    attention_head_dim,
    layer_types,
)
from max.pipelines.architectures.mimo_v2.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from max.pipelines.modeling.types import RequestID
from transformers.configuration_utils import PretrainedConfig

SABOTAGES = ("window_129", "bias_bf16")
"""Defects :class:`Runner` can build in, for checking that a gate fails."""


def kv_params(
    hf: Any, devices: list[DeviceRef], page_size: int
) -> MultiKVCacheParams:
    """Returns the ``{sliding_attention, full_attention}`` KV tree."""
    groups = layer_types(hf)

    def group(
        n_kv_heads: int, num_layers: int, window: int | None
    ) -> MHAKVCacheParams:
        return MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=n_kv_heads,
            head_dim=attention_head_dim(hf),
            num_layers=num_layers,
            devices=devices,
            page_size=page_size,
            window_size=window,
        )

    return MultiKVCacheParams.from_params(
        {
            SLIDING: group(
                hf.swa_num_key_value_heads,
                groups.count(SLIDING),
                hf.sliding_window,
            ),
            FULL: group(hf.num_key_value_heads, groups.count(FULL), None),
        }
    )


@dataclasses.dataclass
class _CapturedDecode:
    keys: list[int]
    inputs: list[Buffer]
    outputs: list[Buffer]


class Runner:
    """A compiled MAX MiMo-V2 that returns every layer's output and logit.

    With ``isolated``, the graph instead takes one hidden-state input per
    layer and runs each layer on its own input, so a layer can be fed the
    gold output of the layer before it and checked on its own.
    """

    def __init__(
        self,
        checkpoint: Path,
        num_devices: int,
        max_seq_len: int,
        isolated: bool = False,
        sabotage: str | None = None,
        kv_cache_bytes: int = 16 * 1024**3,
        fused_sliding_qkv: bool = False,
        fused_full_qkv: bool = False,
    ) -> None:
        """Adapts the checkpoint, then compiles and loads the model.

        Args:
            checkpoint: A directory with ``config.json`` and Xiaomi-format
                safetensors.
            num_devices: The GPUs to shard over.
            max_seq_len: The longest sequence to run.
            isolated: Whether to run each layer on its own input.
            sabotage: One of :data:`SABOTAGES` to build in, or ``None``.
            kv_cache_bytes: The paged KV cache's size per device.
            fused_sliding_qkv: Whether sliding layers use fused QKV preparation.
            fused_full_qkv: Whether full layers use fused QKV preparation.
        """
        self.capture_steps = 0
        self.replay_steps = 0
        self.capture_replays_by_batch_size: dict[int, int] = {}
        self.replay_cache_lengths_by_batch_size: dict[
            int, list[tuple[int, ...]]
        ] = {}
        self._decode_cache_bound = 0
        self._captured_decode: dict[int, _CapturedDecode] = {}
        self.isolated = isolated
        self.hf = PretrainedConfig(
            **json.loads((checkpoint / "config.json").read_text())
        )
        self.devices = [Accelerator(i) for i in range(num_devices)]
        refs = [DeviceRef.GPU(i) for i in range(num_devices)]
        self.kv_params = kv_params(self.hf, refs, 128)
        config = MiMoV2Config.from_huggingface_config(
            self.hf,
            devices=refs,
            kv_params=self.kv_params,
            max_seq_len=max_seq_len,
        )
        config.fused_sliding_qkv = fused_sliding_qkv
        config.fused_full_qkv = fused_full_qkv
        config.return_logits = ReturnLogits.ALL
        config.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        config.target_layer_ids = list(range(self.hf.num_hidden_layers))
        if sabotage == "window_129":
            config.sliding_window += 1
        self.num_layers = self.hf.num_hidden_layers

        t0 = time.time()
        weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
        state = convert_safetensor_state_dict(dict(weights.items()))
        print(f"adapted {len(state)} tensors in {time.time() - t0:.0f} s")
        if sabotage == "bias_bf16":
            for name in state:
                if name.endswith("e_score_correction_bias"):
                    bias = np.from_dlpack(state[name].data).astype(np.float32)
                    rounded = (
                        bf16_bits(bias)
                        .view(ml_dtypes.bfloat16)
                        .astype(np.float32)
                    )
                    state[name] = dataclasses.replace(
                        state[name], data=Buffer.from_numpy(rounded)
                    )

        model = MiMoV2(config)
        model.load_state_dict(state, weight_alignment=1, strict=True)
        self.registry = model.state_dict(auto_initialize=False)
        n = num_devices
        feeds = (
            [
                TensorType(
                    DType.bfloat16,
                    ["total_seq_len", self.hf.hidden_size],
                    device=refs[0],
                )
                for _ in range(self.num_layers)
            ]
            if isolated
            else [
                TensorType(DType.int64, ["total_seq_len"], device=refs[0]),
                TensorType(
                    DType.int64, ["return_n_logits"], device=DeviceRef.CPU()
                ),
            ]
        )
        input_types = [
            *feeds,
            *(
                TensorType(DType.uint32, ["input_row_offsets_len"], device=r)
                for r in refs
            ),
            *Signals(devices=refs).input_types(),
            *tree.leaves(self.kv_params.get_symbolic_inputs()),
        ]
        with Graph("mimo_v2_fixture_check", input_types=input_types) as graph:
            inputs = graph.inputs[: len(feeds)]
            rest = graph.inputs[len(feeds) :]
            offsets = [v.tensor for v in rest[:n]]
            signal_buffers = [v.buffer for v in rest[n : 2 * n]]
            sliding, full = self.kv_params.unflatten_basic_kv_tree(
                iter(rest[2 * n :])
            )
            if isolated:
                kv = {SLIDING: sliding, FULL: full}
                freqs = {
                    group: [
                        ops.cast(rope.freqs_cis, DType.bfloat16).to(r)
                        for r in refs
                    ]
                    for group, rope in model.ropes.items()
                }
                graph.output(
                    [
                        layer(
                            ops.constant(
                                model.kv_layer_indices[i],
                                DType.uint32,
                                device=DeviceRef.CPU(),
                            ),
                            [inputs[i].tensor.to(r) for r in refs],
                            signal_buffers,
                            kv[config.layer_types[i]],
                            freqs[config.layer_types[i]],
                            offsets,
                        )[0]
                        for i, layer in enumerate(model.layers)
                    ]
                )
            else:
                graph.output(
                    model(
                        tokens=inputs[0].tensor,
                        signal_buffers=signal_buffers,
                        sliding_kv_collections=sliding,
                        full_kv_collections=full,
                        return_n_logits=inputs[1].tensor,
                        input_row_offsets=offsets,
                    )
                )
        self.fused_qkv_count = graph.module.mlir_module.asm().count(
            "mo.chunked_qkv_rope_store.ragged.paged"
        )
        t0 = time.time()
        session = InferenceSession(devices=self.devices)
        self.model = session.load(graph, weights_registry=self.registry)
        print(f"compiled and loaded in {time.time() - t0:.0f} s")
        self.signals = Signals.allocate(self.devices)
        self.kv = JengaKVCacheManager.create(
            params=self.kv_params,
            available_bytes=kv_cache_bytes * n,
            max_batch_size=16,
            max_seq_len=max_seq_len,
        )

    def _execute(
        self,
        batch: list[TextContext],
        layer_inputs: list[list[npt.NDArray[np.float32]]] | None,
        device_graph_capture: bool = False,
    ) -> tuple[npt.NDArray[np.float32] | None, list[npt.NDArray[np.float32]]]:
        for ctx in batch:
            self.kv.alloc(ctx)
        sizes = [ctx.tokens.active_length for ctx in batch]
        capture_decode = device_graph_capture and all(
            size == 1 for size in sizes
        )
        # Match production capture: host dispatch bounds and LUT width stay
        # fixed while token IDs, GPU cache lengths and LUT contents change.
        # Real per-step host maxima would change attention's launch trace.
        kv_inputs = self.kv.runtime_inputs(
            [batch],
            max_cache_length=self._decode_cache_bound
            if capture_decode
            else None,
            batch_characteristics=(
                BatchCharacteristics(
                    batch_size=len(batch),
                    max_prompt_length=1,
                    max_cache_valid_length=self._decode_cache_bound,
                )
                if capture_decode
                else None
            ),
        )
        offsets = Buffer.from_numpy(np.cumsum([0] + sizes).astype(np.uint32))
        if layer_inputs is None:
            feeds = [
                Buffer.from_numpy(
                    np.concatenate(
                        [np.asarray(c.tokens.active, np.int64) for c in batch]
                    )
                ).to(self.devices[0]),
                Buffer.from_numpy(np.array([1], dtype=np.int64)),
            ]
        else:
            # The gold rows each context processes this step, per layer.
            feeds = [
                f32_to_bf16(
                    np.concatenate(
                        [
                            layer_inputs[j][i][
                                c.tokens.processed_length : c.tokens.processed_length
                                + c.tokens.active_length
                            ]
                            for j, c in enumerate(batch)
                        ]
                    )
                ).to(self.devices[0])
                for i in range(self.num_layers)
            ]
        inputs = [
            *feeds,
            *(offsets.to(d) for d in self.devices),
            *self.signals,
            *tree.leaves(kv_inputs),
        ]
        if capture_decode:
            # Cache append overwrites the same positions until kv.step, so
            # capture and trace verification leave the teacher history intact.
            batch_size = len(batch)
            captured = self._captured_decode.get(batch_size)
            if captured is None:
                keys = [631 + batch_size] * len(self.devices)
                captured = _CapturedDecode(
                    keys, inputs, self.model.capture(keys, *inputs)
                )
                self._captured_decode[batch_size] = captured
                self.capture_steps += 1
                self.capture_replays_by_batch_size[batch_size] = 0
            else:
                assert [(value.shape, value.dtype) for value in inputs] == [
                    (value.shape, value.dtype) for value in captured.inputs
                ], (
                    "Decode inputs changed signature within one batch-size bucket"
                )
                dsts: list[Buffer] = []
                srcs: list[Buffer] = []
                for index, (live, stable) in enumerate(
                    zip(inputs, captured.inputs, strict=True)
                ):
                    if stable.device.is_host or stable.pinned:
                        np.testing.assert_array_equal(
                            stable.to_numpy(),
                            live.to_numpy(),
                            err_msg=f"Captured host input {index} changed",
                        )
                        continue
                    assert stable.device == live.device
                    dsts.append(stable)
                    srcs.append(live)
                # Replay binds the captured GPU addresses, as production does.
                # Refresh their values; shared cache/signal aliases are skipped.
                batch_inplace_copy(dsts, srcs)
            self.model.debug_verify_replay(captured.keys, *inputs)
            self.model.replay(captured.keys, *captured.inputs)
            self.replay_steps += 1
            self.capture_replays_by_batch_size[batch_size] += 1
            self.replay_cache_lengths_by_batch_size.setdefault(
                batch_size, []
            ).append(tuple(ctx.tokens.processed_length for ctx in batch))
            out = captured.outputs
            if layer_inputs is not None:
                return None, [bf16_to_f32(o) for o in out]
            logits = out[1].to(CPU()).to_numpy().copy()
            hidden = [bf16_to_f32(out[3 + i]) for i in range(self.num_layers)]
            return logits, hidden
        out = self.model.execute(*inputs)
        if layer_inputs is not None:
            return None, [bf16_to_f32(o) for o in out]
        # (last-token logits, all logits, offsets, one capture per layer per
        # device); every device holds the same captures.
        logits = out[1].to(CPU()).to_numpy()
        hidden = [bf16_to_f32(out[3 + i]) for i in range(self.num_layers)]
        return logits, hidden

    def teacher_forced(
        self,
        seqs: list[npt.NDArray[np.int64]],
        chunk: int,
        decode: int,
        layer_inputs: list[list[npt.NDArray[np.float32]]] | None = None,
        device_graph_capture: bool = False,
    ) -> list[
        tuple[npt.NDArray[np.float32] | None, list[npt.NDArray[np.float32]]]
    ]:
        """Returns each sequence's ``[T, vocab]`` logits and layer outputs.

        The sequences run as one ragged batch. Each one's ``seq[:-decode]``
        is prefilled in chunks of at most ``chunk`` tokens, and its last
        ``decode`` tokens are fed one decode step at a time. For an isolated
        runner, ``layer_inputs[i][k]`` is sequence ``i``'s ``[T, hidden]``
        input to layer ``k``. With ``device_graph_capture``, each decode batch
        size is captured once; later steps verify and replay that graph with
        new token inputs and cache metadata before their logits are scored.
        """
        self.capture_steps = 0
        self.replay_steps = 0
        self.capture_replays_by_batch_size.clear()
        self.replay_cache_lengths_by_batch_size.clear()
        self._decode_cache_bound = max((len(seq) for seq in seqs), default=0)
        ctxs = [
            TextContext(
                request_id=RequestID(),
                max_length=len(seq) + 1,
                tokens=TokenBuffer(
                    np.ascontiguousarray(seq[: len(seq) - decode])
                ),
            )
            for seq in seqs
        ]
        rows: list[list[npt.NDArray[np.float32]]] = [[] for _ in seqs]
        hidden: list[list[list[npt.NDArray[np.float32]]]] = [[] for _ in seqs]
        for ctx in ctxs:
            self.kv.claim(ctx)
        live = list(range(len(seqs)))
        try:
            while live:
                chunked = {}
                for i in live:
                    tokens = ctxs[i].tokens
                    chunked[i] = tokens.active_length > chunk
                    if chunked[i]:
                        tokens.chunk(chunk)
                sizes = [ctxs[i].tokens.active_length for i in live]
                logits, h = self._execute(
                    [ctxs[i] for i in live],
                    None
                    if layer_inputs is None
                    else [layer_inputs[i] for i in live],
                    device_graph_capture=device_graph_capture,
                )
                bounds = np.cumsum([0] + sizes)
                for j, i in enumerate(live):
                    if logits is not None:
                        rows[i].append(logits[bounds[j] : bounds[j + 1]])
                    hidden[i].append([x[bounds[j] : bounds[j + 1]] for x in h])
                still = []
                for i in live:
                    tokens, seq = ctxs[i].tokens, seqs[i]
                    done = tokens.processed_length + tokens.active_length
                    if chunked[i]:
                        tokens.advance_chunk()
                    elif done < len(seq):
                        tokens.advance_with_token(int(seq[done]))
                    else:
                        continue
                    self.kv.step(ctxs[i])
                    still.append(i)
                live = still
        finally:
            try:
                with ExitStack() as cleanup:
                    for ctx in ctxs:
                        cleanup.callback(self.kv.release, ctx)
                    for captured in self._captured_decode.values():
                        cleanup.callback(
                            self.model.release_captured_graph, captured.keys
                        )
            finally:
                self._captured_decode.clear()
        return [
            (
                np.concatenate(rows[i]) if rows[i] else None,
                [
                    np.concatenate([step[k] for step in hidden[i]])
                    for k in range(self.num_layers)
                ],
            )
            for i in range(len(seqs))
        ]


def bf16_to_f32(buffer: Any) -> npt.NDArray[np.float32]:
    """Copies a BF16 or F32 device buffer to the host as float32."""
    host = buffer.to(CPU())
    if host.dtype == DType.float32:
        return host.to_numpy()
    return (
        host.view(DType.uint16)
        .to_numpy()
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


def bf16_bits(x: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """Rounds to nearest even, as a float32 -> bfloat16 cast does."""
    return (
        np.asarray(x, dtype=np.float32)
        .astype(ml_dtypes.bfloat16)
        .view(np.uint16)
    )


def f32_to_bf16(x: npt.NDArray[np.float32]) -> Buffer:
    """Rounds ``x`` to a BF16 host buffer."""
    return Buffer.from_numpy(bf16_bits(x)).view(DType.bfloat16)
