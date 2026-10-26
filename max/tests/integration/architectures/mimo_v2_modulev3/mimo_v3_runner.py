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
"""Runs MAX's ModuleV3 MiMo-V2 port on a checkpoint directory, teacher-forced.

The ModuleV3 counterpart of ``../mimo_v2/mimo_runner.py``, which the V2-vs-V3
parity test runs on the same fixture. The model builds under ``F.lazy()`` on
a ``DeviceMesh`` and compiles through ``Module.compile`` instead of a
hand-built graph-API ``Graph``; it takes no signal buffers (the residual
reduces through ``Tensor.to(mapping)``), and returns no per-layer hidden
states (a speculative-decode concern this port refuses), so the runner
scores logits only. The teacher-forced loop is the same discipline as the
V2 runner's: chunked ragged prefill, one-token decode, through the shared
paged KV cache.

Used by ``test_serving_numerics_gpu.py`` (V3 vs HF goldens) and
``test_v2_v3_parity_gpu.py`` (V3 vs V2).
"""

from __future__ import annotations

import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import compilation
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    FULL,
    SLIDING,
    MiMoV2Config,
    attention_head_dim,
    layer_types,
)
from max.pipelines.architectures.mimo_v2_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.modeling.types import RequestID
from transformers.configuration_utils import PretrainedConfig


def kv_params(
    hf: PretrainedConfig, devices: list[DeviceRef], page_size: int
) -> MultiKVCacheParams:
    """Returns the ``{sliding_attention, full_attention}`` KV tree.

    Shared with the V2 runner's ``kv_params`` (the checkpoint format does not
    differ between the ports), written here against this package's config
    helpers.
    """
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


class Runner:
    """A compiled ModuleV3 MiMo-V2 that returns every row's logits.

    The sequences run as one ragged batch: each one's ``seq[:-decode]`` is
    prefilled in chunks of at most ``chunk`` tokens, and its last ``decode``
    tokens are fed one decode step at a time.
    """

    def __init__(
        self,
        checkpoint: Path,
        num_devices: int,
        max_seq_len: int,
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
            kv_cache_bytes: The paged KV cache's size per device.
            fused_sliding_qkv: Whether sliding layers use fused QKV
                preparation.
            fused_full_qkv: Whether full layers use fused QKV preparation.
        """
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
        config.return_logits = ReturnLogits.ALL
        config.fused_sliding_qkv = fused_sliding_qkv
        config.fused_full_qkv = fused_full_qkv

        t0 = time.time()
        weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
        state = convert_safetensor_state_dict(dict(weights.items()))
        print(f"adapted {len(state)} tensors in {time.time() - t0:.0f} s")

        mesh = DeviceMesh(tuple(self.devices), (num_devices,), ("tp",))
        # The module builds under the same contexts the serving path's
        # ``graph_specs`` gives it; ``compile`` runs outside them, as it does
        # there.
        with F.lazy(), default_device(mesh), default_dtype(config.dtype):
            model = MiMoV2(config)
            # ``Module.compile`` silently ignores a checkpoint entry no
            # parameter matches, so pin the name contract the way the
            # graph-API runner's strict load does.
            names = {name for name, _ in model.parameters}
            assert names == set(state), (
                f"module lacks {sorted(set(state) - names)[:8]}; no parameter "
                f"matches {sorted(names - set(state))[:8]}"
            )

        input_types = modulev3_ragged_kv_symbolic_inputs(
            kv_params=self.kv_params, device_refs=refs
        )
        t0 = time.time()
        session = InferenceSession(devices=self.devices)
        # ``Module.compile`` split in two, to count the fused QKV op sites in
        # the staged graph before it compiles.
        staged, registry = model.stage(
            *input_types,
            weights=state,
            name="mimo_v2_modulev3_fixture_check",
        )
        self.fused_qkv_count = str(staged).count(
            "mo.chunked_qkv_rope_store.ragged.paged"
        )
        self.model = compilation.CompiledCallable(
            staged, registry, session=session
        )
        print(f"compiled and loaded in {time.time() - t0:.0f} s")
        self.kv = JengaKVCacheManager.create(
            params=self.kv_params,
            available_bytes=kv_cache_bytes * num_devices,
            max_batch_size=16,
            max_seq_len=max_seq_len,
        )

    def _execute(self, batch: list[TextContext]) -> npt.NDArray[np.float32]:
        for ctx in batch:
            self.kv.alloc(ctx)
        sizes = [ctx.tokens.active_length for ctx in batch]
        kv_inputs = self.kv.runtime_inputs([batch])
        offsets = Buffer.from_numpy(
            np.cumsum([0] + sizes).astype(np.uint32)
        ).to(self.devices[0])
        tokens = Buffer.from_numpy(
            np.concatenate(
                [np.asarray(c.tokens.active, np.int64) for c in batch]
            )
        ).to(self.devices[0])
        # The graph deletes this input; ALL logits are always returned.
        return_n_logits = Buffer.from_numpy(np.array([1], dtype=np.int64))
        outputs = self.model(
            tokens, return_n_logits, offsets, *tree.leaves(kv_inputs)
        )
        assert outputs.logits is not None
        return outputs.logits.driver_tensor.to(CPU()).to_numpy().copy()

    def teacher_forced(
        self,
        seqs: list[npt.NDArray[np.int64]],
        chunk: int,
        decode: int,
    ) -> list[npt.NDArray[np.float32] | None]:
        """Returns each sequence's ``[T, vocab]`` logits.

        The sequences run as one ragged batch. Each one's ``seq[:-decode]``
        is prefilled in chunks of at most ``chunk`` tokens, and its last
        ``decode`` tokens are fed one decode step at a time, so both the
        chunked-prefill and the one-token paths of the KV cache and attention
        are exercised. The final row of each sequence is the last decoded
        token's, matching the V2 runner's coverage.
        """
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
                logits = self._execute([ctxs[i] for i in live])
                bounds = np.cumsum([0] + sizes)
                for j, i in enumerate(live):
                    rows[i].append(logits[bounds[j] : bounds[j + 1]])
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
            with ExitStack() as cleanup:
                for ctx in ctxs:
                    cleanup.callback(self.kv.release, ctx)
        return [
            np.concatenate(rows[i]) if rows[i] else None
            for i in range(len(seqs))
        ]
