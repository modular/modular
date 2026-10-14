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
"""End-to-end integration test for grammar-constrained sampling pipeline.

The graph stitches together every primitive added in DRIV-135/DRIV-136:

  1. A first GPU ``argmax`` over a hand-crafted logits vector to pick a
     token deterministically.
  2. A D2H ``mo.inplace_memcpy`` of that token into pinned host memory.
  3. ``mo.launch_host_func`` runs a Python callback on the GPU stream
     that advances an xgrammar ``XgrammarMatcher`` with the just-arrived
     token and writes the next-step "additive logits mask" (``0.0`` for
     allowed tokens, ``-inf`` for forbidden) into a second pinned buffer.
  4. An H2D ``mo.inplace_memcpy`` of the mask into a GPU buffer.
  5. A second GPU argmax that adds the mask to a fresh logits vector and
     samples the next token.

The grammar is a regex matching exactly ``"ab"``. After consuming token
``"a"``, only token ``"b"`` is allowed. Without the FSM mask the second
logits vector would argmax to ``"c"``; with the mask applied it must argmax
to ``"b"``.
"""

import numpy as np
import pytest
from max import driver
from max._xgrammar import (
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    VocabType,
)
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    Usage,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, ops
from max.nn import kernels
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    XgrammarBackend,
    XgrammarMatcher,
)

# Tiny ASCII vocabulary keeps the test self-contained and deterministic.
#   token 0 -> "a"
#   token 1 -> "b"
#   token 2 -> "c"
#   token 3 -> ""    (EOS)
_VOCAB: list[bytes] = [b"a", b"b", b"c", b""]
_VOCAB_SIZE = len(_VOCAB)
_EOS_TOKEN_ID = 3


def _make_backend() -> XgrammarBackend:
    """Build an XgrammarBackend over the tiny ASCII vocab."""
    tokenizer_info = TokenizerInfo(
        _VOCAB,
        vocab_type=VocabType.RAW,
        vocab_size=_VOCAB_SIZE,
        stop_token_ids=[_EOS_TOKEN_ID],
    )
    compiler = GrammarCompiler(tokenizer_info)
    return XgrammarBackend(compiler)


def test_structured_output_pipeline_e2e() -> None:
    """End-to-end: GPU argmax -> D2H -> FSM update -> H2D -> masked argmax."""
    accelerator = Accelerator()
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("Requires CUDA/HIP accelerator")

    gpu_ref = DeviceRef.from_device(accelerator)
    cpu_ref = DeviceRef.CPU()

    # Build xgrammar backend and compile a regex grammar that matches "ab".
    backend = _make_backend()
    compiled = backend._compiler.compile_regex("ab")
    matcher = XgrammarMatcher(GrammarMatcher(compiled))

    # Pinned host buffers for the cross-device transfers. Backed by
    # `Usage.STAGING` (page-locked host memory tied to the GPU)
    # so D2H/H2D copies can be properly asynchronous and `to_numpy()`
    # exposes a zero-copy host view.
    token_pinned = Buffer(
        dtype=DType.int64, shape=[1], device=accelerator, usage=Usage.STAGING
    )
    mask_pinned = Buffer(
        dtype=DType.float32,
        shape=[_VOCAB_SIZE],
        device=accelerator,
        usage=Usage.STAGING,
    )
    token_pinned_view = token_pinned.to_numpy()
    mask_pinned_view = mask_pinned.to_numpy()
    token_pinned_view[0] = -1
    mask_pinned_view[:] = -1.0

    # Host callback: read the just-arrived D2H token, advance the FSM,
    # convert the bitmask into an additive logits mask (0.0 allowed,
    # -inf forbidden), and write it into mask_pinned in place.
    def host_callback() -> None:
        token = int(token_pinned_view[0])
        matcher.try_consume_tokens([token])
        # Fill the packed int32 bitmask, then unpack it to a float additive mask.
        bitmask = backend.allocate_token_bitmask(1, _VOCAB_SIZE)
        backend.fill_next_token_bitmask(matcher, bitmask, index=0)
        additive = np.array(
            [
                0.0 if (int(bitmask[0, t // 32]) >> (t % 32)) & 1 else -np.inf
                for t in range(_VOCAB_SIZE)
            ],
            dtype=np.float32,
        )
        np.copyto(mask_pinned_view, additive)

    trampoline_ptr, user_data_ptr = driver.__unsafe_pack_py_host_func(
        host_callback
    )

    # Build the graph.
    with Graph(
        "structured_output_pipeline",
        input_types=[
            BufferType(DType.int64, [1], device=cpu_ref),
            BufferType(DType.float32, [_VOCAB_SIZE], device=cpu_ref),
            BufferType(DType.float32, [_VOCAB_SIZE], device=gpu_ref),
            BufferType(DType.int64, [2], device=cpu_ref),
        ],
    ) as graph:
        token_pinned_in = graph.inputs[0].buffer
        mask_pinned_in = graph.inputs[1].buffer
        mask_gpu_in = graph.inputs[2].buffer
        payload_in = graph.inputs[3].buffer

        # 1. Generate token1 from a constant logits vector. Logits favor
        #    token 0 ("a") - the only token allowed at the start of "ab".
        logits1 = ops.constant(
            np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32),
            dtype=DType.float32,
            device=gpu_ref,
        )
        token1 = ops.argmax(logits1, axis=-1)  # shape [1], int64, GPU

        # 2. D2H copy of token1 into pinned CPU memory.
        kernels.inplace_memcpy(dst=token_pinned_in, src=token1)

        # 3. Host callback: consume the token, write the additive mask
        #    into mask_pinned. Runs as a stream callback after the D2H
        #    copy completes.
        kernels.launch_host_func(payload=payload_in, device=gpu_ref)

        # 4. H2D copy of the mask from pinned CPU memory to GPU.
        mask_cpu_tensor = ops.buffer_load(mask_pinned_in)
        kernels.inplace_memcpy(dst=mask_gpu_in, src=mask_cpu_tensor)

        # 5. Sample the next token. Without the mask, argmax of logits2
        #    would pick token 2 ("c") because of the 5.0 entry; with the
        #    mask applied (only token 1 is allowed) the masked argmax
        #    is forced to token 1 ("b").
        logits2 = ops.constant(
            np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32),
            dtype=DType.float32,
            device=gpu_ref,
        )
        mask_gpu_tensor = ops.buffer_load(mask_gpu_in)
        masked_logits2 = logits2 + mask_gpu_tensor
        token2 = ops.argmax(masked_logits2, axis=-1)
        graph.output(token2)

    session = InferenceSession(devices=[accelerator, CPU()])
    model = session.load(graph)

    # Stage the GPU mask buffer (zero-init; overwritten by H2D) and the
    # host_func payload buffer.
    mask_gpu_buffer = Buffer.from_numpy(
        np.zeros(_VOCAB_SIZE, dtype=np.float32)
    ).to(accelerator)
    payload = Buffer(dtype=DType.int64, shape=[2], device=CPU())
    payload[0] = trampoline_ptr
    payload[1] = user_data_ptr

    [final_token] = model.execute(
        token_pinned, mask_pinned, mask_gpu_buffer, payload
    )
    accelerator.synchronize()

    final_token_np = final_token.to(CPU()).to_numpy()
    assert final_token_np.shape == (1,), final_token_np.shape
    assert final_token_np[0] == 1, (
        f"Expected the FSM-masked argmax to pick token 1 ('b') after "
        f"consuming token 0 ('a'), got {final_token_np[0]}. "
        f"D2H token={token_pinned.to_numpy()[0]}, "
        f"mask={mask_pinned.to_numpy()}"
    )
