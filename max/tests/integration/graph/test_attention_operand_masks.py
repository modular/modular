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
"""Checks the masks in `nn.attention.operand_masks`, which read their
operands through `load`, against the pointer-based masks they replace.

Each graph runs both on the same inputs; the outputs must match exactly.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.legacy.kernels import (
    flash_attention_ragged_gpu,
    masked_flash_attention_gpu,
)

HEADS = 4
HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5
LENGTHS = [37, 100]
TOKENS = sum(LENGTHS)
BATCH = len(LENGTHS)
MAX_LEN = max(LENGTHS)
DENSE_SEQ = 64

Inputs = dict[str, torch.Tensor | np.ndarray]
Build = Callable[[dict[str, TensorValue]], Sequence[TensorValue]]


def _gpu(dtype: DType, shape: Sequence[int]) -> TensorType:
    return TensorType(dtype, list(shape), device=DeviceRef.GPU())


def _scale() -> TensorValue:
    return ops.constant(SCALE, DType.float32, device=DeviceRef.CPU())


def _ragged_op(
    name: str,
    x: dict[str, TensorValue],
    mask: str,
    operands: Sequence[TensorValue],
    window: int = -1,
) -> TensorValue:
    q = x["q"]
    return ops.custom(
        name,
        device=DeviceRef.GPU(),
        values=[
            q,
            x["k"],
            x["v"],
            x["offsets"],
            x["max_len"],
            _scale(),
            *operands,
        ],
        out_types=[TensorType(q.dtype, q.shape, device=DeviceRef.GPU())],
        parameters={"mask": mask, "window": window},
    )[0].tensor


def _pointer_and_load(
    mask: str,
    operands: Callable[[dict[str, TensorValue]], Sequence[TensorValue]],
    window: int = -1,
) -> Build:
    def build(x: dict[str, TensorValue]) -> Sequence[TensorValue]:
        return [
            _ragged_op(
                "test_ragged_attention_pointer_mask",
                x,
                mask,
                operands(x),
                window,
            ),
            _ragged_op(
                "test_ragged_attention_load_mask", x, mask, operands(x), window
            ),
        ]

    return build


def _segment(x: dict[str, TensorValue]) -> Sequence[TensorValue]:
    return [
        flash_attention_ragged_gpu(
            x["q"],
            x["k"],
            x["v"],
            x["offsets"],
            x["max_len"],
            MHAMaskVariant.NULL_MASK,
            SCALE,
            segment_ids=x["ids"],
        ),
        _ragged_op(
            "test_ragged_attention_load_mask",
            x,
            "segment",
            [x["ids"], x["offsets"]],
        ),
    ]


def _causal(x: dict[str, TensorValue]) -> Sequence[TensorValue]:
    # `CausalMask` takes no mask operands, so the op gets an empty tuple.
    return [
        flash_attention_ragged_gpu(
            x["q"],
            x["k"],
            x["v"],
            x["offsets"],
            x["max_len"],
            MHAMaskVariant.CAUSAL_MASK,
            SCALE,
        ),
        _ragged_op("test_ragged_attention_load_mask", x, "causal", []),
    ]


def _additive_fused(x: dict[str, TensorValue]) -> Sequence[TensorValue]:
    # An elementwise producer the graph compiler fuses into the load mask's
    # operand; the pointer mask reads it materialized.
    computed = x["mask3"] * 0.5
    return [
        _ragged_op(
            "test_ragged_attention_pointer_mask", x, "additive", [computed]
        ),
        _ragged_op(
            "test_ragged_attention_load_mask", x, "additive", [computed]
        ),
    ]


def _relative_logits_bf16_odd(
    x: dict[str, TensorValue],
) -> Sequence[TensorValue]:
    # An odd extent can't read the bf16 bias in pairs, and the pointer mask
    # rejects it, so the pointer mask reads the bias's exact f32 upcast.
    bias = x["bias_bf16_odd"]
    rest = [x["cache_lengths"], x["offsets"]]
    return [
        _ragged_op(
            "test_ragged_attention_pointer_mask",
            x,
            "relative_logits",
            [bias.cast(DType.float32), *rest],
        ),
        _ragged_op(
            "test_ragged_attention_load_mask",
            x,
            "relative_logits",
            [bias, *rest],
        ),
    ]


def _dense(mask_name: str) -> Build:
    def build(x: dict[str, TensorValue]) -> Sequence[TensorValue]:
        q, k, v, mask = x["dq"], x["dk"], x["dv"], x[mask_name]
        load = ops.custom(
            "test_dense_attention_load_mask",
            device=DeviceRef.GPU(),
            values=[q, k, v, _scale(), mask],
            out_types=[TensorType(q.dtype, q.shape, device=DeviceRef.GPU())],
        )[0].tensor
        return [masked_flash_attention_gpu(q, k, v, mask, SCALE), load]

    return build


CASES: dict[str, tuple[list[str], Build]] = {
    "causal_no_operands": ([], _causal),
    "segment": (["ids"], _segment),
    "additive_rank3": (
        ["mask3"],
        _pointer_and_load("additive", lambda x: [x["mask3"]]),
    ),
    "additive_rank4": (
        ["mask4"],
        _pointer_and_load("additive", lambda x: [x["mask4"]]),
    ),
    # More keys than queries in the bias, so its rows shift by the difference.
    "additive_rank3_wide": (
        ["mask3_wide"],
        _pointer_and_load("additive", lambda x: [x["mask3_wide"]]),
    ),
    "additive_fused_producer": (["mask3"], _additive_fused),
    "dense_additive_rank3": (["dq", "dk", "dv", "dmask3"], _dense("dmask3")),
    "dense_additive_rank4": (["dq", "dk", "dv", "dmask4"], _dense("dmask4")),
    "relative_logits_bf16": (
        ["bias_bf16", "cache_lengths"],
        _pointer_and_load(
            "relative_logits",
            lambda x: [x["bias_bf16"], x["cache_lengths"], x["offsets"]],
        ),
    ),
    "relative_logits_bf16_window": (
        ["bias_bf16", "cache_lengths"],
        _pointer_and_load(
            "relative_logits",
            lambda x: [x["bias_bf16"], x["cache_lengths"], x["offsets"]],
            window=32,
        ),
    ),
    "relative_logits_bf16_cached": (
        ["bias_bf16", "cache_lengths_nonzero"],
        _pointer_and_load(
            "relative_logits",
            lambda x: [
                x["bias_bf16"],
                x["cache_lengths_nonzero"],
                x["offsets"],
            ],
        ),
    ),
    "relative_logits_bf16_odd_extent": (
        ["bias_bf16_odd", "cache_lengths"],
        _relative_logits_bf16_odd,
    ),
    "relative_logits_f32_odd_extent": (
        ["bias_f32", "cache_lengths"],
        _pointer_and_load(
            "relative_logits",
            lambda x: [x["bias_f32"], x["cache_lengths"], x["offsets"]],
        ),
    ),
}

BASE = ["q", "k", "v", "offsets", "max_len"]


def _types() -> dict[str, TensorType]:
    qkv = _gpu(DType.bfloat16, [TOKENS, HEADS, HEAD_DIM])
    dense = _gpu(DType.bfloat16, [BATCH, DENSE_SEQ, HEADS, HEAD_DIM])
    return {
        "q": qkv,
        "k": qkv,
        "v": qkv,
        "offsets": _gpu(DType.uint32, [BATCH + 1]),
        "max_len": TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
        "ids": _gpu(DType.int32, [TOKENS]),
        "mask3": _gpu(DType.float32, [BATCH, MAX_LEN, MAX_LEN]),
        "mask3_wide": _gpu(DType.float32, [BATCH, MAX_LEN, MAX_LEN + 8]),
        "mask4": _gpu(DType.float32, [BATCH, HEADS, MAX_LEN, MAX_LEN]),
        "dq": dense,
        "dk": dense,
        "dv": dense,
        "dmask3": _gpu(DType.float32, [BATCH, DENSE_SEQ, DENSE_SEQ]),
        "dmask4": _gpu(DType.float32, [BATCH, HEADS, DENSE_SEQ, DENSE_SEQ]),
        "bias_bf16": _gpu(DType.bfloat16, [TOKENS, HEADS, 64]),
        "bias_bf16_odd": _gpu(DType.bfloat16, [TOKENS, HEADS, 33]),
        "bias_f32": _gpu(DType.float32, [TOKENS, HEADS, 33]),
        "cache_lengths": _gpu(DType.uint32, [BATCH]),
        "cache_lengths_nonzero": _gpu(DType.uint32, [BATCH]),
    }


def _inputs() -> Inputs:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    def bf16(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.bfloat16)

    def additive(*shape: int) -> np.ndarray:
        values = rng.standard_normal(shape).astype(np.float32)
        return np.where(rng.random(shape) < 0.2, -10000.0, values).astype(
            np.float32
        )

    # Each sequence starts with a few -1 tokens, then nondecreasing segments.
    ids = np.concatenate(
        [
            np.concatenate(
                [np.full(4, -1), np.arange(n - 4) // max(1, (n - 4) // 3)]
            )
            for n in LENGTHS
        ]
    ).astype(np.int32)
    return {
        "q": bf16(TOKENS, HEADS, HEAD_DIM),
        "k": bf16(TOKENS, HEADS, HEAD_DIM),
        "v": bf16(TOKENS, HEADS, HEAD_DIM),
        "offsets": np.cumsum([0, *LENGTHS]).astype(np.uint32),
        "max_len": np.array([MAX_LEN], dtype=np.uint32),
        "ids": ids,
        "mask3": additive(BATCH, MAX_LEN, MAX_LEN),
        "mask3_wide": additive(BATCH, MAX_LEN, MAX_LEN + 8),
        "mask4": additive(BATCH, HEADS, MAX_LEN, MAX_LEN),
        "dq": bf16(BATCH, DENSE_SEQ, HEADS, HEAD_DIM),
        "dk": bf16(BATCH, DENSE_SEQ, HEADS, HEAD_DIM),
        "dv": bf16(BATCH, DENSE_SEQ, HEADS, HEAD_DIM),
        "dmask3": additive(BATCH, DENSE_SEQ, DENSE_SEQ),
        "dmask4": additive(BATCH, HEADS, DENSE_SEQ, DENSE_SEQ),
        "bias_bf16": bf16(TOKENS, HEADS, 64),
        "bias_bf16_odd": bf16(TOKENS, HEADS, 33),
        "bias_f32": rng.standard_normal((TOKENS, HEADS, 33)).astype(np.float32),
        "cache_lengths": np.zeros(BATCH, dtype=np.uint32),
        "cache_lengths_nonzero": np.array([5, 3], dtype=np.uint32),
    }


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])


@pytest.fixture(scope="module")
def ops_path() -> Path:
    path = os.getenv("ATTENTION_MASK_OPS_PATH")
    assert path is not None, "ATTENTION_MASK_OPS_PATH is not set"
    return Path(path)


@pytest.fixture(scope="module")
def inputs() -> Inputs:
    return _inputs()


def _buffer(value: torch.Tensor | np.ndarray, device: Accelerator) -> Buffer:
    buffer = (
        Buffer.from_numpy(value)
        if isinstance(value, np.ndarray)
        else Buffer.from_dlpack(value)
    )
    return buffer.to(device)


@pytest.mark.parametrize("case", list(CASES))
def test_matches_pointer_mask(
    case: str, session: InferenceSession, ops_path: Path, inputs: Inputs
) -> None:
    extra, build = CASES[case]
    names = [*BASE, *extra]
    types = _types()
    with Graph(
        f"attention_{case}",
        input_types=[types[n] for n in names],
        custom_extensions=[ops_path],
    ) as graph:
        x = dict(zip(names, (v.tensor for v in graph.inputs), strict=True))
        graph.output(*build(x))
    model = session.load(graph)

    device = Accelerator()
    args = [
        Buffer.from_numpy(inputs[n])
        if n == "max_len"
        else _buffer(inputs[n], device)
        for n in names
    ]
    reference, actual = (
        torch.from_dlpack(out.to(CPU())).float() for out in model.execute(*args)
    )
    assert torch.equal(reference, actual), (
        f"max abs diff {(reference - actual).abs().max().item()}"
    )
