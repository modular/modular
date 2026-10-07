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
"""Checks the routed-expert matmuls: W4A4 against a dequantized reference,
and each device's share of the experts against all of them."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, TensorValue, ops
from max.graph.weights import WeightData
from max.nn.kernels import moe_create_indices
from max.pipelines.architectures.nemotron_h_modulev3.layers.moe import (
    _routed_experts,
    _with_shared_slices_value,
)
from max.pipelines.architectures.nemotron_h_modulev3.layers.quantized import (
    interleave_expert_scales,
    nvfp4_grouped_matmul,
    nvfp4_matmul,
)
from max.pipelines.architectures.nemotron_h_modulev3.weight_adapters import (
    interleave_nvfp4_scales,
    stack_bf16_experts,
    stack_nvfp4_experts,
)
from max.pipelines.weights._fp8 import e4m3fn_lut
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    e2m1_decode_table,
)

NUM_EXPERTS = 128
TOP_K = 6

# Packed E2M1 codes and their E4M3 block scales.
_Quantized = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]


def _dequantize(
    codes: npt.NDArray[np.uint8],
    scales: npt.NDArray[np.uint8],
    global_scale: float,
) -> npt.NDArray[np.float32]:
    table = e2m1_decode_table(FP4Format.NVFP4)
    values = np.stack([table[codes & 0xF], table[codes >> 4]], axis=-1)
    values = values.reshape(codes.shape[0], -1, 16)
    values = values * e4m3fn_lut()[scales][..., None] * global_scale
    return values.reshape(codes.shape[0], -1).astype(np.float32)


@pytest.mark.parametrize("rows", [128, 1856])
def test_host_interleave_matches_the_numpy_interleave(rows: int) -> None:
    """The model interleaves the checkpoint's scales on the host at init and
    copies them to the device, byte for byte as
    :func:`interleave_nvfp4_scales` lays out one expert."""
    experts, cols = 3, 168
    rng = np.random.default_rng(0)
    scales = rng.integers(1, 256, (experts, rows, cols), dtype=np.uint8)
    device = Accelerator()
    with Graph(
        "interleave",
        input_types=[
            TensorType(DType.float8_e4m3fn, scales.shape, DeviceRef.CPU()),
        ],
    ) as graph:
        (graw,) = (v.tensor for v in graph.inputs)
        graph.output(interleave_expert_scales(graw).to(DeviceRef.GPU()))
    model = InferenceSession(devices=[CPU(), device]).load(graph)
    (out,) = model.execute(
        Buffer.from_numpy(scales).view(DType.float8_e4m3fn, scales.shape),
    )
    assert isinstance(out, Buffer)
    got = torch.from_dlpack(out.view(DType.uint8)).cpu().numpy()
    want = np.stack([interleave_nvfp4_scales(s) for s in scales])
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "n, k, gather_in_kernel",
    [
        # The up projection; N is not a whole 128-row scale granule.
        (1856, 2688, True),
        # The down projection, on activations already in routed order.
        (2688, 1856, False),
    ],
)
@pytest.mark.parametrize("tokens", [1, 64, 700])
def test_w4a4_matches_dequantized_reference(
    n: int, k: int, gather_in_kernel: bool, tokens: int
) -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 256, (NUM_EXPERTS, n, k // 2), dtype=np.uint8)
    # E4M3 codes 0x28..0x48 span 2^-2..2^2.
    scales = rng.integers(0x28, 0x48, (NUM_EXPERTS, n, k // 16), dtype=np.uint8)
    global_scales = rng.uniform(0.005, 0.02, NUM_EXPERTS).astype(np.float32)
    experts = np.stack(
        [rng.choice(NUM_EXPERTS, TOP_K, replace=False) for _ in range(tokens)]
    ).astype(np.int32)
    x = rng.standard_normal((tokens, k)).astype(np.float32)
    # Rows whose magnitudes differ by orders of magnitude, and a few
    # outliers, exercise the per-row scale.
    x *= np.logspace(-2, 2, tokens, dtype=np.float32)[:, None]
    x[:, rng.integers(0, k, 4)] *= 20
    x_bf16 = torch.from_numpy(x).to(torch.bfloat16)

    device = Accelerator()
    dev = DeviceRef.GPU()
    with Graph(
        "w4a4_experts",
        input_types=[
            TensorType(DType.bfloat16, ["tokens", k], dev),
            TensorType(DType.int32, ["tokens", TOP_K], dev),
            TensorType(DType.uint8, [NUM_EXPERTS, n, k // 2], dev),
            # The checkpoint's scales, which the model keeps on the host.
            TensorType(
                DType.float8_e4m3fn, [NUM_EXPERTS, n, k // 16], DeviceRef.CPU()
            ),
            TensorType(DType.float32, [NUM_EXPERTS], dev),
        ],
    ) as graph:
        gx, gexperts, gweight, graw, gglobal = (v.tensor for v in graph.inputs)
        gblock = interleave_expert_scales(graw).to(gweight.device)
        order, starts, restore, expert_ids, _, offsets = moe_create_indices(
            ops.reshape(gexperts, [-1]), NUM_EXPERTS, needs_scales_offset=True
        )
        rows = ops.cast(order // TOP_K, DType.int32)
        out = nvfp4_grouped_matmul(
            gx if gather_in_kernel else ops.gather(gx, rows, axis=0),
            gweight,
            gblock,
            gglobal,
            starts,
            offsets,
            expert_ids,
            rows=rows if gather_in_kernel else None,
        )
        graph.output(ops.gather(out, restore, axis=0))

    model = InferenceSession(devices=[CPU(), device]).load(graph)
    (result,) = model.execute(
        Buffer.from_dlpack(x_bf16).to(device),
        Buffer.from_numpy(experts).to(device),
        Buffer.from_numpy(codes).to(device),
        Buffer.from_numpy(scales).view(DType.float8_e4m3fn, scales.shape),
        Buffer.from_numpy(global_scales).to(device),
    )
    assert isinstance(result, Buffer)
    got = torch.from_dlpack(result).float().cpu().numpy()
    got = got.reshape(tokens, TOP_K, n)

    x_ref = x_bf16.float().numpy()
    want = np.empty_like(got)
    for e in np.unique(experts):
        w = _dequantize(codes[e], scales[e], float(global_scales[e]))
        t, j = np.nonzero(experts == e)
        want[t, j] = x_ref[t] @ w.T

    # FP4 activations cost a few percent of the output norm; a wrong layout
    # or scale costs all of it.
    err = np.linalg.norm(got - want, axis=-1) / np.linalg.norm(want, axis=-1)
    assert err.max() < 0.15, f"worst row relative error {err.max():.3f}"
    if n % 128:
        # The rows past the last whole scale granule.
        tail = slice(n // 128 * 128, n)
        tail_err = np.linalg.norm(
            got[..., tail] - want[..., tail]
        ) / np.linalg.norm(want[..., tail])
        assert tail_err < 0.15, f"padded-granule error {tail_err:.3f}"


def _weight(array: npt.NDArray[np.generic], dtype: DType) -> WeightData:
    return WeightData(
        data=Buffer.from_numpy(np.ascontiguousarray(array)).view(
            dtype, array.shape
        ),
        name="",
        dtype=dtype,
        shape=Shape(array.shape),
    )


def _array(weight: WeightData) -> npt.NDArray[np.generic]:
    host = {DType.bfloat16: DType.uint16, DType.float8_e4m3fn: DType.uint8}
    buffer = weight.to_buffer()
    if weight.dtype in host:
        buffer = buffer.view(host[weight.dtype])
    return np.from_dlpack(buffer)


def _buffer(array: npt.NDArray[np.generic], dtype: DType) -> Buffer:
    return Buffer.from_numpy(np.ascontiguousarray(array)).view(
        dtype, array.shape
    )


def _device_params(
    stacked: dict[str, WeightData], prefix: str, w4a4: bool, n: int, d: int
) -> list[Buffer]:
    """Returns device ``d``'s share of a stacked routed projection, laid out
    as the model hands it to the matmul: the weight, then for W4A4 the
    interleaved block scales and the whole global scales.

    The channels split on axis 1 for ``up`` and axis 2 for ``down``.
    """
    axis = 1 if prefix.endswith(".up") else 2
    weight = stacked[f"{prefix}_weight"]
    share = np.split(_array(weight), n, axis=axis)[d]
    params = [_buffer(share, weight.dtype)]
    if w4a4:
        scales = np.split(
            _array(stacked[f"{prefix}_block_scale"]), n, axis=axis
        )
        block = np.stack(
            [
                interleave_nvfp4_scales(np.ascontiguousarray(s))
                for s in scales[d]
            ]
        )
        params.append(_buffer(block, DType.float8_e4m3fn))
        params.append(
            _buffer(_array(stacked[f"{prefix}_scale"]), DType.float32)
        )
    return params


def _nvfp4_checkpoint(
    rng: np.random.Generator,
    module: str,
    rows: int,
    cols: int,
    checkpoint: dict[str, WeightData],
) -> npt.NDArray[np.float32]:
    """Adds a random NVFP4 module to ``checkpoint`` and returns it dense."""
    codes = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
    scales = rng.integers(0x28, 0x48, (rows, cols // 16), dtype=np.uint8)
    global_scale = np.float32(rng.uniform(0.005, 0.02))
    checkpoint[f"{module}.weight"] = _weight(codes, DType.uint8)
    checkpoint[f"{module}.weight_scale"] = _weight(scales, DType.float8_e4m3fn)
    checkpoint[f"{module}.weight_scale_2"] = _weight(
        np.array(global_scale), DType.float32
    )
    return _dequantize(codes, scales, float(global_scale))


def _run_routed(
    x: torch.Tensor,
    ids: npt.NDArray[np.int32],
    weights: npt.NDArray[np.float32],
    shares: list[tuple[list[Buffer], list[Buffer]]],
) -> list[npt.NDArray[np.float32]]:
    """Runs :func:`_routed_experts` once per device share, on one GPU."""
    device = Accelerator()
    dev = DeviceRef.GPU()
    inputs = [
        Buffer.from_dlpack(x),
        Buffer.from_numpy(ids),
        Buffer.from_numpy(weights),
    ]
    sizes = []
    for up, down in shares:
        inputs += up + down
        sizes.append((len(up), len(down)))
    with Graph(
        "routed_shares",
        input_types=[TensorType(b.dtype, b.shape, dev) for b in inputs],
    ) as graph:
        gx, gids, gweights, *params = (v.tensor for v in graph.inputs)
        outputs: list[TensorValue] = []
        for n_up, n_down in sizes:
            gup, params = params[:n_up], params[n_up:]
            gdown, params = params[:n_down], params[n_down:]
            outputs.append(
                ops.cast(
                    _routed_experts(gx, gids, gweights, gup, gdown),
                    DType.float32,
                )
            )
        graph.output(*outputs)
    model = InferenceSession(devices=[device]).load(graph)
    results = model.execute(*(b.to(device) for b in inputs))
    return [torch.from_dlpack(r).cpu().numpy() for r in results]


@pytest.mark.parametrize("w4a4", [True, False])
def test_device_shares_of_the_routed_experts_sum_to_the_whole(
    w4a4: bool,
) -> None:
    """Two devices' shares of every expert's channels sum to the whole.

    Each device runs every routed row through its 96 of the 192 channels,
    padded with zeros to 128. No token picks expert 3, so its group is
    empty.
    """
    experts, top_k, hidden, inner, tokens, n = 8, 2, 256, 192, 37, 2
    mixer = "backbone.layers.1.mixer"
    shapes = {"up_proj": (inner, hidden), "down_proj": (hidden, inner)}
    rng = np.random.default_rng(0)
    checkpoint: dict[str, WeightData] = {}
    dense: dict[tuple[int, str], npt.NDArray[np.float32]] = {}
    for e in range(experts):
        for proj, (rows, cols) in shapes.items():
            module = f"{mixer}.experts.{e}.{proj}"
            if w4a4:
                dense[e, proj] = _nvfp4_checkpoint(
                    rng, module, rows, cols, checkpoint
                )
            else:
                values = torch.from_numpy(
                    rng.standard_normal((rows, cols)) * 0.05
                ).to(torch.bfloat16)
                checkpoint[f"{module}.weight"] = _weight(
                    values.view(torch.int16).numpy().view(np.uint16),
                    DType.bfloat16,
                )
                dense[e, proj] = values.float().numpy()
    if w4a4:
        whole = stack_nvfp4_experts(checkpoint, experts, {mixer: 0})
        split = stack_nvfp4_experts(
            checkpoint, experts, {mixer: 0}, num_devices=n
        )
    else:
        whole = stack_bf16_experts(checkpoint, experts, {mixer})
        split = stack_bf16_experts(checkpoint, experts, {mixer}, num_devices=n)

    ids = np.stack(
        [
            rng.choice([e for e in range(experts) if e != 3], top_k, False)
            for _ in range(tokens)
        ]
    ).astype(np.int32)
    weights = rng.uniform(0.1, 1.0, (tokens, top_k)).astype(np.float32)
    x = torch.from_numpy(rng.standard_normal((tokens, hidden))).to(
        torch.bfloat16
    )
    shares = [
        (
            _device_params(stacked, f"{mixer}.up", w4a4, count, d),
            _device_params(stacked, f"{mixer}.down", w4a4, count, d),
        )
        for stacked, count in ((whole, 1), (split, n))
        for d in range(count)
    ]
    got_whole, *got_shares = _run_routed(x, ids, weights, shares)

    x_ref = x.float().numpy()
    want = np.zeros((tokens, hidden), dtype=np.float32)
    for t in range(tokens):
        for e, w in zip(ids[t], weights[t], strict=True):
            up = np.maximum(x_ref[t] @ dense[e, "up_proj"].T, 0) ** 2
            want[t] += w * (up @ dense[e, "down_proj"].T)

    assert all(np.isfinite(o).all() for o in (got_whole, *got_shares))
    # FP4 activations cost a few percent of the output norm per matmul, and
    # the second quantizes relu2's wide-ranging outputs: the whole experts
    # land near 0.2 for the worst token. A wrong share costs all of it.
    tolerance = 0.3 if w4a4 else 0.02
    for name, got in (("whole", got_whole), ("shares", sum(got_shares))):
        err = np.linalg.norm(got - want, axis=-1) / np.linalg.norm(
            want, axis=-1
        )
        assert err.max() < tolerance, (
            f"{name}: worst token relative error {err.max():.3f}"
        )


@pytest.mark.parametrize(
    "n, k",
    [
        # The LM head: a vocabulary's worth of whole granules.
        (2048, 2688),
        # One device's share of the shared-expert up projection at TP=2;
        # N is not a whole 128-row scale granule.
        (1856, 2688),
        # One device's share of the shared-expert down projection at TP=2.
        (2688, 1856),
    ],
)
@pytest.mark.parametrize("tokens", [1, 64, 700])
def test_dense_w4a4_matches_dequantized_reference(
    n: int, k: int, tokens: int
) -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.integers(0x28, 0x48, (n, k // 16), dtype=np.uint8)
    global_scale = np.array([0.01], dtype=np.float32)
    x = rng.standard_normal((tokens, k)).astype(np.float32)
    x *= np.logspace(-2, 2, tokens, dtype=np.float32)[:, None]
    x[:, rng.integers(0, k, 4)] *= 20
    x_bf16 = torch.from_numpy(x).to(torch.bfloat16)

    device = Accelerator()
    dev = DeviceRef.GPU()
    block = interleave_nvfp4_scales(scales)[None]
    with Graph(
        "w4a4_dense",
        input_types=[
            TensorType(DType.bfloat16, ["tokens", k], dev),
            TensorType(DType.uint8, [n, k // 2], dev),
            TensorType(DType.float8_e4m3fn, list(block.shape), dev),
            TensorType(DType.float32, [1], dev),
        ],
    ) as graph:
        graph.output(nvfp4_matmul(*(v.tensor for v in graph.inputs)))

    model = InferenceSession(devices=[device]).load(graph)
    (result,) = model.execute(
        Buffer.from_dlpack(x_bf16).to(device),
        Buffer.from_numpy(codes).to(device),
        Buffer.from_numpy(block)
        .view(DType.float8_e4m3fn, block.shape)
        .to(device),
        Buffer.from_numpy(global_scale).to(device),
    )
    assert isinstance(result, Buffer)
    got = torch.from_dlpack(result).float().cpu().numpy()

    want = x_bf16.float().numpy() @ _dequantize(codes, scales, 0.01).T
    err = np.linalg.norm(got - want, axis=-1) / np.linalg.norm(want, axis=-1)
    assert err.max() < 0.15, f"worst row relative error {err.max():.3f}"
    if n % 128:
        tail = slice(n // 128 * 128, n)
        tail_err = np.linalg.norm(got[:, tail] - want[:, tail]) / (
            np.linalg.norm(want[:, tail])
        )
        assert tail_err < 0.15, f"padded-granule error {tail_err:.3f}"


@pytest.mark.parametrize("num_devices", [1, 2])
def test_shared_expert_slices_sum_to_the_shared_expert(
    num_devices: int,
) -> None:
    """The shared expert, split into expert-wide slices that every token
    picks at weight 1, adds the shared MLP's output to the routed sum.

    At two devices each slice splits by channel like a routed expert, 96 of
    its 192 channels per device padded to 128, and the devices' partial sums
    add up to the whole.
    """
    experts, slices, top_k, hidden, inner, tokens = 8, 2, 2, 256, 192, 37
    mixer = "backbone.layers.1.mixer"
    rng = np.random.default_rng(0)
    checkpoint: dict[str, WeightData] = {}
    routed: dict[tuple[int, str], npt.NDArray[np.float32]] = {}
    for e in range(experts):
        routed[e, "up"] = _nvfp4_checkpoint(
            rng, f"{mixer}.experts.{e}.up_proj", inner, hidden, checkpoint
        )
        routed[e, "down"] = _nvfp4_checkpoint(
            rng, f"{mixer}.experts.{e}.down_proj", hidden, inner, checkpoint
        )
    shared = f"{mixer}.shared_experts"
    shared_up = _nvfp4_checkpoint(
        rng, f"{shared}.up_proj", slices * inner, hidden, checkpoint
    )
    shared_down = _nvfp4_checkpoint(
        rng, f"{shared}.down_proj", hidden, slices * inner, checkpoint
    )
    fused = stack_nvfp4_experts(
        checkpoint, experts, {mixer: slices}, num_devices=num_devices
    )
    separate = stack_nvfp4_experts(checkpoint, experts, {mixer: 0})

    ids = np.stack(
        [rng.choice(experts, top_k, replace=False) for _ in range(tokens)]
    ).astype(np.int32)
    weights = rng.uniform(0.1, 1.0, (tokens, top_k)).astype(np.float32)
    x_bf16 = torch.from_numpy(
        rng.standard_normal((tokens, hidden)).astype(np.float32)
    ).to(torch.bfloat16)

    def dense_params(module: str) -> list[Buffer]:
        codes = _array(checkpoint[f"{module}.weight"])
        scales = _array(checkpoint[f"{module}.weight_scale"])
        block = interleave_nvfp4_scales(scales.view(np.uint8))[None]
        return [
            _buffer(codes, DType.uint8),
            _buffer(block, DType.float8_e4m3fn),
            _buffer(
                _array(checkpoint[f"{module}.weight_scale_2"]).reshape(1),
                DType.float32,
            ),
        ]

    fused_shares = [
        (
            _device_params(fused, f"{mixer}.up", True, num_devices, d),
            _device_params(fused, f"{mixer}.down", True, num_devices, d),
        )
        for d in range(num_devices)
    ]
    routed_only = (
        _device_params(separate, f"{mixer}.up", True, 1, 0),
        _device_params(separate, f"{mixer}.down", True, 1, 0),
    )
    shared_params = dense_params(f"{shared}.up_proj") + dense_params(
        f"{shared}.down_proj"
    )

    device = Accelerator()
    dev = DeviceRef.GPU()
    inputs = [
        Buffer.from_dlpack(x_bf16),
        Buffer.from_numpy(ids),
        Buffer.from_numpy(weights),
        *routed_only[0],
        *routed_only[1],
        *shared_params,
    ]
    for up, down in fused_shares:
        inputs += up + down
    with Graph(
        "shared_slices",
        input_types=[TensorType(b.dtype, b.shape, dev) for b in inputs],
    ) as graph:
        gx, gids, gweights, *params = (v.tensor for v in graph.inputs)
        gup, gdown, gshared = params[:3], params[3:6], params[6:12]
        params = params[12:]
        slice_ids, slice_weights = _with_shared_slices_value(
            gids, gweights, experts, slices
        )
        fused_outs = []
        for _ in range(num_devices):
            fup, fdown, params = params[:3], params[3:6], params[6:]
            fused_outs.append(
                ops.cast(
                    _routed_experts(gx, slice_ids, slice_weights, fup, fdown),
                    DType.float32,
                )
            )
        hidden_act = ops.relu(nvfp4_matmul(gx, *gshared[:3]))
        separate_out = _routed_experts(
            gx, gids, gweights, gup, gdown
        ) + nvfp4_matmul(hidden_act * hidden_act, *gshared[3:])
        graph.output(ops.cast(separate_out, DType.float32), *fused_outs)
    model = InferenceSession(devices=[device]).load(graph)
    results = model.execute(*(b.to(device) for b in inputs))
    separate_out, *fused_parts = (
        torch.from_dlpack(r).cpu().numpy() for r in results
    )
    fused_out = sum(fused_parts)

    def mlp(
        h: npt.NDArray[np.float32],
        up: npt.NDArray[np.float32],
        down: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        return (np.maximum(h @ up.T, 0) ** 2) @ down.T

    x_ref = x_bf16.float().numpy()
    want = mlp(x_ref, shared_up, shared_down)
    for t in range(tokens):
        for j in range(top_k):
            e = ids[t, j]
            want[t] += (
                weights[t, j]
                * mlp(x_ref[t : t + 1], routed[e, "up"], routed[e, "down"])[0]
            )

    def error(
        got: npt.NDArray[np.float32], ref: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        return np.linalg.norm(got - ref, axis=-1) / np.linalg.norm(ref, axis=-1)

    assert np.isfinite(fused_out).all()
    # The slices quantize the down projection's input with each slice's row
    # scale instead of the whole row's, so the two round differently, by
    # about the FP4 error of either. A wrong slice would be off by all of it.
    apart = error(fused_out, separate_out)
    assert np.median(apart) < 0.05, f"median apart {np.median(apart):.3f}"
    assert apart.max() < 0.15, f"worst token apart {apart.max():.3f}"
    # Slicing loses no accuracy against the unquantized reference.
    fused_err = error(fused_out, want).mean()
    separate_err = error(separate_out, want).mean()
    assert fused_err < 1.1 * separate_err, (
        f"fused error {fused_err:.3f} vs separate {separate_err:.3f}"
    )
