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
"""Validates the fused mHC pre-site op against float64 and the op chain.

`mo.mhc.pre` replaces the projection, :func:`hyper_connection_gates`, the
stream collapse and the optional input RMSNorm of an mHC site. Its gates are
held to 1e-4 of a float64 reference; the collapsed row, rounded to bf16, to
``rtol * |ref| + 1e-2 * rms(ref)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy.kernels import (
    hyper_connection_gates,
    mhc_pre,
    mhc_pre_supports,
)
from torch.utils.dlpack import from_dlpack

_HC = 4
_MIX = 2 * _HC + _HC * _HC
_ITERS = 20
_HC_EPS = 1e-6
_RMS_EPS = 1e-5
_NORM_EPS = 1e-5
_HIDDEN = 4096

_GATE_ATOL = 1e-4
_COLLAPSED_RTOL = 2.0**-7
_NORMED_RTOL = 2.0**-6
_VALUE_ATOL_FRAC = 1e-2

_TORCH_DTYPE = {DType.float32: torch.float32, DType.bfloat16: torch.bfloat16}


@dataclass(frozen=True)
class _Config:
    fn_dtype: DType
    has_norm: bool
    hidden: int = _HIDDEN


@dataclass
class _Outputs:
    post: torch.Tensor
    comb: torch.Tensor
    collapsed: torch.Tensor
    pre: torch.Tensor


def _input_types(cfg: _Config, device: DeviceRef) -> list[TensorType]:
    types = [
        TensorType(DType.bfloat16, ["tokens", _HC, cfg.hidden], device=device),
        TensorType(cfg.fn_dtype, [_MIX, _HC * cfg.hidden], device=device),
        TensorType(DType.float32, [_MIX], device=device),
        TensorType(DType.float32, [3], device=device),
    ]
    if cfg.has_norm:
        types.append(TensorType(DType.bfloat16, [cfg.hidden], device=device))
    return types


def _fused(cfg: _Config, inputs: list[TensorValue]) -> list[TensorValue]:
    streams, hc_fn, base, scale = inputs[:4]
    return list(
        mhc_pre(
            streams,
            hc_fn,
            base,
            scale,
            hc_sinkhorn_iters=_ITERS,
            rms_eps=_RMS_EPS,
            hc_eps=_HC_EPS,
            gamma=inputs[4] if cfg.has_norm else None,
            norm_eps=_NORM_EPS if cfg.has_norm else None,
        )
    )


def _op_chain(cfg: _Config, inputs: list[TensorValue]) -> list[TensorValue]:
    """The pre site as `max.nn.HyperConnection` builds it, plus the norm."""
    streams_in, hc_fn, base, scale = inputs[:4]
    rows = streams_in.shape[0]
    streams = ops.cast(streams_in, DType.float32)
    flat = ops.reshape(streams, [rows, _HC * cfg.hidden])
    rms = ops.rsqrt(ops.mean(flat * flat, axis=-1) + _RMS_EPS)
    proj = (flat @ ops.cast(hc_fn, DType.float32).T) * rms
    pre, post, comb = hyper_connection_gates(
        proj,
        base,
        scale,
        hc_mult=_HC,
        hc_eps=_HC_EPS,
        hc_sinkhorn_iters=_ITERS,
    )
    collapsed = ops.cast(
        ops.squeeze(ops.sum(ops.unsqueeze(pre, -1) * streams, axis=1), 1),
        streams_in.dtype,
    )
    if cfg.has_norm:
        collapsed = ops.rms_norm(collapsed, inputs[4], _NORM_EPS)
    return [post, ops.reshape(comb, [rows, _HC, _HC]), collapsed, pre]


def _load(
    session: InferenceSession,
    cfg: _Config,
    with_chain: bool,
    device: DeviceRef,
) -> Model:
    name = (
        f"mhc_pre_{cfg.fn_dtype}_{int(cfg.has_norm)}_{cfg.hidden}"
        f"_{int(with_chain)}_{device.id}"
    )
    with Graph(name, input_types=_input_types(cfg, device)) as graph:
        inputs = [v.tensor for v in graph.inputs]
        outs = _fused(cfg, inputs)
        if with_chain:
            outs += _op_chain(cfg, inputs)
        graph.output(*outs)
    return session.load(graph)


@pytest.fixture(scope="module")
def compiled(
    gpu_session: InferenceSession,
) -> Callable[[_Config, bool], Model]:
    """Compiles each config once; the symbolic token dim serves every T."""
    cache: dict[tuple[_Config, bool], Model] = {}

    def get(cfg: _Config, with_chain: bool) -> Model:
        if (cfg, with_chain) not in cache:
            cache[cfg, with_chain] = _load(
                gpu_session, cfg, with_chain, DeviceRef.GPU()
            )
        return cache[cfg, with_chain]

    return get


def _inputs(
    cfg: _Config, tokens: int, seed: int, device: str = "cuda"
) -> list[torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    k = _HC * cfg.hidden
    streams = torch.randn(
        tokens, _HC, cfg.hidden, generator=gen, device=device
    ).to(torch.bfloat16)
    # fp32 weights keep their low bits, so a bf16-only projection fails.
    hc_fn = (torch.randn(_MIX, k, generator=gen, device=device) / k**0.5).to(
        _TORCH_DTYPE[cfg.fn_dtype]
    )
    base = 0.85 * (2 * torch.rand(_MIX, generator=gen, device=device) - 1)
    scale = torch.tensor([0.5, 0.5, 1.0], device=device)
    tensors = [streams, hc_fn, base, scale]
    if cfg.has_norm:
        gamma = 1 + 0.085 * (
            2 * torch.rand(cfg.hidden, generator=gen, device=device) - 1
        )
        tensors.append(gamma.to(torch.bfloat16))
    return tensors


def _reference(cfg: _Config, inputs: list[torch.Tensor]) -> _Outputs:
    """The pre site in float64, in row chunks to bound its memory."""
    chunk = 1024
    parts = [
        _reference_rows(cfg, [inputs[0][r : r + chunk], *inputs[1:]])
        for r in range(0, inputs[0].shape[0], chunk)
    ]
    post, comb, collapsed, pre = (
        torch.cat([getattr(p, name) for p in parts])
        for name in ("post", "comb", "collapsed", "pre")
    )
    return _Outputs(post, comb, collapsed, pre)


def _reference_rows(cfg: _Config, inputs: list[torch.Tensor]) -> _Outputs:
    """The pre site in float64; the norm reads the bf16-rounded collapse."""
    streams, hc_fn, base, scale = (t.double() for t in inputs[:4])
    rows = streams.shape[0]
    flat = streams.reshape(rows, -1)
    rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + _RMS_EPS)
    mix = (flat @ hc_fn.T) * rms
    pre_w, post_w, comb_w = mix.split([_HC, _HC, _HC * _HC], dim=-1)
    pre_b, post_b, comb_b = base.split([_HC, _HC, _HC * _HC])

    pre = torch.sigmoid(pre_w * scale[0] + pre_b) + _HC_EPS
    post = 2 * torch.sigmoid(post_w * scale[1] + post_b)
    comb = (comb_w * scale[2] + comb_b).view(rows, _HC, _HC)
    comb = torch.softmax(comb, dim=-1) + _HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + _HC_EPS)
    for _ in range(_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + _HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + _HC_EPS)

    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=1)
    if cfg.has_norm:
        collapsed = collapsed.to(torch.bfloat16).double()
        collapsed = collapsed * torch.rsqrt(
            collapsed.square().mean(-1, keepdim=True) + _NORM_EPS
        )
        collapsed = collapsed * inputs[4].double()
    return _Outputs(post, comb, collapsed, pre)


def _to_outputs(buffers: list[Buffer], start: int = 0) -> _Outputs:
    post, comb, collapsed, pre = (
        from_dlpack(b).clone() for b in buffers[start : start + 4]
    )
    return _Outputs(post, comb, collapsed, pre)


def _worst(*errors: float) -> float:
    """Returns the largest error, a NaN counting as infinite.

    The builtin ``max`` keeps a NaN only when it comes first.
    """
    return math.inf if any(math.isnan(e) for e in errors) else max(errors)


def _errors(
    cfg: _Config, actual: _Outputs, expected: _Outputs
) -> tuple[float, float]:
    """Returns the worst gate error and the collapsed error over its atol."""
    gate = _worst(
        (actual.post.double() - expected.post).abs().max().item(),
        (actual.comb.double() - expected.comb).abs().max().item(),
        (actual.pre.double() - expected.pre).abs().max().item(),
    )
    rtol = _NORMED_RTOL if cfg.has_norm else _COLLAPSED_RTOL
    ref = expected.collapsed.double()
    atol = _VALUE_ATOL_FRAC * ref.square().mean().sqrt().item()
    excess = (actual.collapsed.double() - ref).abs() - rtol * ref.abs()
    # NaN compares false, so it is turned into a failure explicitly.
    if not torch.isfinite(actual.collapsed).all():
        return gate, float("inf")
    return gate, excess.max().item() / atol


def _check(
    cfg: _Config, actual: _Outputs, expected: _Outputs, label: str
) -> None:
    gate, collapsed = _errors(cfg, actual, expected)
    print(f"{label}: gates {gate:.3e} collapsed/atol {collapsed:.3e}")
    assert gate <= _GATE_ATOL, f"{label}: gate error {gate:.3e}"
    assert collapsed <= 1.0, f"{label}: collapsed error {collapsed:.3f}x atol"


def _as_reference(outputs: _Outputs) -> _Outputs:
    return _Outputs(
        outputs.post.double(),
        outputs.comb.double(),
        outputs.collapsed.double(),
        outputs.pre.double(),
    )


_CONFIGS = [
    _Config(fn_dtype, has_norm)
    for fn_dtype in (DType.float32, DType.bfloat16)
    for has_norm in (False, True)
]


@pytest.mark.parametrize("tokens", [1, 8, 16, 128, 1024, 8192])
@pytest.mark.parametrize("cfg", _CONFIGS, ids=str)
def test_pre_matches_float64_and_op_chain(
    compiled: Callable[[_Config, bool], Model], cfg: _Config, tokens: int
) -> None:
    inputs = _inputs(cfg, tokens, seed=tokens)
    outs = compiled(cfg, True).execute(*inputs)
    fused, chain = _to_outputs(outs), _to_outputs(outs, 4)
    assert fused.collapsed.shape == (tokens, cfg.hidden)
    assert fused.comb.shape == (tokens, _HC, _HC)

    reference = _reference(cfg, inputs)
    label = f"{cfg} T={tokens}"
    _check(cfg, fused, reference, f"{label} fused vs fp64")
    _check(cfg, chain, reference, f"{label} chain vs fp64")
    # The two implementations each sit within the gate of fp64.
    gate, collapsed = _errors(cfg, fused, _as_reference(chain))
    print(f"{label} fused vs chain: gates {gate:.3e} collapsed {collapsed:.3e}")
    assert gate <= 2 * _GATE_ATOL
    assert collapsed <= 1.0


@pytest.mark.parametrize(
    "cfg",
    [
        _Config(DType.float32, False, 5120),
        _Config(DType.float32, True, 5120),
        # Caps the K split at 64, like 5120, and spreads six 512-wide chunks
        # unevenly over the finish kernel's four waves.
        _Config(DType.bfloat16, True, 3072),
        # The widest supported stream.
        _Config(DType.float32, True, 8192),
    ],
    ids=str,
)
def test_pre_other_hidden_sizes(
    compiled: Callable[[_Config, bool], Model], cfg: _Config
) -> None:
    tokens = 128
    inputs = _inputs(cfg, tokens, seed=cfg.hidden)
    fused = _to_outputs(compiled(cfg, False).execute(*inputs))
    _check(cfg, fused, _reference(cfg, inputs), f"{cfg} T={tokens}")


def test_pre_one_graph_across_token_counts(
    compiled: Callable[[_Config, bool], Model],
) -> None:
    """At D = 4096, T > 3072 runs one launch; smaller T split K over two."""
    cfg = _Config(DType.bfloat16, True)
    model = compiled(cfg, False)
    # An empty batch launches nothing and returns empty outputs.
    empty = _to_outputs(model.execute(*_inputs(cfg, 0, seed=0)))
    assert empty.collapsed.shape == (0, cfg.hidden)
    assert empty.comb.shape == (0, _HC, _HC)
    for tokens in (8192, 1, 2048, 8, 4096, 16, 3072, 1024, 3):
        inputs = _inputs(cfg, tokens, seed=tokens + 1)
        fused = _to_outputs(model.execute(*inputs))
        _check(cfg, fused, _reference(cfg, inputs), f"dynamic T={tokens}")


def test_pre_capture_replay_matches_eager(
    compiled: Callable[[_Config, bool], Model],
) -> None:
    """The split-K workspace stays valid when the op runs from a graph."""
    cfg = _Config(DType.float32, True)
    tokens = 8
    model = compiled(cfg, False)
    device = model.input_devices[0]
    buffers = [
        Buffer.from_dlpack(t).to(device) for t in _inputs(cfg, tokens, 0)
    ]
    key = 3842
    captured = model.capture(key, *buffers)
    try:
        for seed in range(1, 6):
            inputs = _inputs(cfg, tokens, seed)
            for buffer, value in zip(buffers, inputs, strict=True):
                buffer.inplace_copy_from(Buffer.from_dlpack(value).to(device))
            model.replay(key, *buffers)
            # An eager run of another size between replays draws its own
            # workspace from the same pool.
            other = _inputs(cfg, 16, seed + 100)
            other_out = _to_outputs(model.execute(*other))
            replayed = _to_outputs(captured)
            eager = _to_outputs(model.execute(*buffers))

            for name in ("post", "comb", "collapsed", "pre"):
                assert torch.equal(
                    getattr(replayed, name), getattr(eager, name)
                ), f"replay {seed}: {name} differs from eager"
            _check(
                cfg,
                replayed,
                _reference(cfg, inputs),
                f"replay {seed} T={tokens}",
            )
            _check(cfg, other_out, _reference(cfg, other), "eager T=16")
    finally:
        model.release_captured_graph(key)


def test_pre_on_two_devices() -> None:
    if accelerator_count() < 2:
        pytest.skip("needs two GPUs")
    session = InferenceSession(devices=[Accelerator(0), Accelerator(1)])
    cfg = _Config(DType.float32, True)
    models = [_load(session, cfg, False, DeviceRef.GPU(i)) for i in (0, 1)]
    for tokens in (8, 1024):
        for i, model in enumerate(models):
            # DLPack exports a tensor only from torch's current device.
            with torch.cuda.device(i):
                inputs = _inputs(
                    cfg, tokens, seed=10 * tokens + i, device=f"cuda:{i}"
                )
                fused = _to_outputs(model.execute(*inputs))
            assert fused.post.device == torch.device(f"cuda:{i}")
            _check(
                cfg,
                fused,
                _reference(cfg, inputs),
                f"device {i} T={tokens}",
            )


def _fake_device(arch: str, api: str = "hip") -> MagicMock:
    device = MagicMock(spec=Accelerator)
    device.api = api
    device.architecture_name = arch
    return device


@pytest.mark.parametrize("output", ["post", "comb", "pre", "collapsed"])
def test_nan_output_fails_the_check(output: str) -> None:
    """One NaN in any output fails the check, wherever it sits."""
    shape = {"post": (2, _HC), "comb": (2, _HC, _HC), "pre": (2, _HC)}
    expected = _Outputs(
        **{k: torch.rand(v, dtype=torch.float64) for k, v in shape.items()},
        collapsed=torch.rand(2, 8, dtype=torch.float64),
    )
    actual = _Outputs(**{k: v.clone() for k, v in vars(expected).items()})
    getattr(actual, output).view(-1)[-1] = math.nan
    error = "collapsed error" if output == "collapsed" else "gate error"
    with pytest.raises(AssertionError, match=error):
        _check(_Config(DType.float32, False), actual, expected, output)


def test_pre_supports() -> None:
    gfx950 = _fake_device("gfx950")
    bf16, f32 = DType.bfloat16, DType.float32
    assert mhc_pre_supports(_HC, 4096, bf16, f32, gfx950)
    assert mhc_pre_supports(_HC, 5120, bf16, bf16, gfx950)
    assert mhc_pre_supports(_HC, 8192, bf16, f32, gfx950)
    assert mhc_pre_supports(_HC, 4096, bf16, f32, DeviceRef.GPU())

    assert not mhc_pre_supports(_HC, 4096, bf16, f32, _fake_device("gfx942"))
    assert not mhc_pre_supports(
        _HC, 4096, bf16, f32, _fake_device("sm_100a", api="cuda")
    )
    assert not mhc_pre_supports(_HC, 4096, bf16, f32, DeviceRef.CPU())
    assert not mhc_pre_supports(
        _HC, 4096, bf16, f32, DeviceRef.GPU(accelerator_count())
    )
    assert not mhc_pre_supports(8, 4096, bf16, f32, gfx950)
    assert not mhc_pre_supports(2, 4096, bf16, f32, gfx950)
    assert mhc_pre_supports(_HC, 3072, bf16, f32, gfx950)
    assert not mhc_pre_supports(_HC, 1536, bf16, f32, gfx950)
    assert not mhc_pre_supports(_HC, 3584, bf16, f32, gfx950)
    assert not mhc_pre_supports(_HC, 16384, bf16, f32, gfx950)
    assert not mhc_pre_supports(_HC, 4096, DType.float16, f32, gfx950)
    assert not mhc_pre_supports(_HC, 4096, bf16, DType.float16, gfx950)


def test_pre_rejects_unsupported_hidden_size() -> None:
    cfg = _Config(DType.float32, False, hidden=1536)
    device = DeviceRef.GPU()
    with Graph(
        "mhc_pre_bad_hidden", input_types=_input_types(cfg, device)
    ) as graph:
        with pytest.raises(ValueError, match="multiple of 1024"):
            _fused(cfg, [v.tensor for v in graph.inputs])
