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

"""Shared reference machinery for the V4.1 mHC GPU tests.

The gate oracle is an independent NumPy transcription of the
``mo.hyper_connection.gates`` math, and the chain helpers build and
simulate the trunk's shifted schedule (initial identity -> default
(fresh-``pre``) site -> identity site -> final collapse) over one device
or a two-way mesh.


The gate oracle is an independent NumPy transcription of the
``mo.hyper_connection.gates`` math: ``pre = sigmoid(proj * s_pre + b) +
hc_eps``, ``post = 2 * sigmoid(proj * s_post + b)``, and ``comb`` a
row-softmax over the row-major ``[input, output]`` mixer followed by
Sinkhorn passes that end on a column normalization.

The chain tests execute the exact shifted schedule the trunk runs
(initial identity -> default (fresh-``pre``) site -> identity site ->
final collapse), first on one device and then over a two-way mesh, which
exercises the ruleless gate outputs' placement claims and the mapping-
built identity mix.
"""

from __future__ import annotations

import numpy as np
import torch
from max.driver import Accelerator, Device, DLPackArray, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.module import Module
from max.pipelines.architectures.deepseekV41_modulev3.layers.mhc import (
    mhc_final_collapse,
    mhc_initial,
    mhc_post,
    mhc_pre,
)
from max.tensor.sharding import DeviceMapping, DeviceMesh, Replicated
from max.tensor.tensor import Tensor, TensorLayout, TensorType

HC = 4
HC_EPS = 1e-6
NORM_EPS = 1e-20
SINKHORN_ITERS = 20
ROWS = 7
HIDDEN = 512


def round_bf16(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rounded << 16).view(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gates_oracle(
    proj: np.ndarray, base: np.ndarray, scale: np.ndarray, iters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(pre, post, comb)`` from the projected stream, as the kernel does."""
    rows = proj.shape[0]
    pre = sigmoid(proj[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2.0 * sigmoid(proj[:, HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    logits = proj[:, 2 * HC :].reshape(rows, HC, HC) * scale[2] + base[
        2 * HC :
    ].reshape(1, HC, HC)
    # Row softmax over the output axis, then Sinkhorn ending column-pass.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    comb = exp / exp.sum(axis=-1, keepdims=True) + HC_EPS
    comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    return pre, post, comb.reshape(rows, HC * HC)


def proj_oracle(stream: np.ndarray, fn: np.ndarray, hidden: int) -> np.ndarray:
    flat = stream.astype(np.float32).reshape(stream.shape[0], HC * hidden)
    rms = 1.0 / np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + NORM_EPS)
    return (flat @ fn.T) * rms


def collapse_oracle(stream: np.ndarray, pre_mix: np.ndarray) -> np.ndarray:
    return np.sum(
        pre_mix.astype(np.float32)[:, :, None] * stream.astype(np.float32),
        axis=1,
    )


def post_oracle(
    y: np.ndarray, stream: np.ndarray, post: np.ndarray, comb: np.ndarray
) -> np.ndarray:
    y32 = y.astype(np.float32)
    mixed = np.einsum(
        "rij,rih->rjh",
        comb.reshape(-1, HC, HC).astype(np.float32),
        stream.astype(np.float32),
    )
    return post.astype(np.float32)[:, :, None] * y32[:, None, :] + mixed


def site_params(
    rng: np.random.Generator, hidden: int
) -> tuple[np.ndarray, ...]:
    mix = (2 + HC) * HC
    fn = (rng.standard_normal((mix, HC * hidden)) * 0.02).astype(np.float32)
    base = (rng.standard_normal(mix) * 0.5).astype(np.float32)
    scale = (1.0 + rng.standard_normal(3) * 0.1).astype(np.float32)
    return fn, base, scale


def param(array: np.ndarray, device: Device | DeviceMesh) -> Tensor:
    return Tensor.from_dlpack(np.ascontiguousarray(array)).to(device)


def to_f32_device(t: np.ndarray) -> Tensor:
    return Tensor.from_dlpack(np.ascontiguousarray(t)).to(Accelerator())


def to_bf16_device(t: np.ndarray) -> Tensor:
    return (
        Tensor.from_dlpack(np.ascontiguousarray(t))
        .to(Accelerator())
        .cast(DType.bfloat16)
    )


class Chain(Module[[Tensor], Tensor]):
    """The trunk's shifted schedule over three sites, as one body.

    ``mhc_initial`` (identity carry) -> ``mhc_pre`` with the default
    (fresh-``pre``) collapse -> ``mhc_pre`` with ``None`` (identity) ->
    ``mhc_final_collapse`` with the last site's ``pre``.
    """

    fn0: Tensor
    base0: Tensor
    scale0: Tensor
    y0: Tensor
    fn1: Tensor
    base1: Tensor
    scale1: Tensor
    y1: Tensor
    fn2: Tensor
    base2: Tensor
    scale2: Tensor
    y2: Tensor

    def __init__(self, hidden: int, mesh: DeviceMesh | None = None) -> None:
        super().__init__()
        device = mesh or Accelerator()
        rng = np.random.default_rng(41)
        # Host values for ``compile(weights=...)`` and the oracle; the
        # bf16 sublayer outputs ride as torch tensors (numpy has no bf16).
        self.host: dict[str, DLPackArray] = {}
        for site in range(3):
            fn, base, scale = site_params(rng, hidden)
            setattr(self, f"fn{site}", param(fn, device))
            setattr(self, f"base{site}", param(base, device))
            setattr(self, f"scale{site}", param(scale, device))
            y = round_bf16(
                (rng.standard_normal((ROWS, hidden)) * 0.2).astype(np.float32)
            )
            setattr(self, f"y{site}", param(y, device).cast(DType.bfloat16))
            self.host[f"fn{site}"] = fn
            self.host[f"base{site}"] = base
            self.host[f"scale{site}"] = scale
            self.host[f"y{site}"] = torch.from_numpy(y).to(torch.bfloat16)

    def forward(self, stream: Tensor) -> Tensor:
        post0, comb0, _pre0, _x0 = mhc_initial(
            stream,
            self.fn0,
            self.scale0,
            self.base0,
            HC,
            norm_eps=NORM_EPS,
            hc_eps=HC_EPS,
            sinkhorn_iters=SINKHORN_ITERS,
        )
        s = mhc_post(self.y0, stream, post0, comb0, HC)
        post1, comb1, _pre1, _x1 = mhc_pre(
            s,
            self.fn1,
            self.scale1,
            self.base1,
            HC,
            norm_eps=NORM_EPS,
            hc_eps=HC_EPS,
            sinkhorn_iters=SINKHORN_ITERS,
        )
        s = mhc_post(self.y1, s, post1, comb1, HC)
        post2, comb2, pre2, _x2 = mhc_pre(
            s,
            self.fn2,
            self.scale2,
            self.base2,
            HC,
            None,
            norm_eps=NORM_EPS,
            hc_eps=HC_EPS,
            sinkhorn_iters=SINKHORN_ITERS,
        )
        s = mhc_post(self.y2, s, post2, comb2, HC)
        return mhc_final_collapse(s, pre2)


def chain_oracle(chain: Chain, stream: np.ndarray) -> np.ndarray:
    """The chain's final collapse, simulated with the gate oracle."""
    hidden = stream.shape[-1]
    s = stream
    pres: list[np.ndarray] = []
    for site in range(3):
        fn = np.asarray(chain.host[f"fn{site}"], dtype=np.float32)
        base = np.asarray(chain.host[f"base{site}"], dtype=np.float32)
        scale = np.asarray(chain.host[f"scale{site}"], dtype=np.float32)
        proj = proj_oracle(s, fn, hidden)
        pre, post, comb = gates_oracle(proj, base, scale, SINKHORN_ITERS)
        if site not in (0, 2):
            # Site 1 collapses with its own fresh pre; sites 0 and 2 use
            # the identity carry.
            _ = round_bf16(collapse_oracle(s, pre))
        y = (
            torch.as_tensor(chain.host[f"y{site}"])
            .float()
            .numpy()
            .reshape(ROWS, hidden)
        )
        s = round_bf16(post_oracle(y, s, post, comb))
        pres.append(pre)
    # The final collapse uses the LAST site's pre.
    return round_bf16(collapse_oracle(s, pres[-1]))


def mesh() -> DeviceMesh:
    return DeviceMesh(
        devices=(Accelerator(0), Accelerator(1)),
        mesh_shape=(2,),
        axis_names=("tp",),
    )


def require_two_devices(context: str) -> tuple[Accelerator, Accelerator]:
    """TP2 tests execute on exactly the two GPUs the target requests.

    These never skip: the target's ``test.resources:gpu-memory`` is 2, so
    a worker that exposes fewer devices is an infrastructure failure that
    must surface, not hide.
    """
    count = accelerator_count()
    assert count >= 2, (
        f"{context} needs two visible GPUs; accelerator_count() = {count}"
    )
    return Accelerator(0), Accelerator(1)


def run_chain(
    mesh: DeviceMesh | None, hidden: int
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the chain, compiles it, runs it, and returns ``(got, want)``."""
    rng = np.random.default_rng(42)
    stream = round_bf16(
        (rng.standard_normal((ROWS, HC, hidden)) * 0.3).astype(np.float32)
    )
    chain = Chain(hidden, mesh)
    want = chain_oracle(chain, stream)

    input_type: TensorType | TensorLayout
    placement: Accelerator | DeviceMapping
    if mesh is None:
        input_type = TensorType(
            DType.bfloat16, [ROWS, HC, hidden], DeviceRef.GPU()
        )
        placement = Accelerator()
    else:
        input_type = TensorLayout(
            DType.bfloat16,
            [ROWS, HC, hidden],
            DeviceMapping(mesh, (Replicated(),)),
        )
        placement = DeviceMapping(mesh, (Replicated(),))
    run = chain.compile(input_type, weights=dict(chain.host))
    # The bf16 input is built through torch's DLPack export: the eager
    # interpreter has no CPU float32->bfloat16 cast, and the compiled
    # input type is bfloat16 either way.
    x_in = Tensor.from_dlpack(torch.from_numpy(stream).to(torch.bfloat16))
    out = run(x_in.to(placement))
    return out.cast(DType.float32).to_numpy(), want


# ---------------------------------------------------------------------------
# The production-shaped chain: every site's sublayer output is a function
# of the collapsed input, so the compiler cannot dead-code any collapse.
# The transform is fixed pointwise arithmetic -- bf16(x * 0.5 - 0.25),
# computed in fp32 and rounded once -- with an exact NumPy oracle.
# ---------------------------------------------------------------------------


def sublayer_of(collapsed: Tensor) -> Tensor:
    """The stand-in sublayer: one fp32 elementwise pass, one bf16 round."""
    return (collapsed.cast(DType.float32) * 0.5 - 0.25).cast(DType.bfloat16)


class ProdChain(Module[[Tensor], Tensor]):
    """The trunk's schedule with the sublayers pinned to the collapses."""

    fn0: Tensor
    base0: Tensor
    scale0: Tensor
    fn1: Tensor
    base1: Tensor
    scale1: Tensor
    fn2: Tensor
    base2: Tensor
    scale2: Tensor

    def __init__(self, hidden: int, mesh: DeviceMesh | None = None) -> None:
        super().__init__()
        device = mesh or Accelerator()
        rng = np.random.default_rng(45)
        self.host: dict[str, DLPackArray] = {}
        for site in range(3):
            fn, base, scale = site_params(rng, hidden)
            setattr(self, f"fn{site}", param(fn, device))
            setattr(self, f"base{site}", param(base, device))
            setattr(self, f"scale{site}", param(scale, device))
            self.host[f"fn{site}"] = fn
            self.host[f"base{site}"] = base
            self.host[f"scale{site}"] = scale

    def forward(self, stream: Tensor) -> Tensor:
        post0, comb0, _pre0, x0 = mhc_initial(
            stream,
            self.fn0,
            self.scale0,
            self.base0,
            HC,
            norm_eps=NORM_EPS,
            hc_eps=HC_EPS,
            sinkhorn_iters=SINKHORN_ITERS,
        )
        s = mhc_post(sublayer_of(x0), stream, post0, comb0, HC)
        post1, comb1, _pre1, x1 = mhc_pre(
            s,
            self.fn1,
            self.scale1,
            self.base1,
            HC,
            norm_eps=NORM_EPS,
            hc_eps=HC_EPS,
            sinkhorn_iters=SINKHORN_ITERS,
        )
        s = mhc_post(sublayer_of(x1), s, post1, comb1, HC)
        post2, comb2, pre2, x2 = mhc_pre(
            s,
            self.fn2,
            self.scale2,
            self.base2,
            HC,
            None,
            norm_eps=NORM_EPS,
            hc_eps=HC_EPS,
            sinkhorn_iters=SINKHORN_ITERS,
        )
        s = mhc_post(sublayer_of(x2), s, post2, comb2, HC)
        return mhc_final_collapse(s, pre2)


def prod_chain_oracle(chain: ProdChain, stream: np.ndarray) -> np.ndarray:
    """The production chain's final collapse, simulated with the oracle."""
    hidden = stream.shape[-1]
    s = stream
    pres: list[np.ndarray] = []
    for site in range(3):
        fn = np.asarray(chain.host[f"fn{site}"], dtype=np.float32)
        base = np.asarray(chain.host[f"base{site}"], dtype=np.float32)
        scale = np.asarray(chain.host[f"scale{site}"], dtype=np.float32)
        proj = proj_oracle(s, fn, hidden)
        pre, post, comb = gates_oracle(proj, base, scale, SINKHORN_ITERS)
        if site in (0, 2):
            x = round_bf16(s[:, 0])  # identity carry
        else:
            x = round_bf16(collapse_oracle(s, pre))  # fresh-pre collapse
        y = round_bf16(x.astype(np.float32) * 0.5 - 0.25)
        s = round_bf16(post_oracle(y, s, post, comb))
        pres.append(pre)
    return round_bf16(collapse_oracle(s, pres[-1]))


def run_prod_chain(
    mesh: DeviceMesh | None, hidden: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(46)
    stream = round_bf16(
        (rng.standard_normal((ROWS, HC, hidden)) * 0.3).astype(np.float32)
    )
    chain = ProdChain(hidden, mesh)
    want = prod_chain_oracle(chain, stream)

    input_type: TensorType | TensorLayout
    placement: Accelerator | DeviceMapping
    if mesh is None:
        input_type = TensorType(
            DType.bfloat16, [ROWS, HC, hidden], DeviceRef.GPU()
        )
        placement = Accelerator()
    else:
        input_type = TensorLayout(
            DType.bfloat16,
            [ROWS, HC, hidden],
            DeviceMapping(mesh, (Replicated(),)),
        )
        placement = DeviceMapping(mesh, (Replicated(),))
    run = chain.compile(input_type, weights=dict(chain.host))
    # The bf16 input is built through torch's DLPack export: the eager
    # interpreter has no CPU float32->bfloat16 cast, and the compiled
    # input type is bfloat16 either way.
    x_in = Tensor.from_dlpack(torch.from_numpy(stream).to(torch.bfloat16))
    out = run(x_in.to(placement))
    return out.cast(DType.float32).to_numpy(), want
