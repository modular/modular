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

"""One mHC site, tensor-level, against the pinned torch reference.

Scoped to a single site deliberately: GLM-5.3-Flash has 90 of them, and an
end-to-end logit check cannot attribute a mismatch in any one. The dimensions
here are the real model's -- ``hidden_size`` 4096, ``hc_mult`` 4,
``hc_sinkhorn_iters`` 20 -- because the mapping weight's shape is
``[24, hc_mult * hidden_size]`` and a narrowed fixture would exercise a matmul
the model never runs.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from max.driver import Accelerator, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.legacy.layer import Module
from max.nn.legacy.norm import RMSNorm
from max.pipelines.architectures.glm5_next.layers import (
    hyper_connection as hyper_connection_module,
)
from max.pipelines.architectures.glm5_next.layers.hyper_connection import (
    HyperConnection,
    StreamMixing,
)
from torch_reference import mhc_mapping, mhc_write_back, rms_norm

HIDDEN_SIZE = 4096
HC_MULT = 4
SINKHORN_ITERS = 20
HC_EPS = 1e-6
RMS_NORM_EPS = 1e-5
TOKENS = 17
SITES = 90
"""Sites in the real model: two per layer on layers 0-44."""

# The mapping runs entirely in float32 on both sides, so the only slack is
# reassociation in the 16384-wide reductions and the 39-step Sinkhorn chain.
# Measured on this fixture: `post` 1.19e-6 against a peak of 1.99, `comb`
# 5.96e-7 against a peak of 0.977 -- about ten float32 ulps, which is what a
# chain of that length costs.
MAPPING_RTOL = 1e-5
MAPPING_ATOL = 2e-6

BF16_ULP = 2.0**-8
"""One bfloat16 ulp relative to 1.0."""

# The collapse and the write-back land in bfloat16, and the write-back is a sum
# of two same-magnitude terms, so cancellation leaves an error floor set by the
# *operands* rather than by the result: two bfloat16 ulps at the tensor's peak
# magnitude. Measured on this fixture: 7.8e-3 on the collapse (peak 4.5, half an
# ulp) and 6.25e-2 on the write-back (peak 9.6, two ulps).
BF16_ULPS_ALLOWED = 2


class _Site(Module):
    """One mHC site plus the block's input layernorm, as the decoder wires it."""

    def __init__(self) -> None:
        super().__init__()
        self.hc = HyperConnection(
            hidden_size=HIDDEN_SIZE,
            hc_mult=HC_MULT,
            dtype=DType.bfloat16,
            sinkhorn_iters=SINKHORN_ITERS,
            eps=HC_EPS,
            rms_norm_eps=RMS_NORM_EPS,
            name="hc_attn",
        )
        self.norm = RMSNorm(
            HIDDEN_SIZE,
            DType.bfloat16,
            RMS_NORM_EPS,
            multiply_before_cast=False,
        )

    def __call__(
        self, streams: TensorValue, y: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
        mixing = self.hc(streams, self.norm.weight)
        return (
            mixing.post,
            mixing.comb,
            mixing.xs,
            self.hc.write_back(streams, y, mixing),
        )


@pytest.fixture(scope="module")
def weights() -> dict[str, torch.Tensor]:
    """Site weights, at the reference's init scales.

    ``fn`` is normal at the reference's initializer std, ``base`` is given a
    non-zero spread so a dropped bias would show, and ``scale`` is perturbed
    around 1 so a swapped scale index would show.
    """
    generator = torch.Generator().manual_seed(20260826)
    mix = (2 + HC_MULT) * HC_MULT
    return {
        "hc_attn_fn": (
            torch.randn(
                mix, HC_MULT * HIDDEN_SIZE, generator=generator
            ).bfloat16()
            * 0.02
        ),
        "hc_attn_base": torch.randn(mix, generator=generator) * 0.5,
        "hc_attn_scale": 1.0 + torch.randn(3, generator=generator) * 0.1,
        "norm.weight": (
            1.0 + torch.randn(HIDDEN_SIZE, generator=generator) * 0.05
        ).bfloat16(),
    }


@pytest.fixture(scope="module")
def streams() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(
        TOKENS, HC_MULT, HIDDEN_SIZE, generator=generator
    ).bfloat16()


@pytest.fixture(scope="module")
def sublayer_output() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(TOKENS, HIDDEN_SIZE, generator=generator).bfloat16()


@pytest.fixture(scope="module")
def site_weight_names(weights: dict[str, torch.Tensor]) -> set[str]:
    del weights
    return set(_Site().state_dict())


@pytest.fixture(scope="module")
def max_outputs(
    weights: dict[str, torch.Tensor],
    streams: torch.Tensor,
    sublayer_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the site once in MAX; returns ``(post, comb, xs, streams')``."""
    site = _Site()
    site.load_state_dict(weights)
    device = Accelerator(0)
    session = InferenceSession(devices=[device])
    graph = Graph(
        "Glm5NextHyperConnectionSite",
        site,
        input_types=(
            TensorType(
                DType.bfloat16,
                (TOKENS, HC_MULT, HIDDEN_SIZE),
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.bfloat16, (TOKENS, HIDDEN_SIZE), device=DeviceRef.GPU()
            ),
        ),
    )
    compiled = session.load(graph, weights_registry=site.state_dict())
    outputs = compiled.execute(
        streams.cuda().contiguous(), sublayer_output.cuda().contiguous()
    )
    post, comb, xs, new_streams = (
        torch.from_dlpack(out).cpu() for out in outputs
    )
    return post, comb, xs, new_streams


@pytest.fixture(scope="module")
def reference_outputs(
    weights: dict[str, torch.Tensor],
    streams: torch.Tensor,
    sublayer_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    post, comb, collapsed = mhc_mapping(
        streams,
        weights["hc_attn_fn"],
        weights["hc_attn_base"],
        weights["hc_attn_scale"],
        hc_mult=HC_MULT,
        sinkhorn_iters=SINKHORN_ITERS,
        hc_eps=HC_EPS,
        rms_norm_eps=RMS_NORM_EPS,
    )
    xs = rms_norm(collapsed, weights["norm.weight"], RMS_NORM_EPS)
    new_streams = mhc_write_back(streams, sublayer_output, post, comb)
    return post, comb, xs, new_streams


def _assert_bf16_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compares two bfloat16 tensors to a peak-magnitude ulp budget."""
    atol = BF16_ULPS_ALLOWED * BF16_ULP * expected.abs().max().item()
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=0.0, atol=atol
    )


def test_weight_names_match_the_checkpoint(
    site_weight_names: set[str],
) -> None:
    """The flat ``hc_attn_*`` names load with no rename in the weight adapter.

    Also the ``hc_head`` gate: GLM's final collapse is an unweighted mean, so a
    site owns exactly three tensors and the model owns no collapse weight at
    all.
    """
    assert site_weight_names == {
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "norm.weight",
    }


def test_mapping_matches_reference(
    max_outputs: tuple[torch.Tensor, ...],
    reference_outputs: tuple[torch.Tensor, ...],
) -> None:
    """``post`` and ``comb`` in float32, and the normalized collapse."""
    post, comb, xs, _ = max_outputs
    ref_post, ref_comb, ref_xs, _ = reference_outputs

    assert post.dtype == torch.float32
    assert comb.dtype == torch.float32
    torch.testing.assert_close(
        post, ref_post, rtol=MAPPING_RTOL, atol=MAPPING_ATOL
    )
    torch.testing.assert_close(
        comb, ref_comb, rtol=MAPPING_RTOL, atol=MAPPING_ATOL
    )
    _assert_bf16_close(xs, ref_xs)


def test_write_back_matches_reference(
    streams: torch.Tensor,
    sublayer_output: torch.Tensor,
    max_outputs: tuple[torch.Tensor, ...],
    reference_outputs: tuple[torch.Tensor, ...],
) -> None:
    """``streams' = post * y + comb^T @ streams``, landing in bfloat16.

    The reference rounds three times -- the outer product, the batched matmul,
    and their sum -- and MAX rounds once on AMD (float32 kernel) or the same
    three times elsewhere, so agreeing to a fixed tolerance is only half the
    claim. The second assertion is the load-bearing one: against the same
    expression evaluated in float64 and rounded once, MAX is no further from the
    truth than the reference is, which rules out a tolerance wide enough to hide
    a real error.
    """
    post, comb, _, new_streams = max_outputs
    _, _, _, ref_streams = reference_outputs

    assert new_streams.shape == (TOKENS, HC_MULT, HIDDEN_SIZE)
    _assert_bf16_close(new_streams, ref_streams)

    exact = (
        post.double().unsqueeze(-1) * sublayer_output.double().unsqueeze(-2)
        + torch.matmul(comb.double().transpose(-1, -2), streams.double())
    ).bfloat16()
    max_error = (new_streams.float() - exact.float()).abs().max().item()
    reference_error = (ref_streams.float() - exact.float()).abs().max().item()
    assert max_error <= 1.5 * reference_error, (
        f"MAX write-back is {max_error:.3e} from the float64 result while the "
        f"reference is {reference_error:.3e}"
    )


@pytest.mark.parametrize(
    ("hc_mult", "hidden_size", "supported"),
    [
        (HC_MULT, HIDDEN_SIZE, True),
        (HC_MULT, 1024, True),
        (HC_MULT, 1536, False),
        (3, HIDDEN_SIZE, False),
    ],
)
def test_write_back_falls_back_outside_the_kernel_limits(
    hc_mult: int, hidden_size: int, supported: bool
) -> None:
    """Shapes ``mhc_post`` rejects build the batched-matmul path instead."""
    hc = HyperConnection(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        dtype=DType.bfloat16,
        sinkhorn_iters=SINKHORN_ITERS,
        eps=HC_EPS,
        rms_norm_eps=RMS_NORM_EPS,
        name="hc_attn",
    )
    device = DeviceRef.GPU()
    with Graph(
        "Glm5NextWriteBackRouting",
        input_types=(
            TensorType(
                DType.bfloat16, ("t", hc_mult, hidden_size), device=device
            ),
            TensorType(DType.bfloat16, ("t", hidden_size), device=device),
            TensorType(DType.float32, ("t", hc_mult), device=device),
            TensorType(DType.float32, ("t", hc_mult, hc_mult), device=device),
        ),
    ) as graph:
        streams, y, post, comb = (v.tensor for v in graph.inputs)
        mixing = StreamMixing(post=post, comb=comb, xs=y)
        graph.output(hc.write_back(streams, y, mixing))
    uses_kernel = supported and accelerator_api() == "hip"
    assert ("mo.mhc.post" in str(graph)) == uses_kernel


def _on_gfx950() -> bool:
    if accelerator_api() != "hip":
        return False
    return Accelerator(0).architecture_name.startswith("gfx950")


@pytest.mark.parametrize("with_norm", [False, True])
@pytest.mark.parametrize(
    ("hc_mult", "hidden_size", "supported"),
    [
        (HC_MULT, HIDDEN_SIZE, True),
        (HC_MULT, 1024, True),
        (HC_MULT, 1536, False),
        (8, 1024, False),
    ],
)
def test_mapping_uses_the_fused_kernel_on_gfx950(
    hc_mult: int, hidden_size: int, supported: bool, with_norm: bool
) -> None:
    """``mo.mhc.pre`` replaces the mapping chain where the kernel runs."""
    hc = HyperConnection(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        dtype=DType.bfloat16,
        sinkhorn_iters=SINKHORN_ITERS,
        eps=HC_EPS,
        rms_norm_eps=RMS_NORM_EPS,
        name="hc_attn",
    )
    device = DeviceRef.GPU()
    with Graph(
        "Glm5NextMappingRouting",
        input_types=(
            TensorType(
                DType.bfloat16, ("t", hc_mult, hidden_size), device=device
            ),
            TensorType(DType.bfloat16, (hidden_size,), device=device),
        ),
    ) as graph:
        streams, gamma = (v.tensor for v in graph.inputs)
        mixing = hc(streams, gamma if with_norm else None)
        graph.output(mixing.post, mixing.comb, mixing.xs)
    uses_kernel = supported and _on_gfx950()
    text = str(graph)
    assert ("mo.mhc.pre" in text) == uses_kernel
    assert ("mo.hyper_connection.gates" in text) != uses_kernel


class _FusedAndChainSites(Module):
    """The site twice on shared weights: as routed, and forced onto the chain.

    Off gfx950 both halves are the chain, so they must agree bitwise there.
    """

    def __init__(self, with_norm: bool) -> None:
        super().__init__()
        self.with_norm = with_norm
        self.hc = HyperConnection(
            hidden_size=HIDDEN_SIZE,
            hc_mult=HC_MULT,
            dtype=DType.bfloat16,
            sinkhorn_iters=SINKHORN_ITERS,
            eps=HC_EPS,
            rms_norm_eps=RMS_NORM_EPS,
            name="hc_attn",
        )
        self.norm = RMSNorm(
            HIDDEN_SIZE,
            DType.bfloat16,
            RMS_NORM_EPS,
            multiply_before_cast=False,
        )

    def __call__(self, streams: TensorValue) -> tuple[TensorValue, ...]:
        gamma = self.norm.weight if self.with_norm else None
        routed = self.hc(streams, gamma)
        with patch.object(
            hyper_connection_module, "accelerator_api", return_value="cuda"
        ):
            chain = self.hc(streams, gamma)
        return (
            routed.post,
            routed.comb,
            routed.xs,
            chain.post,
            chain.comb,
            chain.xs,
        )


# The fused kernel applies the stream RMS after the projection rather than
# before it, so it rounds differently from the chain: the gates are held to the
# fused op's own gate and the bf16 sequence to its value tolerance, both
# against float64 (`max/tests/integration/nn/test_mhc_pre_gpu.py`).
FUSED_GATE_ATOL = 1e-4
COLLAPSED_RTOL = 2.0**-7
NORMED_RTOL = 2.0**-6
VALUE_ATOL_FRAC = 1e-2


def _value_error(
    actual: torch.Tensor, expected: torch.Tensor, rtol: float
) -> float:
    """Returns the worst error over ``rtol * |ref|``, in units of the atol."""
    assert torch.isfinite(actual).all()
    ref = expected.double()
    atol = VALUE_ATOL_FRAC * ref.square().mean().sqrt().item()
    excess = (actual.double() - ref).abs() - rtol * ref.abs()
    return excess.max().item() / atol


def _gate_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.double() - expected.double()).abs().max().item()


@pytest.mark.parametrize("with_norm", [False, True])
def test_fused_mapping_matches_chain_and_float64(
    weights: dict[str, torch.Tensor], with_norm: bool
) -> None:
    """The routed site agrees with the chain and with float64 at every T.

    The float64 gate is the fused op's. Off gfx950 the routed site is the chain
    itself, whose float32 matmul is not held to that gate on every GPU, so the
    test only requires the two halves to be identical there.
    """
    site = _FusedAndChainSites(with_norm)
    site.load_state_dict(weights)
    session = InferenceSession(devices=[Accelerator(0)])
    graph = Graph(
        "Glm5NextFusedMapping",
        site,
        input_types=(
            TensorType(
                DType.bfloat16,
                ("tokens", HC_MULT, HIDDEN_SIZE),
                device=DeviceRef.GPU(),
            ),
        ),
    )
    fused = _on_gfx950()
    assert ("mo.mhc.pre" in str(graph)) == fused
    compiled = session.load(graph, weights_registry=site.state_dict())
    rtol = NORMED_RTOL if with_norm else COLLAPSED_RTOL

    for tokens in (1, 8, 128, 1024):
        generator = torch.Generator().manual_seed(tokens)
        streams = torch.randn(
            tokens, HC_MULT, HIDDEN_SIZE, generator=generator
        ).bfloat16()
        outputs = compiled.execute(streams.cuda().contiguous())
        post, comb, xs, chain_post, chain_comb, chain_xs = (
            torch.from_dlpack(out).cpu() for out in outputs
        )

        ref_post, ref_comb, ref_xs = mhc_mapping(
            streams,
            weights["hc_attn_fn"],
            weights["hc_attn_base"],
            weights["hc_attn_scale"],
            hc_mult=HC_MULT,
            sinkhorn_iters=SINKHORN_ITERS,
            hc_eps=HC_EPS,
            rms_norm_eps=RMS_NORM_EPS,
            mapping_dtype=torch.float64,
        )
        if with_norm:
            ref_xs = rms_norm(ref_xs, weights["norm.weight"], RMS_NORM_EPS)

        errors = {
            "post vs fp64": _gate_error(post, ref_post),
            "comb vs fp64": _gate_error(comb, ref_comb),
            "post vs chain": _gate_error(post, chain_post),
            "comb vs chain": _gate_error(comb, chain_comb),
            "xs vs fp64": _value_error(xs, ref_xs, rtol),
            "xs vs chain": _value_error(xs, chain_xs, rtol),
        }
        print(
            f"T={tokens} norm={with_norm}: "
            + ", ".join(f"{k} {v:.3e}" for k, v in errors.items())
        )
        if not fused:
            assert torch.equal(post, chain_post)
            assert torch.equal(comb, chain_comb)
            assert torch.equal(xs, chain_xs)
            continue
        for name, error in errors.items():
            limit = 1.0 if name.startswith("xs") else FUSED_GATE_ATOL
            assert error <= limit, f"T={tokens} {name}: {error:.3e}"


# `comb / (comb.sum + hc_eps)` has its fixed point at `1 / (1 + hc_eps)`, so a
# normalized axis lands `hc_eps` below 1 by construction and never closer.
# Measured: 1.13e-6 on the columns.
SINKHORN_COLUMN_ATOL = 2 * HC_EPS

# The rows are one pass staler than the columns and 20 iterations do not close
# the gap on a 4x4 that softmax has made fairly peaked. Measured: 2.14e-2. This
# is the recorded bound, not a target.
SINKHORN_ROW_ATOL = 5e-2


def test_sinkhorn_ends_on_a_column_pass(
    max_outputs: tuple[torch.Tensor, ...],
) -> None:
    """``comb`` is column-stochastic, and only approximately row-stochastic.

    That asymmetry is the point of the test rather than a defect of it. The
    projection runs one column pass, then ``hc_sinkhorn_iters - 1`` (row,
    column) pairs, so it ends on a column pass -- and the write-back contracts
    over the *first* stream axis, which means it is the column sums that make it
    a convex combination of the incoming streams. Swapping the pass order
    normalizes the wrong axis, which no shape check and no logit tolerance would
    catch, so this test requires the columns to be tight *and* the rows not to
    be.
    """
    _, comb, _, _ = max_outputs
    column_error = (comb.sum(dim=-2) - 1).abs().max().item()
    row_error = (comb.sum(dim=-1) - 1).abs().max().item()

    assert column_error <= SINKHORN_COLUMN_ATOL, (
        f"columns are not stochastic: {column_error:.3e}"
    )
    assert row_error <= SINKHORN_ROW_ATOL, (
        f"rows are further from stochastic than recorded: {row_error:.3e}"
    )
    assert row_error > 100 * SINKHORN_COLUMN_ATOL, (
        "rows are as tight as the columns, which means the projection ended on "
        f"a row pass: row {row_error:.3e} vs column {column_error:.3e}"
    )


def test_float32_mapping_is_load_bearing(
    weights: dict[str, torch.Tensor],
    streams: torch.Tensor,
    sublayer_output: torch.Tensor,
    max_outputs: tuple[torch.Tensor, ...],
) -> None:
    """Placebo check: a bfloat16 mapping diverges, so float32 was necessary.

    Everything in the mapping is float32 in the reference and in MAX. This test
    exists so that the choice is recorded as measured rather than assumed: it
    runs the same site with the mapping in bfloat16 and requires the resulting
    error to be far outside the tolerance the float32 path is held to, both at
    one site and compounded over the model's 90.
    """
    reference_kwargs = {
        "hc_mult": HC_MULT,
        "sinkhorn_iters": SINKHORN_ITERS,
        "hc_eps": HC_EPS,
        "rms_norm_eps": RMS_NORM_EPS,
    }
    fp32_post, fp32_comb, _ = mhc_mapping(
        streams,
        weights["hc_attn_fn"],
        weights["hc_attn_base"],
        weights["hc_attn_scale"],
        **reference_kwargs,  # type: ignore[arg-type]
    )
    bf16_post, bf16_comb, _ = mhc_mapping(
        streams,
        weights["hc_attn_fn"],
        weights["hc_attn_base"],
        weights["hc_attn_scale"],
        mapping_dtype=torch.bfloat16,
        **reference_kwargs,  # type: ignore[arg-type]
    )

    comb_error = (bf16_comb.float() - fp32_comb).abs().max().item()
    post_error = (bf16_post.float() - fp32_post).abs().max().item()
    # MAX's float32 mapping agrees with the reference to here; the bfloat16
    # mapping must be at least an order of magnitude worse for the fp32 choice
    # to have been doing anything.
    _, max_comb, _, _ = max_outputs
    fp32_agreement = (max_comb - fp32_comb).abs().max().item()
    assert comb_error > 10 * max(fp32_agreement, MAPPING_ATOL), (
        f"bfloat16 comb error {comb_error:.3e} is not meaningfully worse than "
        f"MAX's float32 agreement {fp32_agreement:.3e}"
    )
    assert post_error > 10 * MAPPING_ATOL

    # `comb` is applied to the residual at every site, so a bias in the
    # projection compounds over the depth. Iterate the write-back with each
    # mapping and compare where the residual ends up.
    fp32_streams = streams.clone()
    bf16_streams = streams.clone()
    for _ in range(SITES):
        fp32_streams = mhc_write_back(
            fp32_streams, sublayer_output, fp32_post, fp32_comb
        )
        bf16_streams = mhc_write_back(
            bf16_streams, sublayer_output, bf16_post, bf16_comb
        )
    compounded = (
        (bf16_streams.float() - fp32_streams.float()).abs().max().item()
    )
    single_site = (
        (
            mhc_write_back(streams, sublayer_output, bf16_post, bf16_comb)
            .float()
            .sub(
                mhc_write_back(
                    streams, sublayer_output, fp32_post, fp32_comb
                ).float()
            )
        )
        .abs()
        .max()
        .item()
    )
    assert compounded > single_site, (
        f"mapping error did not compound over {SITES} sites: "
        f"{single_site:.3e} -> {compounded:.3e}"
    )
