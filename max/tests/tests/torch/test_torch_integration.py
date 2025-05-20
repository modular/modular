# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import pytest
import torch
from max.torch import CustomOpLibrary


def torch_grayscale(img: torch.Tensor) -> torch.Tensor:
    rgb_mask = torch.as_tensor([0.21, 0.71, 0.07], dtype=torch.float32)

    img = img.to(torch.float32) * rgb_mask

    result = torch.minimum(
        img.sum(dim=-1, dtype=torch.float32),
        torch.as_tensor([255], dtype=torch.float32),
    ).to(torch.uint8)

    return result


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_grayscale(op_library: CustomOpLibrary, backend: str):
    @torch.compile(backend=backend)
    def grayscale(pic):
        result = pic.new_empty(pic.shape[:-1])
        op_library.grayscale(result, pic)
        return result

    img = (torch.rand(64, 64, 3) * 255).to(torch.uint8)
    result = grayscale(img)

    # For some reason we differ by 1 in a small number of locations.
    np.testing.assert_allclose(
        result,
        torch_grayscale(img),
        equal_nan=True,
        rtol=1e-4,
        atol=1,
    )


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_binary_add(op_library: CustomOpLibrary, backend: str):
    myadd_kernel = op_library.myadd

    @torch.compile(backend=backend, fullgraph=True)
    def myadd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        C = torch.zeros(A.shape)
        myadd_kernel(C, A, B)
        return C

    A = torch.rand(64, 64, dtype=torch.float32)
    B = torch.rand(64, 64, dtype=torch.float32)
    C = myadd(A, B)

    # For some reason we differ by 1 in a small number of locations.
    np.testing.assert_allclose(
        C,
        A + B,
        equal_nan=True,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_binary_add_multiple_sizes(op_library: CustomOpLibrary, backend: str):
    myadd_kernel = op_library.myadd

    @torch.compile(backend=backend, fullgraph=True)
    def myadd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        C = torch.zeros(A.shape)
        myadd_kernel(C, A, B)
        return C

    A = torch.rand(64, 64, dtype=torch.float32)
    B = torch.rand(64, 64, dtype=torch.float32)
    C = myadd(A, B)

    # For some reason we differ by 1 in a small number of locations.
    np.testing.assert_allclose(
        C,
        A + B,
        equal_nan=True,
        rtol=1e-4,
        atol=1e-4,
    )

    A = torch.rand(128, 128, dtype=torch.float32)
    B = torch.rand(128, 128, dtype=torch.float32)
    C = myadd(A, B)

    # For some reason we differ by 1 in a small number of locations.
    np.testing.assert_allclose(
        C,
        A + B,
        equal_nan=True,
        rtol=1e-4,
        atol=1e-4,
    )
