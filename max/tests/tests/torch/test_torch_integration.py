# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import pytest
import torch
from max import mlir
from max.torch import CustomOpLibrary


def torch_grayscale(img: torch.Tensor) -> torch.Tensor:
    rgb_mask = torch.as_tensor([0.21, 0.71, 0.07], dtype=torch.float32)

    img = img.to(torch.float32) * rgb_mask

    result = torch.minimum(
        img.sum(dim=-1, dtype=torch.float32),
        torch.as_tensor([255], dtype=torch.float32),
    ).to(torch.uint8)

    return result


def test_missing_operation(op_library: CustomOpLibrary):
    with pytest.raises(AttributeError):
        _ = op_library.some_kernel_that_doesnt_exist[{"const": 10}]


def test_unsupported_arg_type_error(op_library: CustomOpLibrary):
    # Attempting to access the unsupported_type_op should raise ValueError
    # because it has a String parameter which is not a supported type.
    with pytest.raises(
        ValueError,
        match="Unsupported argument type 'stdlib::String' in custom op 'unsupported_type_op'.",
    ):
        _ = op_library.unsupported_type_op


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


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_parameters(op_library: CustomOpLibrary, backend: str):
    parameter_increment_42 = op_library.parameter_increment[{"increment": 42}]

    @torch.compile(backend=backend)
    def increment_42(input):
        result = torch.empty_like(input)
        parameter_increment_42(result, input)
        return result

    A = torch.rand(64, 64, dtype=torch.float32)
    C = increment_42(A)

    # For some reason we differ by 1 in a small number of locations.
    np.testing.assert_allclose(
        C,
        A + 42,
        equal_nan=True,
        rtol=1e-4,
        atol=1e-4,
    )

    parameter_increment_17 = op_library.parameter_increment[{"increment": 17}]

    @torch.compile(backend=backend)
    def increment_17(input):
        result = torch.empty_like(input)
        parameter_increment_17(result, input)
        return result

    A = torch.rand(64, 64, dtype=torch.float32)
    C = increment_17(A)

    # For some reason we differ by 1 in a small number of locations.
    np.testing.assert_allclose(
        C,
        A + 17,
        equal_nan=True,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_scalar_add(op_library: CustomOpLibrary, backend: str):
    scalar_add_kernel = op_library.scalar_add

    @torch.compile(backend=backend, fullgraph=True)
    def add_scalars(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(a)
        scalar_add_kernel(result, a, b)
        return result

    # Test with float32 scalars
    a = torch.tensor(3.14, dtype=torch.float32)
    b = torch.tensor(2.71, dtype=torch.float32)
    result = add_scalars(a, b)

    expected = a + b
    np.testing.assert_allclose(
        result.item(),
        expected.item(),
        equal_nan=True,
        rtol=1e-6,
        atol=1e-6,
    )

    # Test with int32 scalars
    a_int = torch.tensor(42, dtype=torch.int32)
    b_int = torch.tensor(17, dtype=torch.int32)
    result_int = add_scalars(a_int, b_int)

    expected_int = a_int + b_int
    assert result_int.item() == expected_int.item()


# This just gut-checks that we run other tests without an active MLIR context
def test_GEX_2285(op_library: CustomOpLibrary):
    assert not mlir.Context.current
