# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import torch
import torch.nn as nn
import torch.nn.functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import MLPV2

DTYPE = DType.float32
TORCH_DTYPE = torch.float32

ACTIVATION_FUNCTION = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
}


def generate_tensor(
    shape: tuple[int, int], dtype: DType, seed: int = 1234
) -> torch.Tensor:
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return torch.randn(shape, dtype=dtype)


def torch_linear(weight, **kwargs) -> nn.Linear:
    linear = nn.Linear(*weight.shape, **kwargs)
    linear.weight = nn.Parameter(weight)
    return linear


class TorchMLP(nn.Module):
    def __init__(
        self,
        gate_proj: torch.Tensor,
        down_proj: torch.Tensor,
        up_proj: torch.Tensor,
        activation_function: str = "silu",
    ):
        super().__init__()
        self.gate_proj = torch_linear(gate_proj, bias=False)
        self.down_proj = torch_linear(down_proj, bias=False)
        self.up_proj = torch_linear(up_proj, bias=False)
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            ACTIVATION_FUNCTION[self.activation_function](self.gate_proj(x))
            * self.up_proj(x)
        )


def mlp_output(
    gate_proj: torch.Tensor,
    down_proj: torch.Tensor,
    up_proj: torch.Tensor,
    x: torch.Tensor,
    activation_function: str,
    dtype: DType,
    is_gpu: bool = False,
):
    state_dict = {
        "gate_proj.weight": gate_proj.cpu(),
        "down_proj.weight": down_proj.cpu(),
        "up_proj.weight": up_proj.cpu(),
    }

    mlp = MLPV2(
        dtype,
        None,
        gate_proj.shape[1],
        gate_proj.shape[0],
        activation_function=activation_function,
    )
    mlp.load_state_dict(state_dict)

    session = (
        InferenceSession(devices=[Accelerator(0)])
        if is_gpu
        else InferenceSession()
    )
    graph = Graph(
        "MLP",
        mlp,
        input_types=(
            TensorType(
                dtype,
                (
                    x.shape[0],
                    x.shape[1],
                ),
                device=DeviceRef.GPU() if is_gpu else DeviceRef.CPU(),
            ),
        ),
    )

    compiled = session.load(graph, weights_registry=mlp.state_dict())
    return compiled.execute(x)


def compare_mlp_outputs(
    hidden_dim: int,
    dim: int,
    activation_function: str,
    torch_dtype: torch.dtype,
    dtype: DType,
    is_gpu: bool = False,
):
    gate_proj = generate_tensor((hidden_dim, dim), torch_dtype, seed=42)
    down_proj = generate_tensor((dim, hidden_dim), torch_dtype, seed=43)
    up_proj = generate_tensor((hidden_dim, dim), torch_dtype, seed=44)
    x = generate_tensor((1, dim), torch_dtype, seed=45)

    max_output = mlp_output(
        gate_proj,
        down_proj,
        up_proj,
        x,
        activation_function,
        dtype,
        is_gpu=is_gpu,
    )

    device = "cuda" if is_gpu else "cpu"
    torch_output = (
        TorchMLP(
            gate_proj.to(device),
            down_proj.to(device),
            up_proj.to(device),
            activation_function,
        )(x.to(device))
        .detach()
        .to(torch_dtype)
        .to(device)
    )

    torch.testing.assert_close(
        torch_output,
        torch.from_dlpack(max_output[0]).to(torch_dtype),
        rtol=2e-3,
        atol=3 * torch.finfo(torch_dtype).eps,
    )


def test_mlp():
    compare_mlp_outputs(1024, 1024, "silu", torch.float32, DType.float32)
    compare_mlp_outputs(2048, 1024, "gelu", torch.float32, DType.float32)
    compare_mlp_outputs(1024, 512, "gelu_tanh", torch.float32, DType.float32)
    compare_mlp_outputs(256, 1024, "tanh", torch.float32, DType.float32)

    # TODO: Investigate why the following tests fail
    # compare_mlp_outputs(4096, 2048, "relu", TORCH_DTYPE, DTYPE)
    # compare_mlp_outputs(2048, 4096, "sigmoid", TORCH_DTYPE, DTYPE)
