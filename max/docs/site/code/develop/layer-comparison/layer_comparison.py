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
# DOC: develop/layer-comparison.mdx

"""Localizes the first MAX layer that diverges from a PyTorch reference.

The functions follow the snippets in layer-comparison.mdx: capture the MAX
layer outputs, capture the PyTorch layer outputs through a layer map, save
both, then compare them layer by layer.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_F
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Embedding, Linear, Module, ModuleList, RMSNorm
from max.experimental.tensor import Tensor
from torch import nn

VOCAB = 32
HIDDEN = 8
NUM_LAYERS = 2
RMS_EPS = 1e-6
TOKEN_IDS = [[1, 7, 13, 3]]  # one sequence of four tokens
EPS = 1e-10  # Guards the relative-difference denominator against divide-by-zero

# Qualified paths of the MAX layers to capture, in execution order.
LAYER_NAMES = ["embed_tokens", "layers.0", "layers.1", "norm"]

# Keys are MAX layer names and values are PyTorch submodule paths. The
# reference model nests its decoder under ``model``, as Hugging Face causal
# language models do.
LAYER_MAP = {
    "embed_tokens": "model.embed_tokens",
    "layers.0": "model.layers.0",
    "layers.1": "model.layers.1",
    "norm": "model.norm",
}


# --- PyTorch reference model ----------------------------------------------- #
class ReferenceBlock(nn.Module):
    """A residual ``silu(linear(x))`` block."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch_F.silu(self.proj(x))


class ReferenceDecoder(nn.Module):
    """A two-layer decoder: ``embed_tokens`` -> ``layers`` -> ``norm``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList(
            [ReferenceBlock() for _ in range(NUM_LAYERS)]
        )
        self.norm = nn.RMSNorm(HIDDEN, eps=RMS_EPS)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class ReferenceModel(nn.Module):
    """Holds the decoder under ``model``, the layout of Hugging Face models."""

    def __init__(self) -> None:
        super().__init__()
        self.model = ReferenceDecoder()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


def build_reference_model() -> ReferenceModel:
    """Builds a seeded, random-weight reference model in float32.

    Returns:
        The reference model in evaluation mode.
    """
    torch.manual_seed(0)
    reference_model = ReferenceModel().eval().to(torch.float32)
    with torch.no_grad():
        # Scale weights up so activations are O(1) and the SiLU/GELU gap
        # moves cosine similarity, not only the absolute error.
        nn.init.normal_(reference_model.model.embed_tokens.weight, std=1.0)
        for block in reference_model.model.layers:
            nn.init.normal_(block.proj.weight, std=0.5)
    return reference_model


# --- MAX model, from max.experimental.nn building blocks ------------------- #
class Block(Module[[Tensor], Tensor]):
    """A residual ``activation(linear(x))`` block."""

    def __init__(self, activation: Callable[[Tensor], Tensor]) -> None:
        self.activation = activation
        self.proj = Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.activation(self.proj(x))


class MyModel(Module[[Tensor], Tensor]):
    """A two-layer decoder: ``embed_tokens`` -> ``layers`` -> ``norm``."""

    def __init__(
        self, activations: Sequence[Callable[[Tensor], Tensor]]
    ) -> None:
        self.embed_tokens = Embedding(VOCAB, dim=HIDDEN)
        self.layers = ModuleList(
            [Block(activation) for activation in activations]
        )
        self.norm = RMSNorm(HIDDEN, eps=RMS_EPS)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


# --- Capture MAX layer outputs --------------------------------------------- #
@contextmanager
def capture_layer_outputs(
    model: Module[..., Any], names: list[str]
) -> Iterator[dict[str, np.ndarray]]:
    """Records the output of each named layer while the model runs eagerly.

    Args:
        model: The MAX model to instrument.
        names: Qualified paths of the descendants to capture.

    Yields:
        A dictionary that maps each layer name to its captured output.
    """
    captured: dict[str, np.ndarray] = {}
    modules = dict(model.descendants)

    def wrap(name: str, forward: Callable[..., Any]) -> Callable[..., Any]:
        def forward_and_capture(*args: Any, **kwargs: Any) -> Any:
            output = forward(*args, **kwargs)
            # A layer can return a tuple; its first element is the hidden state.
            first = output[0] if isinstance(output, tuple) else output
            # NumPy has no bfloat16, the default dtype on accelerators.
            captured[name] = first.cast(DType.float32).to_numpy()
            return output

        return forward_and_capture

    try:
        for name in names:
            module = modules[name]
            # Shadow the class's forward() with the capturing wrapper.
            object.__setattr__(module, "forward", wrap(name, module.forward))
        yield captured
    finally:
        for name in names:
            vars(modules[name]).pop("forward", None)


def save_max_tensors(
    activations: Sequence[Callable[[Tensor], Tensor]],
    state_dict: dict[str, torch.Tensor],
    path: Path,
) -> None:
    """Runs the MAX model eagerly and saves each layer's output to ``path``.

    Args:
        activations: The activation function for each MAX decoder block.
        state_dict: The weights to load into the model.
        path: The ``.npz`` file to write.
    """
    # The model runs on the default device and dtype: an accelerator in
    # bfloat16 when one is present, otherwise the CPU in float32.
    model = MyModel(activations)
    # auto_cast loads the float32 reference weights into a bfloat16 model.
    model.load_state_dict(state_dict, auto_cast=True)
    input_ids = Tensor(TOKEN_IDS, dtype=DType.int64)
    with capture_layer_outputs(model, LAYER_NAMES) as max_tensors:
        model(input_ids)
    # NumPy's stub types **kwds as arrays and allow_pickle as bool, so an
    # array dict reads as a possible allow_pickle value. Name the keys' type.
    arrays: dict[str, Any] = max_tensors
    np.savez(path, **arrays)


# --- Capture PyTorch layer outputs ----------------------------------------- #
def save_reference_tensors(reference_model: nn.Module, path: Path) -> None:
    """Runs the reference model and saves each mapped layer's output to ``path``.

    Args:
        reference_model: The PyTorch model to run.
        path: The ``.pt`` file to write.
    """
    reference_tensors: dict[str, torch.Tensor] = {}

    def make_hook(name: str) -> Callable[..., None]:
        def hook(_module: object, _inputs: object, output: object) -> None:
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                reference_tensors[name] = output.detach().cpu()

        return hook

    handles = []
    try:
        for max_name, torch_name in LAYER_MAP.items():
            module = reference_model.get_submodule(torch_name)
            handles.append(module.register_forward_hook(make_hook(max_name)))

        with torch.no_grad():
            reference_model(input_ids=torch.tensor(TOKEN_IDS))

        torch.save(reference_tensors, path)
    finally:
        for handle in handles:
            handle.remove()


# --- Compare the layers ---------------------------------------------------- #
@dataclass
class LayerComparison:
    """Comparison metrics for one layer's output."""

    name: str
    cosine_similarity: float
    max_abs_diff: float
    max_rel_diff: float


def compare_layer(
    name: str, max_output: np.ndarray, reference: torch.Tensor
) -> LayerComparison:
    """Compares one layer's MAX output against the reference output.

    Args:
        name: The layer name.
        max_output: The layer's output from the MAX model.
        reference: The layer's output from the reference model.

    Returns:
        The comparison metrics for the layer.

    Raises:
        ValueError: If the two outputs hold different numbers of elements.
    """
    # Cast to float32 so small differences aren't obscured by rounding
    max_tensor = torch.tensor(max_output, dtype=torch.float32)
    reference = reference.cpu().to(torch.float32)

    if max_tensor.shape != reference.shape:
        if max_tensor.numel() != reference.numel():
            raise ValueError(
                f"{name}: shape mismatch, MAX={tuple(max_tensor.shape)}, "
                f"reference={tuple(reference.shape)}"
            )
        max_tensor = max_tensor.reshape(reference.shape)

    abs_diff = (max_tensor - reference).abs()
    rel_diff = abs_diff / reference.abs().clamp_min(EPS)

    cosine_similarity = torch.nn.functional.cosine_similarity(
        max_tensor.flatten(),
        reference.flatten(),
        dim=0,
    ).item()

    return LayerComparison(
        name=name,
        cosine_similarity=cosine_similarity,
        max_abs_diff=abs_diff.max().item(),
        max_rel_diff=rel_diff.max().item(),
    )


def print_report(results: list[LayerComparison]) -> None:
    """Prints one row of comparison metrics per layer.

    Args:
        results: The comparison for each layer, in execution order.
    """
    header = f"{'layer':<16}{'cosine':>12}{'max_abs':>14}{'max_rel':>14}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<16}{result.cosine_similarity:>12.6f}"
            f"{result.max_abs_diff:>14.3e}{result.max_rel_diff:>14.3e}"
        )


def main() -> None:
    """Compares the MAX model to the reference model layer by layer."""
    reference_model = build_reference_model()

    # Introduce a bug: the MAX model uses GELU in layer 1, where the
    # reference uses SiLU. The MAX names match the reference decoder's.
    activations = [F.silu, F.gelu]
    state_dict = reference_model.model.state_dict()

    with tempfile.TemporaryDirectory() as tmp:
        max_path = Path(tmp) / "max_tensors.npz"
        reference_path = Path(tmp) / "reference_tensors.pt"
        save_max_tensors(activations, state_dict, max_path)
        save_reference_tensors(reference_model, reference_path)

        # Load both sets of tensors.
        max_tensors = np.load(max_path)
        reference_tensors = torch.load(reference_path, weights_only=True)

        # Calculate and print the results.
        results = [
            compare_layer(name, max_tensors[name], tensor)
            for name, tensor in reference_tensors.items()
        ]
    print_report(results)

    # The first diverging layer localizes the bug: everything up to and
    # including layer 0 matches, while layer 1 and the downstream final norm
    # diverge. These assertions fail if an API change alters that result.
    by_name = {result.name: result for result in results}
    assert list(by_name) == LAYER_NAMES
    assert by_name["embed_tokens"].cosine_similarity > 0.9999
    assert by_name["layers.0"].cosine_similarity > 0.9999
    assert by_name["layers.1"].cosine_similarity < 0.9999
    assert by_name["norm"].cosine_similarity < 0.9999


if __name__ == "__main__":
    main()
