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
"""Tests that a PrintHook'd v3 module compiles against the weights it is given."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from max import driver
from max.driver.buffer import load_max_buffer
from max.engine import InferenceSession
from max.engine.api import PrintStyle
from max.experimental import functional as F
from max.experimental.nn.module import Module, module_dataclass
from max.experimental.tensor import Tensor, TensorType, defaults
from max.nn.hooks.print_hook import PrintHook

WEIGHT = np.ones((4, 4), dtype=np.float32)
X = np.arange(12, dtype=np.float32).reshape(3, 4)


@module_dataclass
class Lin(Module[[Tensor], Tensor]):
    weight: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


@module_dataclass
class Outer(Module[[Tensor], Tensor]):
    lin: Lin

    def forward(self, x: Tensor) -> Tensor:
        return self.lin(x) + 1


def test_name_layers_v3_compiles_with_given_weights(tmp_path: Path) -> None:
    """The hooked trace reads ``compile(weights=...)``, not the init values."""
    with F.lazy():
        model = Outer(lin=Lin(weight=Tensor.zeros([4, 4])))
    hook = PrintHook()
    hook.name_layers_v3(model)

    dtype, mesh = defaults()
    device = mesh.devices[0]
    session = InferenceSession(devices=list(mesh.devices))
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=tmp_path
    )
    compiled = model.compile(
        TensorType(dtype, [3, 4], device=device),
        weights={"lin.weight": driver.Buffer.from_numpy(WEIGHT).to(device)},
        session=session,
    )
    result = compiled(Tensor(storage=driver.Buffer.from_numpy(X).to(device)))
    hook.remove()

    np.testing.assert_allclose(result.to_numpy(), X @ WEIGHT.T + 1)
    dumped = load_max_buffer(tmp_path / "model.lin-output.max").to_numpy()
    np.testing.assert_allclose(dumped, X @ WEIGHT.T)
