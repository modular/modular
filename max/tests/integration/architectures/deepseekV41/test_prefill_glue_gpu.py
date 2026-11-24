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
"""Checks the CSA2 compressor's bf16 projection against the float32 one.

With ``MAX_DSV41_COMPRESSOR_BF16`` on, the compressor multiplies the
checkpoint's bf16 ``wkv``/``wgate`` with float32 accumulation and output,
where it widened both operands to float32 before. The products of bf16
values are exact in both, so the two differ only in accumulation order and
each must match a float64 reference.
"""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator
from max.engine import InferenceSession, Model
from prefill_glue_cases import compressor_model, compressor_weights, hidden
from torch.utils.dlpack import from_dlpack


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])


def _outputs(model: Model, *args: torch.Tensor) -> list[torch.Tensor]:
    # Copied, so no comparison holds a view into the runtime's buffers.
    return [from_dlpack(t).clone() for t in model.execute(*args)]


@pytest.fixture(scope="module")
def compressors(
    session: InferenceSession,
) -> dict[int, tuple[dict[str, torch.Tensor], Model, Model]]:
    out = {}
    for ratio in (1, 2):
        weights = compressor_weights(ratio)
        out[ratio] = (
            weights,
            compressor_model(session, ratio, False, weights),
            compressor_model(session, ratio, True, weights),
        )
    return out


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("rows", [1, 129, 8312])
def test_compressor_bf16_projection(
    compressors: dict[int, tuple[dict[str, torch.Tensor], Model, Model]],
    ratio: int,
    rows: int,
) -> None:
    weights, old, new = compressors[ratio]
    h = hidden(rows, rows)
    (raw_old,) = _outputs(old, h)
    (raw_new,) = _outputs(new, h)
    reference = (
        h.double()
        @ torch.cat([w.double().cuda() for w in weights.values()], dim=0).T
    )
    # Both multiply exact bf16 products and accumulate in float32.
    torch.testing.assert_close(raw_new.double(), reference, rtol=0, atol=1e-4)
    torch.testing.assert_close(raw_old.double(), reference, rtol=0, atol=1e-4)
