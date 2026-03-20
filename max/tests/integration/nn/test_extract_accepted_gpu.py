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

import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import extract_accepted_hs


def _run_extract_hs(
    hs: torch.Tensor,
    hs_offsets: torch.Tensor,
    first_rejected: torch.Tensor,
    num_draft_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile and run the extract_accepted_hs graph on GPU."""
    device = Accelerator(0)
    device_ref = DeviceRef.GPU()
    input_types = [
        TensorType(DType.float32, ["total_hs", 8], device=device_ref),
        TensorType(DType.uint32, ["offsets_len"], device=device_ref),
        TensorType(DType.int64, ["local_batch"], device=device_ref),
        TensorType(DType.int64, [1], device=DeviceRef.CPU()),
    ]
    with Graph("test_extract_accepted_hs", input_types=input_types) as graph:
        hs_in, hs_offsets_in, fr_in, num_draft_in = graph.inputs
        accepted_hs, accepted_offsets = extract_accepted_hs(
            hs_in.tensor,
            hs_offsets_in.tensor,
            fr_in.tensor,
            num_draft_in.tensor,
        )
        graph.output(accepted_hs, accepted_offsets)

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)
    results = compiled.execute(
        Buffer.from_dlpack(hs).to(device),
        Buffer.from_dlpack(hs_offsets).to(device),
        Buffer.from_dlpack(first_rejected).to(device),
        Buffer.from_dlpack(num_draft_tokens),
    )
    accepted_hs_out = torch.from_dlpack(results[0]).to("cpu")
    accepted_offsets_out = torch.from_dlpack(results[1]).to("cpu")
    return accepted_hs_out, accepted_offsets_out


def test_extract_accepted_hs_prefill() -> None:
    """K=0: passthrough, all HS returned unchanged."""
    hidden = 8
    hs = torch.randn(5, hidden, dtype=torch.float32)
    hs_offsets = torch.tensor([0, 3, 5], dtype=torch.uint32)
    first_rejected = torch.tensor([0, 0], dtype=torch.int64)
    num_draft = torch.tensor([0], dtype=torch.int64)

    accepted_hs, accepted_offsets = _run_extract_hs(
        hs, hs_offsets, first_rejected, num_draft
    )

    assert torch.all(accepted_offsets == hs_offsets), (
        f"{accepted_offsets} != {hs_offsets}"
    )
    total = int(hs_offsets[-1])
    assert torch.allclose(accepted_hs[:total], hs), (
        "HS mismatch in prefill passthrough"
    )


def test_extract_accepted_hs_decode_all_accepted() -> None:
    """K>0, all accepted: extracts HS at positions [0..K] per request."""
    hidden = 8
    K = 3
    hs = torch.randn(8, hidden, dtype=torch.float32)
    hs_offsets = torch.tensor([0, 4, 8], dtype=torch.uint32)
    first_rejected = torch.tensor([3, 3], dtype=torch.int64)
    num_draft = torch.tensor([K], dtype=torch.int64)

    accepted_hs, accepted_offsets = _run_extract_hs(
        hs, hs_offsets, first_rejected, num_draft
    )

    expected_offsets = torch.tensor([0, 4, 8], dtype=torch.uint32)
    assert torch.all(accepted_offsets == expected_offsets), (
        f"{accepted_offsets} != {expected_offsets}"
    )
    total = int(expected_offsets[-1])
    expected_hs = torch.cat([hs[0:4], hs[4:8]])
    assert torch.allclose(accepted_hs[:total], expected_hs), (
        "HS mismatch decode all"
    )


def test_extract_accepted_hs_decode_mixed() -> None:
    """K>0, mixed fr values including fr=0."""
    hidden = 8
    K = 2
    hs = torch.randn(6, hidden, dtype=torch.float32)
    hs_offsets = torch.tensor([0, 3, 6], dtype=torch.uint32)
    first_rejected = torch.tensor([0, 2], dtype=torch.int64)
    num_draft = torch.tensor([K], dtype=torch.int64)

    accepted_hs, accepted_offsets = _run_extract_hs(
        hs, hs_offsets, first_rejected, num_draft
    )

    expected_offsets = torch.tensor([0, 1, 4], dtype=torch.uint32)
    assert torch.all(accepted_offsets == expected_offsets), (
        f"{accepted_offsets} != {expected_offsets}"
    )
    total = int(expected_offsets[-1])
    expected_hs = torch.cat([hs[0:1], hs[3:6]])
    assert torch.allclose(accepted_hs[:total], expected_hs), (
        "HS mismatch decode mixed"
    )
