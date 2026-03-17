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

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache.utils import AttentionDispatchResolver

N_KV_HEADS = 8
TEST_CASES = [
    (1, 17, 128),
    (4, 17, 768),
    (16, 17, 2048),
    (64, 17, 8192),
]


def _build_reference_mha_model(
    session: InferenceSession, device: DeviceRef, n_kv_heads: int
) -> Model:
    with Graph(
        "decode_num_partitions_reference",
        input_types=[
            TensorType(DType.int64, shape=[2], device=DeviceRef.CPU())
        ],
    ) as graph:
        (request,) = graph.inputs
        (num_partitions,) = ops.custom(
            "mo.mha.decode.get_num_partitions",
            values=[request.tensor],
            out_types=[
                TensorType(DType.int64, shape=[1], device=DeviceRef.CPU())
            ],
            parameters={"n_kv_heads": n_kv_heads},
            device=device,
        )
        graph.output(num_partitions.tensor)
    return session.load(graph)


def _resolve_reference(
    model: Model, batch_size: int, max_cache_valid_length: int
) -> int:
    request = Buffer.from_numpy(
        np.array([batch_size, max_cache_valid_length], dtype=np.int64)
    )
    (output,) = model(request)
    return int(output.to_numpy()[0])


@pytest.fixture(scope="module")
def gpu_session() -> InferenceSession:
    return InferenceSession(devices=[CPU(), Accelerator()])


@pytest.fixture(scope="module")
def gpu_device_ref() -> DeviceRef:
    return DeviceRef.GPU()


@pytest.fixture(scope="module")
def mha_resolver(
    gpu_session: InferenceSession, gpu_device_ref: DeviceRef
) -> AttentionDispatchResolver:
    return AttentionDispatchResolver(
        session=gpu_session,
        device=gpu_device_ref,
        is_mla=False,
        n_kv_heads_per_device=N_KV_HEADS,
    )


@pytest.fixture(scope="module")
def reference_mha_model(
    gpu_session: InferenceSession, gpu_device_ref: DeviceRef
) -> Model:
    return _build_reference_mha_model(
        gpu_session, gpu_device_ref, n_kv_heads=N_KV_HEADS
    )


@pytest.mark.parametrize(
    ("batch_size", "max_prompt_length", "max_cache_valid_length"),
    TEST_CASES,
)
def test_mha_dispatch_resolver_matches_reference_graph(
    mha_resolver: AttentionDispatchResolver,
    reference_mha_model: Model,
    batch_size: int,
    max_prompt_length: int,
    max_cache_valid_length: int,
) -> None:
    expected_num_partitions = _resolve_reference(
        reference_mha_model, batch_size, max_cache_valid_length
    )

    metadata = mha_resolver(
        batch_size, max_prompt_length, max_cache_valid_length
    ).to_numpy()

    np.testing.assert_array_equal(
        metadata,
        np.array(
            [batch_size, 1, expected_num_partitions, max_cache_valid_length],
            dtype=np.int64,
        ),
    )
