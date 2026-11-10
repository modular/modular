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
"""Where a spec-decode signature declares expert parallelism's inputs."""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
    decode_spec_decode_input_values,
)

CPU = DeviceRef.CPU()
EP = (
    BufferType(DType.int64, [3], device=CPU),
    TensorType(DType.int32, [5], device=CPU),
)


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=64,
        num_layers=2,
        devices=[CPU],
        n_kv_heads=8,
    )


def test_ep_inputs_follow_the_tail_of_a_signature_that_isnt_distributed() -> (
    None
):
    # A target with tensor-parallel attention and expert-parallel experts.
    spec = SpecDecodeInputTypeSpec(devices=[CPU], distributed=False)
    kv = _kv_params()
    types = build_spec_decode_input_types(spec, kv_params=kv, ep_input_types=EP)
    without = build_spec_decode_input_types(spec, kv_params=kv)
    assert types == (*without, *EP)
    with Graph("ep_inputs", input_types=types) as graph:
        named = decode_spec_decode_input_values(
            graph.inputs, spec, kv_params=kv, ep_input_types=EP
        )
        decoded = []
        for value in named.ep_inputs:
            t = value.type
            assert isinstance(t, (BufferType, TensorType))
            decoded.append((t.dtype, [int(d) for d in t.shape]))
        graph.output()
    assert decoded == [(DType.int64, [3]), (DType.int32, [5])]
