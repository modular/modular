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


def test_distributed_fused_spec_signature_declares_hisparse_once() -> None:
    devices = [DeviceRef.GPU(i) for i in range(8)]
    spec = SpecDecodeInputTypeSpec(devices=devices, distributed=True)
    kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=64,
        num_layers=2,
        devices=devices,
        n_kv_heads=8,
    )
    sparse = (
        *(TensorType(DType.int32, ["batch", 12], device=d) for d in devices),
        *(TensorType(DType.uint32, ["batch", 113], device=d) for d in devices),
        TensorType(DType.uint8, ["mirror_bytes"], device=CPU),
    )
    base = build_spec_decode_input_types(spec, kv_params=kv, ep_input_types=EP)
    types = build_spec_decode_input_types(
        spec, kv_params=kv, ep_input_types=EP, hisparse_input_types=sparse
    )
    assert len(types) == len(base) + 17
    ep_end = next(
        i + len(EP)
        for i in range(len(base) - len(EP) + 1)
        if base[i : i + len(EP)] == EP
    )
    assert types[ep_end : ep_end + 17] == sparse
    with Graph("hisparse_fused_spec_signature", input_types=types) as graph:
        named = decode_spec_decode_input_values(
            graph.inputs,
            spec,
            kv_params=kv,
            ep_input_types=EP,
            hisparse_input_types=sparse,
        )
        assert len(named.hisparse) == 17
        graph.output()
