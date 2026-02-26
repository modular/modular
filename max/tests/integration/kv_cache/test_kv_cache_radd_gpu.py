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

from dataclasses import dataclass

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.buffer_utils import cast_tensor_to
from max.kv_cache import PagedKVCacheManager
from max.nn.kernels import kv_cache_ragged_radd
from max.nn.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from test_common.context_utils import create_text_context


@dataclass(frozen=True)
class KVCacheRaddModel:
    """Model containing a single kv_cache_ragged_radd op."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index to apply the radd operation to."""

    def __call__(
        self,
        a: TensorValue,
        input_row_offsets: TensorValue,
        batch_offset: TensorValue,
        kv_blocks: TensorValue,
        cache_lengths: TensorValue,
        lookup_table: TensorValue,
        max_lengths: TensorValue,
    ) -> None:
        """Apply the radd operation to the KV cache."""
        kv_cache_ragged_radd(
            kv_params=self.kv_params,
            a=a,
            kv_collection=PagedCacheValues(
                kv_blocks=kv_blocks.buffer,
                cache_lengths=cache_lengths.tensor,
                lookup_table=lookup_table.tensor,
                max_lengths=max_lengths.tensor,
            ),
            input_row_offsets=input_row_offsets,
            batch_offset=batch_offset,
            layer_idx=self.layer_idx,
        )


def test_kv_cache_radd_basic() -> None:
    """Test basic functionality of kv_cache_ragged_radd."""
    dtype = DType.bfloat16
    batch_size = 2
    prompt_lens = [10, 20]
    num_active_loras = 1
    layer_idx = 1
    num_layers = 2
    seq_len = 100
    max_seq_len = 1024
    device = Accelerator()
    session = InferenceSession(devices=[device])

    kv_params = KVCacheParams(
        n_kv_heads=8,
        head_dim=128,
        dtype=dtype,
        cache_strategy="paged",
        num_layers=num_layers,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )

    kv_manager = PagedKVCacheManager(
        kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    # Calculate total length and offsets
    total_length = sum(prompt_lens)
    a_length = sum(prompt_lens[batch_size - num_active_loras :])
    input_row_offsets_np = np.array(
        [0, prompt_lens[0], total_length], dtype=np.uint32
    )
    batch_offset = batch_size - num_active_loras

    # Stage the fetch op + custom kv_cache_ragged_radd op graph
    a_type = TensorType(
        dtype,
        ["total_length", kv_params.n_kv_heads * kv_params.head_dim * 2],
        device=DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_length"], device=DeviceRef.GPU()
    )
    batch_offset_type = TensorType(DType.uint32, [], device=DeviceRef.CPU())

    graph = Graph(
        "kv_cache_radd_test",
        forward=KVCacheRaddModel(kv_params, layer_idx),
        input_types=[
            a_type,
            input_row_offsets_type,
            batch_offset_type,
            *kv_params.get_symbolic_inputs()[0],
        ],
    )

    # Compile and init the model
    model = session.load(graph)

    # Create contexts and claim seq_ids in cache
    batch = []
    for i in range(batch_size):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        batch.append(context)

    kv_inputs = kv_manager.runtime_inputs([batch])[0]

    a_np = np.ones(
        (a_length, kv_params.n_kv_heads * kv_params.head_dim * 2),
        dtype=np.float32,
    )
    a_data = cast_tensor_to(Buffer.from_numpy(a_np), dtype).to(device)
    input_row_offsets_data = Buffer.from_numpy(input_row_offsets_np).to(device)

    output = model(a_data, input_row_offsets_data, batch_offset, *kv_inputs)

    # simple smoke test, we do more thorough testing in the test_lora_gpu.py test
    assert output is not None
