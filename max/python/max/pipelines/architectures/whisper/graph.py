# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.abc import Mapping

from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import WeightData
from transformers import AutoConfig

from .encoder import WhisperEncoder


def build_graph(
    state_dict: Mapping[str, DLPackArray | WeightData],
    huggingface_config: AutoConfig,
    dtype: DType,
    device: DeviceRef,
) -> Graph:
    # Audio input_features.
    input_features_type = TensorType(
        DType.float32,
        shape=["batch_size", "num_mel_bins", "sequence_length"],
        device=DeviceRef.CPU(),
    )

    # Initialize Graph.
    with Graph(
        "whisper_audio_encoder", input_types=[input_features_type]
    ) as graph:
        model = WhisperEncoder(huggingface_config, dtype, device)
        model.load_state_dict(state_dict)
        input_features = graph.inputs[0]
        outputs = model(input_features=input_features.tensor)
        graph.output(*outputs)
    return graph
