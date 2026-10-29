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
"""The compile inputs the V4.1 pipeline model hands ``compile``.

``DeepseekV41Model._get_compile_input_types`` is the pipeline's own code path:
it asks the batch processor for its symbolic inputs and appends the model's
extra input types. The model needs a session and devices to construct, so this
builds one without ``__init__`` and sets the four attributes that method reads,
leaving the method and the batch processor real. Staging then appends one signal
buffer per device past the first.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import abi_contract
from max import functional as F
from max.driver import CPU, DeviceSpec
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.deepseekV41_modulev3.batch_processor import (
    DeepseekV41BatchProcessor,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
    kv_cache_config,
)
from max.pipelines.architectures.deepseekV41_modulev3.model import (
    DeepseekV41Model,
)
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
)
from max.tensor import default_device
from max.tensor.compilation import StagedGraph
from stub import (
    DeepseekV41Stub,
    make_mesh,
)


def pipeline_compile_input_types(
    model: str, n_devices: int
) -> tuple[V41Config, MultiKVCacheParams, list[TensorType | BufferType]]:
    """The inputs ``DeepseekV41Model`` compiles with at TP``n_devices``."""
    hf = abi_contract.hf_config(model)
    config = V41Config.from_hf(hf)
    device_refs = [DeviceRef.GPU(i) for i in range(n_devices)]
    kv_params = build_kv_params(config, kv_cache_config(), device_refs)

    # A batch processor over a minimal runtime: the config and the
    # pipeline config are stand-ins read only through the fields named
    # here (the same stand-in the serve-capture tests build). The one
    # device spec is what a real EP1 PipelineConfig carries, so the
    # model's layout check counts a single device and stays on the
    # EP1 path that never touches the device list.
    runtime = BatchProcessorRuntime(
        pipeline_config=cast(
            Any,
            SimpleNamespace(
                model=SimpleNamespace(
                    huggingface_config=hf,
                    data_parallel_degree=1,
                    device_specs=[DeviceSpec.accelerator(0)],
                )
            ),
        ),
        devices=[CPU()],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=1,
        max_seq_len=1,
        max_batch_input_tokens=1,
    )
    pipeline: Any = object.__new__(DeepseekV41Model)
    pipeline.pipeline_config = runtime.pipeline_config
    pipeline.pipeline_config.runtime = SimpleNamespace(ep_size=1)
    pipeline._batch_processor = DeepseekV41BatchProcessor(
        cast(Any, SimpleNamespace()), runtime
    )
    pipeline.kv_params = kv_params
    pipeline.device_refs = device_refs
    pipeline._init_distributed_runtime(config)
    return config, kv_params, list(pipeline._get_compile_input_types(config))


def stage_pipeline(
    model: str, n_devices: int
) -> tuple[StagedGraph[..., Any], V41Config, MultiKVCacheParams]:
    """Traces the stub with the pipeline's compile inputs.

    Must run inside :func:`abi_contract.virtual_devices`.
    """
    config, kv_params, input_types = pipeline_compile_input_types(
        model, n_devices
    )
    mesh = make_mesh(n_devices)
    with F.lazy(), default_device(mesh):
        stub = DeepseekV41Stub(config, kv_params, mesh)
    return stub.trace(*input_types), config, kv_params
