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

from .arch import llama_arch
from .distributed_llama import DistributedLlama3
from .llama3 import Llama3
from .model_config import Llama3Config
from .pipeline_parallel_llama3 import PipelineParallelLlama3


def create_llama3_model(config: Llama3Config):
    """Factory function to create the appropriate Llama3 model based on parallelism configuration.

    Args:
        config: Llama3Config with parallelism settings

    Returns:
        The appropriate model instance:
        - PipelineParallelLlama3 if pipeline_parallel_degree > 1
        - DistributedLlama3 if tensor_parallel_degree > 1
        - Llama3 for single-GPU case
    """
    if (
        config.pipeline_parallel_degree > 1
        and config.tensor_parallel_degree > 1
    ):
        raise ValueError(
            "Hybrid pipeline + tensor parallelism is not currently supported. "
            f"Got pipeline_parallel_degree={config.pipeline_parallel_degree}, "
            f"tensor_parallel_degree={config.tensor_parallel_degree}"
        )

    if config.pipeline_parallel_degree > 1:
        # Pipeline parallel model
        return PipelineParallelLlama3(config)
    elif config.tensor_parallel_degree > 1:
        # Tensor parallel model
        return DistributedLlama3(config)
    else:
        # Single-GPU model
        return Llama3(config)


__all__ = ["create_llama3_model", "llama_arch"]
