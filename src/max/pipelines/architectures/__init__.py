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

from max.pipelines import PIPELINE_REGISTRY


def register_all_models():
    """Imports model architectures, thus registering the architecture in the shared PIPELINE_REGISTRY."""
    from .exaone import exaone_arch
    from .granite import granite_arch
    from .llama3 import llama_arch
    from .llama_vision import llama_vision_arch
    from .mistral import mistral_arch
    from .mpnet import mpnet_arch
    from .olmo import olmo_arch
    from .phi3 import phi3_arch
    from .pixtral import pixtral_arch
    from .qwen2 import qwen2_arch
    from .replit import replit_arch

    architectures = [
        exaone_arch,
        llama_arch,
        llama_vision_arch,
        mistral_arch,
        mpnet_arch,
        olmo_arch,
        phi3_arch,
        pixtral_arch,
        qwen2_arch,
        replit_arch,
        granite_arch,
    ]

    for arch in architectures:
        PIPELINE_REGISTRY.register(arch)


__all__ = ["register_all_models"]
