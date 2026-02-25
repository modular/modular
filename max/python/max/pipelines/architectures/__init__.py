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

from max.pipelines.lib.registry import PIPELINE_REGISTRY

_MODELS_ALREADY_REGISTERED = False


def register_all_models() -> None:
    """Register all built-in model architectures with the global registry.

    This function imports each supported model architecture module (Llama, Mistral,
    Qwen, Gemma, DeepSeek, etc.) and registers their :class:`~max.pipelines.SupportedArchitecture`
    definitions with :obj:`~max.pipelines.PIPELINE_REGISTRY`.

    This function is called automatically when :mod:`max.pipelines` is imported,
    so you typically don't need to call it manually. It uses an internal flag to
    ensure architectures are only registered once, making repeated calls safe but
    unnecessary.
    """
    global _MODELS_ALREADY_REGISTERED

    if _MODELS_ALREADY_REGISTERED:
        return

    from .bert import bert_arch
    from .deepseekV2 import deepseekV2_arch
    from .deepseekV3 import deepseekV3_arch
    from .deepseekV3_2 import deepseekV3_2_arch
    from .deepseekV3_nextn import deepseekV3_nextn_arch
    from .eagle_llama3 import eagle_llama_arch
    from .exaone import exaone_arch
    from .exaone_modulev3 import exaone_modulev3_arch
    from .flux1 import flux1_arch
    from .flux2 import flux2_arch
    from .gemma3 import gemma3_arch
    from .gemma3multimodal import gemma3_multimodal_arch
    from .gpt_oss import gpt_oss_arch
    from .gpt_oss_modulev3 import gpt_oss_modulev3_arch
    from .granite import granite_arch
    from .granite_modulev3 import granite_modulev3_arch
    from .idefics3 import idefics3_arch
    from .internvl import internvl_arch
    from .llama3 import llama_arch
    from .llama3_modulev3 import llama_modulev3_arch
    from .llama4 import llama4_arch
    from .mistral import mistral_arch
    from .mistral3 import mistral3_arch
    from .mpnet import mpnet_arch
    from .olmo import olmo_arch
    from .olmo2 import olmo2_arch
    from .olmo3 import olmo3_arch
    from .phi3 import phi3_arch
    from .phi3_modulev3 import phi3_modulev3_arch
    from .pixtral import pixtral_arch
    from .qwen2 import qwen2_arch
    from .qwen2_5vl import qwen2_5_vl_arch
    from .qwen3 import qwen3_arch, qwen3_moe_arch
    from .qwen3_embedding import qwen3_embedding_arch
    from .qwen3vl_moe import qwen3vl_arch, qwen3vl_moe_arch

    architectures = [
        exaone_arch,
        exaone_modulev3_arch,
        deepseekV2_arch,
        deepseekV3_arch,
        deepseekV3_2_arch,
        deepseekV3_nextn_arch,
        eagle_llama_arch,
        flux1_arch,
        flux2_arch,
        gemma3_arch,
        gemma3_multimodal_arch,
        granite_arch,
        granite_modulev3_arch,
        gpt_oss_arch,
        gpt_oss_modulev3_arch,
        internvl_arch,
        idefics3_arch,
        llama_arch,
        llama_modulev3_arch,
        llama4_arch,
        bert_arch,
        mistral_arch,
        mistral3_arch,
        mpnet_arch,
        olmo_arch,
        olmo2_arch,
        olmo3_arch,
        phi3_arch,
        phi3_modulev3_arch,
        pixtral_arch,
        qwen2_arch,
        qwen2_5_vl_arch,
        qwen3_arch,
        qwen3_moe_arch,
        qwen3_embedding_arch,
        qwen3vl_arch,
        qwen3vl_moe_arch,
    ]

    for arch in architectures:
        PIPELINE_REGISTRY.register(arch)

    _MODELS_ALREADY_REGISTERED = True


__all__ = ["register_all_models"]
