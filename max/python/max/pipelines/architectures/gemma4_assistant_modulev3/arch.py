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
"""Registers Gemma4Assistant on ModuleV3."""

from dataclasses import replace

from ..gemma4_assistant.arch import gemma4_assistant_arch
from .model import Gemma4AssistantModuleV3DraftArchConfig

# The same checkpoint as the graph API assistant, drafting for a ModuleV3
# Gemma 4 target.
gemma4_assistant_modulev3_arch = replace(
    gemma4_assistant_arch,
    name="Gemma4AssistantForCausalLM_ModuleV3",
    config=Gemma4AssistantModuleV3DraftArchConfig,
)
