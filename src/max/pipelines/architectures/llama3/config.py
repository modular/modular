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
"""All configurable parameters for Llama3."""

from __future__ import annotations

from max.pipelines import HuggingFaceFile, SupportedEncoding


def get_llama_huggingface_file(
    version: str, encoding: SupportedEncoding, revision: str | None = None
) -> HuggingFaceFile:
    if version == "3":
        filenames = {
            SupportedEncoding.bfloat16: "llama-3-8b-instruct-bf16.gguf",
            SupportedEncoding.float32: "llama-3-8b-f32.gguf",
            SupportedEncoding.q4_k: "llama-3-8b-instruct-q4_k_m.gguf",
            SupportedEncoding.q4_0: "llama-3-8b-instruct-q4_0.gguf",
            SupportedEncoding.q6_k: "llama-3-8b-instruct-q6_k.gguf",
        }
        filename = filenames.get(encoding)
        if filename is None:
            raise ValueError(
                f"encoding does not have default hf file: {encoding}"
            )
        return HuggingFaceFile("modularai/llama-3", filename, revision)

    elif version == "3.1":
        filenames = {
            SupportedEncoding.bfloat16: "llama-3.1-8b-instruct-bf16.gguf",
            SupportedEncoding.float32: "llama-3.1-8b-instruct-f32.gguf",
            SupportedEncoding.q4_k: "llama-3.1-8b-instruct-q4_k_m.gguf",
            SupportedEncoding.q4_0: "llama-3.1-8b-instruct-q4_0.gguf",
            SupportedEncoding.q6_k: "llama-3.1-8b-instruct-q6_k.gguf",
        }
        filename = filenames.get(encoding)
        if filename is None:
            raise ValueError(
                f"encoding does not have default hf file: {encoding}"
            )
        return HuggingFaceFile("modularai/llama-3.1", filename, revision)

    else:
        raise ValueError(f"version {version} not supported for llama")
