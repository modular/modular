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

"""Weight name adapters for the V2 Qwen3 text encoder."""

from __future__ import annotations

# Maps raw safetensor keys to the names used by
# :class:`Qwen3TextEncoderTransformer`. The V2 text encoder uses unfused
# ``q_proj``/``k_proj``/``v_proj`` Linears, so no QKV-fusion remap is
# applied — unlike Llama3's ``LLAMA_SAFETENSOR_MAPPING``.
QWEN3_TEXT_ENCODER_SAFETENSOR_MAP: dict[str, str] = {
    "model.": "",
}
