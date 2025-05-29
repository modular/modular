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

"""Pipeline Tasks."""

from enum import Enum


class PipelineTask(str, Enum):
    TEXT_GENERATION = "text_generation"
    EMBEDDINGS_GENERATION = "embeddings_generation"
    AUDIO_GENERATION = "audio_generation"
    SPEECH_TOKEN_GENERATION = "speech_token_generation"
