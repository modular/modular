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

from typing import Callable as _Callable
from typing import Union as _Union

from max.interfaces import TokenGenerator

from .context import TextAndVisionContext, TextContext, TTSContext
from .interfaces import (
    AudioGenerator,
    AudioGeneratorContext,
    EmbeddingsGenerator,
)
from .serialization import msgpack_numpy_decoder, msgpack_numpy_encoder

PipelinesFactory = _Callable[
    [], _Union[TokenGenerator, EmbeddingsGenerator, AudioGenerator]
]

__all__ = [
    "AudioGenerator",
    "AudioGeneratorContext",
    "EmbeddingsGenerator",
    "PipelinesFactory",
    "TTSContext",
    "TTSContext",
    "TTSContext",
    "TextAndVisionContext",
    "TextContext",
    "TokenGeneratorContext",
    "msgpack_numpy_decoder",
    "msgpack_numpy_encoder",
]
