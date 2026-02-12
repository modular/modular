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
"""Imports the respective loader if its dependency is available, or adds a stub if not."""

try:
    import gguf  # type: ignore # noqa: F401

    from .load_gguf import GGUFWeights
except ImportError:

    class GGUFWeights:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Unable to load gguf file, gguf not installed")
