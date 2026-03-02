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
"""Activation function modules for MAX neural networks.

This module is a compatibility shim. Use max.nn.module_v3 directly.
"""

from .module_v3.activations import (  # noqa: F401
    GEGLU,
    GELU,
    ApproximateGELU,
    FeedForward,
    Identity,
    SiLU,
    SwiGLU,
)
