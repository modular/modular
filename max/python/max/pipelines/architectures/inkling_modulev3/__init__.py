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

"""Inkling architecture on ModuleV3: a text-and-vision pipeline model with KV
cache and short-convolution state.
"""

from .arch import inkling_modulev3_arch
from .model import InklingModel

__all__ = ["InklingModel", "inkling_modulev3_arch"]
