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

from functools import partial

from max import functional as F

ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": partial(F.gelu, approximate="tanh"),
    "gelu_quick": partial(F.gelu, approximate="quick"),
    "quick_gelu": partial(F.gelu, approximate="quick"),
    "relu": F.relu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
}
