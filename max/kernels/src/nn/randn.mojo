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

from random import randn

from layout import Layout, LayoutTensor


fn random_normal[
    type: DType,
    mean: Float64,
    variance: Float64,
](output: LayoutTensor[mut=True, type, *_, **_]):
    """
    Fill `output` with values generated from Normal(mean, variance) distribution.

    Args:
        output: The output buffer.
    """
    randn(output.ptr, output.size(), mean, variance)
