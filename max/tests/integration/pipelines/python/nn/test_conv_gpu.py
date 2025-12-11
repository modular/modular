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
from max.engine.api import InferenceSession
from shared_conv_impl import conv1d_impl, conv2d_impl, conv3d_impl
from test_common.numerics import pytorch_disable_tf32_dtype


@pytorch_disable_tf32_dtype
def test_conv3d_gpu(gpu_session: InferenceSession) -> None:
    conv3d_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv1d_gpu(gpu_session: InferenceSession) -> None:
    conv1d_impl(gpu_session)


@pytorch_disable_tf32_dtype
def test_conv2d_gpu(gpu_session: InferenceSession) -> None:
    conv2d_impl(gpu_session)
