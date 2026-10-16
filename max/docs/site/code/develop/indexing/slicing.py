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
# DOC: develop/indexing.mdx

from max.experimental.tensor import Tensor

hidden = Tensor.ones((2, 4, 8))  # (batch=2, seq_len=4, features=8)

print(hidden[0].shape)  # drop batch dim
print(hidden[:, 1:3].shape)  # window along seq_len
print(hidden[:, -1].shape)  # last token
print(hidden[..., -1].shape)  # last feature, any rank
