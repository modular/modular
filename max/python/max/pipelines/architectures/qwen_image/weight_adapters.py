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

"""Weight key remapping for QwenImage transformer.

QwenImage HuggingFace weight keys follow the same pattern as Flux2 since
both use diffusers naming conventions. The keys map directly.
"""

# The QwenImage transformer weights from HuggingFace use the same naming
# convention as the MAX implementation, so no remapping is needed.
# Weight keys like:
#   transformer_blocks.0.attn.to_q.weight
#   img_in.weight
#   txt_in.weight
#   norm_out.linear.weight
#   proj_out.weight
# map directly to our Module attribute names.
#
# The only adaptation needed is in the ComponentModel.load_model() method
# which strips component prefixes during weight loading.
