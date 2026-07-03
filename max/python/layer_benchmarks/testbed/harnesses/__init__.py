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

"""Built-in harness implementations. Importing this module registers all
harnesses with the global registry."""

import testbed.harnesses.attention_with_rope
import testbed.harnesses.gemma3_attention
import testbed.harnesses.gemma4_attention
import testbed.harnesses.gpt_oss_attention
import testbed.harnesses.olmo2_attention
import testbed.harnesses.qwen2_5vl_attention
import testbed.harnesses.qwen3_attention
import testbed.harnesses.rms_norm
import testbed.harnesses.text_encoder
