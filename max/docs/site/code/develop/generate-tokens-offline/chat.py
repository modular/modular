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
# DOC: develop/generate-tokens-offline.mdx

from max.pipelines import AutoModel, AutoTokenizer

model_id = "LiquidAI/LFM3-1.7B"
tokenizer = AutoTokenizer(model_id)
messages = [[{"role": "user", "content": "What is the capital of France?"}]]

with AutoModel(model_id, max_length=4096) as model:
    result = model.generate(
        messages=messages,
        chat_template_kwargs={"enable_thinking": False},
        max_new_tokens=32,
        top_k=1,
    )

reply = result.sequences[0][result.prompt_lengths[0] :]
print(tokenizer.decode(reply, skip_special_tokens=True))

assert len(reply) > 0
