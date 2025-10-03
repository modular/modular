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

# suppress extraneous logging
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

text = "Paris is the [MASK] of France."

# Create a triton client
triton_client = httpclient.InferenceServerClient(url="127.0.0.1:8000")

# Preprocess input statement
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(
    text,
    return_tensors="np",
    return_token_type_ids=True,
    padding="max_length",
    truncation=True,
    max_length=128,
)

# Set the input data
triton_inputs = [
    httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT32"),
    httpclient.InferInput(
        "attention_mask", inputs["attention_mask"].shape, "INT32"
    ),
    httpclient.InferInput(
        "token_type_ids", inputs["token_type_ids"].shape, "INT32"
    ),
]
triton_inputs[0].set_data_from_numpy(inputs["input_ids"].astype(np.int32))
triton_inputs[1].set_data_from_numpy(inputs["attention_mask"].astype(np.int32))
triton_inputs[2].set_data_from_numpy(inputs["token_type_ids"].astype(np.int32))

# Executing
output = triton_client.infer("bert-base-uncased", triton_inputs)

# Post-processing
masked_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()[1]
logits = output.as_numpy("result0")[0, masked_index, :]
predicted_token_ids = logits.argmax(axis=-1)
predicted_text = tokenizer.decode(predicted_token_ids)
output_text = text.replace("[MASK]", predicted_text)
print(output_text)
