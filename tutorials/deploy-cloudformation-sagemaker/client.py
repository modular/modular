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

import json

import boto3
import numpy as np
import transformers
from botocore.config import Config

config = Config(region_name="us-east-1")
client = boto3.client("sagemaker-runtime", config=config)

# NOTE: Paste your endpoint here
endpoint_name = "YOUR-ENDPOINT-GOES-HERE"

text = "The quick brown fox jumped over the lazy dog."

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(
    text, padding="max_length", max_length=128, return_tensors="pt"
)

# Convert tensor inputs to list for payload
input_ids = inputs["input_ids"].tolist()[0]
attention_mask = inputs["attention_mask"].tolist()[0]
token_type_ids = inputs["token_type_ids"].tolist()[0]

payload = {
    "inputs": [
        {
            "name": "input_ids",
            "shape": [1, 128],
            "datatype": "INT32",
            "data": input_ids,
        },
        {
            "name": "attention_mask",
            "shape": [1, 128],
            "datatype": "INT32",
            "data": attention_mask,
        },
        {
            "name": "token_type_ids",
            "shape": [1, 128],
            "datatype": "INT32",
            "data": token_type_ids,
        },
    ]
}

http_response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/octet-stream",
    Body=json.dumps(payload),
)
response = json.loads(http_response["Body"].read().decode("utf8"))
outputs = response["outputs"]


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


# Process the output
for output in outputs:
    logits = output["data"]
    logits = np.array(logits).reshape(output["shape"])

    print(f"Logits shape: {logits.shape}")

    if (
        len(logits.shape) == 3
    ):  # Shape [batch_size, sequence_length, num_classes]
        token_probabilities = softmax(logits)
        predicted_classes = np.argmax(token_probabilities, axis=-1)

        print(f"Predicted classes shape: {predicted_classes.shape}")
        print(
            f"Predicted class indices range: {np.min(predicted_classes)},"
            f" {np.max(predicted_classes)}"
        )

        # Map predicted indices to tokens
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_classes[0])

        # Pair each input token with its predicted token
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        token_pairs = list(zip(input_tokens, predicted_tokens))

        print("Predicted Token Pairs:")
        print("-" * 45)
        print("| {:<20} | {:<18} |".format("Input Token", "Predicted Token"))
        print("-" * 45)
        for input_token, predicted_token in token_pairs:
            if input_token != "[PAD]":  # Exclude padding tokens
                print(f"| {input_token:<20} | {predicted_token:<18} |")
        print("-" * 45)
