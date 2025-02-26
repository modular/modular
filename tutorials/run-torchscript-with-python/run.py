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

from pathlib import Path

import torch
from max import engine
from max.dtype import DType
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The HuggingFace model name and TorchScript file name
HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
MODEL_PATH = Path("roberta.torchscript")


def main():
    batch = 1
    seqlen = 128
    input_spec = {
        "input_ids": torch.zeros((batch, seqlen), dtype=torch.int64),
        "attention_mask": torch.zeros((batch, seqlen), dtype=torch.int64),
    }

    # We use the same `input_spec` that we used to trace the model (download-model.py)
    input_spec_list = [
        engine.TorchInputSpec(shape=tensor.size(), dtype=DType.int64)
        for tensor in input_spec.values()
    ]

    session = engine.InferenceSession()
    model = session.load(MODEL_PATH, input_specs=input_spec_list)

    for tensor in model.input_metadata:
        print(
            f"name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}"
        )

    # The model input
    text_input = (
        "There are many exciting developments in the field of AI"
        " Infrastructure!"
    )

    # Tokenize the input
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seqlen,
    )
    print(inputs)

    outputs = model.execute_legacy(**inputs)

    # Extract class prediction from output
    hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    predicted_class_id = outputs["result0"]["logits"].argmax(axis=-1)[0]
    classification = hf_model.config.id2label[predicted_class_id]

    print(f"The sentiment is: {classification}")


if __name__ == "__main__":
    main()
