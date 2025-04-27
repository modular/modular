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

#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from transformers import BertTokenizer

HF_MODEL_NAME = "bert-base-uncased"


def main():
    parser = ArgumentParser(description="Preprocessing for BERT inputs")
    parser.add_argument(
        "--text",
        type=str,
        metavar="<text>",
        required=True,
        help="Statement to classify.",
    )

    args = parser.parse_args()

    print("Generating input tensors...")
    print(f'Input sentence: "{args.text}".')

    tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
    encoded_inputs = tokenizer(args.text, return_tensors="pt")

    print("Saving inputs to disk...")
    input_dir = Path("inputs")
    input_dir.mkdir(exist_ok=True)

    created_files = []
    for name, value in encoded_inputs.items():
        value = value.numpy().astype(np.int32)
        filename = input_dir / name
        filename = filename.with_suffix(".bin")
        filename.unlink(missing_ok=True)
        value.tofile(filename)

        shape = np.array(value.shape).astype(np.int64)
        shape_file = input_dir / f"{name}_shape.bin"
        shape.tofile(shape_file)
        created_files += [str(filename), str(shape_file)]
    print("Inputs saved.")


if __name__ == "__main__":
    main()
