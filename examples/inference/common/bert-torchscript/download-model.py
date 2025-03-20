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

import os

# suppress extraneous logging
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    BertForMaskedLM,
    logging,
)

HF_MODEL_NAME = "bert-base-uncased"
DEFAULT_MODEL_PATH = "../../models/bert-mlm.torchscript"


def main():
    parser = ArgumentParser(description="Download model for inference.")
    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Whether to use the Bert's Masked Language Model variant",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Location to save the model",
        default=DEFAULT_MODEL_PATH,
    )

    args = parser.parse_args()

    torch.set_default_device("cpu")

    model_path = Path(args.output_path)

    print("Downloading model...")
    logging.set_verbosity_error()  # Disable warning suggesting to train the model
    if args.mlm:
        model = BertForMaskedLM.from_pretrained(HF_MODEL_NAME)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_NAME
        )

    model.eval()
    # We set return_dict to False to return Tensors directly
    model.config.return_dict = False

    print("Saving model in TorchScript format...")
    model_path = Path(args.output_path)
    if model_path.exists():
        print(f"'{args.output_path}' already exists.\n")
    else:
        print("Converting the model to TorchScript format...")
        batch = 1
        seqlen = 128
        inputs = {
            "input_ids": torch.zeros((batch, seqlen), dtype=torch.int64),
            "attention_mask": torch.zeros((batch, seqlen), dtype=torch.int64),
            "token_type_ids": torch.zeros((batch, seqlen), dtype=torch.int64),
        }
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model, example_kwarg_inputs=dict(inputs), strict=False
            )

        torch.jit.save(traced_model, model_path)
        print("Model saved.")


if __name__ == "__main__":
    main()
