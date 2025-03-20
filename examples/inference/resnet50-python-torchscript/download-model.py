#!/usr/bin/env python3
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
from argparse import ArgumentParser

# suppress extraneous logging
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"

from pathlib import Path

import torch
from transformers import ResNetForImageClassification

DEFAULT_MODEL_PATH = "../../models/resnet50.torchscript"
DESCRIPTION = "Download a ResNet-50 model."
HF_MODEL_NAME = "microsoft/resnet-50"


def main():
    # Parse args
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Location to save the model.",
    )
    args = parser.parse_args()

    torch.set_default_device("cpu")

    print("Downloading model...")
    model = ResNetForImageClassification.from_pretrained(HF_MODEL_NAME)
    model.eval()
    # We set return_dict to False to return Tensors directly
    model.config.return_dict = False

    print("Saving model in TorchScript format...")
    model_path = Path(args.output_path)
    if model_path.exists():
        print(f"'{args.output_path}' already exists.\n")
    else:
        print("Converting the model to TorchScript format...")
        input_batch = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, input_batch)

        torch.jit.save(traced_model, model_path)
        print(f"Model saved to {args.output_path}.\n")


if __name__ == "__main__":
    main()
