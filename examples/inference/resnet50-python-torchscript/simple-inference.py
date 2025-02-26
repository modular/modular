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
import platform
import signal
from argparse import ArgumentParser

import numpy as np
from max import engine
from max.dtype import DType
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL_PATH = "../../models/resnet50.torchscript"
DESCRIPTION = "Classify an input image."
HF_MODEL_NAME = "microsoft/resnet-50"


def execute(model_path, inputs):
    session = engine.InferenceSession()
    input_spec_list = [
        engine.TorchInputSpec(shape=(1, 3, 224, 224), dtype=DType.float32)
    ]

    print("Loading and compiling model...")
    model = session.load(model_path, input_specs=input_spec_list)
    print("Model compiled.\n")

    print("Executing model...")
    outputs = model.execute_legacy(pixel_values=inputs["pixel_values"])
    print("Model executed.\n")
    return outputs


def main():
    # Parse args
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--input",
        type=str,
        metavar="<jpg>",
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Location of the downloaded model.",
    )
    args = parser.parse_args()

    # Improves model compilation speed dramatically on intel CPUs
    if "Intel" in platform.processor():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Preprocess input image
    print("Processing input...")
    image = Image.open(args.input)
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(images=image, return_tensors="np")
    print("Input processed.\n")

    # Classify input image
    outputs = execute(args.model_path, inputs)

    # Extract class predictions from output
    print("Extracting class from outputs...")
    predicted_label = np.argmax(outputs["result0"], axis=-1)[0]
    model = AutoModelForImageClassification.from_pretrained(HF_MODEL_NAME)
    predicted_class = model.config.id2label[predicted_label]

    print(
        "\nThe input image is likely one of the following classes:"
        f" \n{predicted_class}"
    )


if __name__ == "__main__":
    main()
