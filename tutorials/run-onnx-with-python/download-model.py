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

import torch
from torch.onnx import export
from transformers import ResNetForImageClassification

# The HuggingFace model name and exported file name
HF_MODEL_NAME = "microsoft/resnet-50"
MODEL_PATH = "resnet50.onnx"


def main():
    # Load the ResNet model from HuggingFace in evaluation mode
    model = ResNetForImageClassification.from_pretrained(HF_MODEL_NAME)
    model.eval()

    # Create random input for tracing, then export the model to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    export(
        model,
        dummy_input,
        MODEL_PATH,
        opset_version=11,
        input_names=["pixel_values"],
        output_names=["output"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Model saved as {MODEL_PATH}")


if __name__ == "__main__":
    main()
