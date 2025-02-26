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

import numpy as np
from datasets import load_dataset
from max import engine
from transformers import AutoImageProcessor, AutoModelForImageClassification

# The HuggingFace model name and exported file name
HF_MODEL_NAME = "microsoft/resnet-50"
MODEL_PATH = "resnet50.onnx"


def main():
    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]
    # optionally, save the image to see it yourself:
    # image.save("cat.png")

    image_processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = image_processor(image, return_tensors="np")

    print("Keys:", inputs.keys())
    print("Shape:", inputs["pixel_values"].shape)

    session = engine.InferenceSession()
    model = session.load(MODEL_PATH)
    outputs = model.execute_legacy(**inputs)

    print("Output shape:", outputs["output"].shape)

    predicted_label = np.argmax(outputs["output"], axis=-1)[0]
    hf_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_NAME)
    predicted_class = hf_model.config.id2label[predicted_label]

    print(f"Prediction: {predicted_class}")


if __name__ == "__main__":
    main()
