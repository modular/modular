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

import argparse
from pathlib import Path

import numpy as np
from max.entrypoints.diffusion import PixelGenerator
from max.pipelines.lib import PixelGenerationConfig
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="black-forest-labs/FLUX.1-dev"
    )
    parser.add_argument("--output-path", type=str, default="output.png")
    args = parser.parse_args()

    model_path = args.model_path
    pipeline_config = PixelGenerationConfig(model_path=model_path)
    pipe = PixelGenerator(pipeline_config)

    prompt = "A cat holding a sign that says hello world"

    result = pipe.generate(
        prompts=prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        model_name=model_path,
    )

    image: np.ndarray = result.outputs[0].pixel_data[0]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(output_path)

    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
