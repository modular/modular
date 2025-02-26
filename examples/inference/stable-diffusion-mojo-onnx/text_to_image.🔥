#!/usr/bin/env mojo
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

import sys
import random
import os
from pathlib import Path

from collections import List
from memory import memcpy
from python import Python
from random import seed

from max.engine import InferenceSession
from max.tensor import Tensor, TensorShape, TensorSpec

# Local Mojo imports
import python_utils
from scheduler import Scheduler


var GUIDANCE_SCALE_FACTOR = Float32(7.5)
var LATENT_SCALE_FACTOR = Float32(0.18215)
var OUTPUT_HEIGHT = 512
var OUTPUT_WIDTH = 512
var LATENT_WIDTH = OUTPUT_WIDTH // 8
var LATENT_HEIGHT = OUTPUT_HEIGHT // 8
var LATENT_CHANNELS = 4


def vstack[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    """Concatenate tensors a & b along the outermost dimension."""
    # Generate return shape.
    out_shape = List[Int]()
    out_shape.append(a.shape()[0] + b.shape()[0])
    for i in range(1, a.shape().rank()):
        out_shape.append(a.shape()[i])

    # Allocate return tensor.
    out = Tensor[dtype](TensorShape(out_shape))

    # Fill data.
    memcpy(out.unsafe_ptr().address, a.unsafe_ptr().address, a.num_elements())
    memcpy(
        (out.unsafe_ptr() + a.num_elements()).address,
        b.unsafe_ptr().address,
        b.num_elements(),
    )
    return out


def split[dtype: DType](x: Tensor[dtype], i: Int) -> Tensor[dtype]:
    """Return the ith slice of the outermost dim; i.e., x[idx, :, :, ..., :]."""
    # Generate return shape
    shape = List[Int]()
    shape.append(1)
    for i in range(1, x.shape().rank()):
        shape.append(x.shape()[i])
    # Allocate return tensor
    ret = Tensor[dtype](TensorShape(shape))
    # Fill data
    memcpy(
        ret.unsafe_ptr().address,
        (x.unsafe_ptr() + (x.num_elements() // 2) * i).address,
        x.num_elements() // 2,
    )
    return ret


def main():
    # Parse args.
    USAGE = (
        "Usage: ./text_to_image.🔥 --prompt <str> [--negative-prompt <str>]"
        " [--num-steps <int>] [--seed <int>] [-o <str>]"
    )

    argv = sys.argv()
    if len(argv) % 2 == 0:
        print(USAGE)
        raise Error("All options require an argument")

    # Set default values
    seed()
    prompt = String("")
    negative_prompt = String("")
    num_steps = 25
    hf = Python.import_module("huggingface_hub")
    model_dir = Path(
        String(hf.snapshot_download("modularai/stable-diffusion-1.5-onnx"))
    )
    output = String("output.png")

    for i in range(1, len(argv), 2):
        if argv[i] == "--prompt":
            prompt = String(argv[i + 1])
        elif argv[i] == "--negative-prompt":
            negative_prompt = String(argv[i + 1])
        elif argv[i] == "--num-steps":
            num_steps = atol(argv[i + 1])
        elif argv[i] == "--seed":
            seed(atol(argv[i + 1]))
        elif argv[i] == "-o" or argv[i] == "--output":
            output = String(argv[i + 1])
        else:
            print(USAGE)
            raise Error(String("Unknown option: ") + argv[i])

    # Only required arg is --prompt
    if prompt == "":
        print(USAGE)
        raise Error("--prompt option is required")

    # Import python modules.
    np = Python.import_module("numpy")
    Image = Python.import_module("PIL.Image")
    transformers = Python.import_module("transformers")

    # Compile & load models - this may take a few minutes.
    print("Loading and compiling models...")
    session = InferenceSession()
    txt_encoder = session.load(model_dir / "text_encoder" / "model.onnx")
    img_decoder = session.load(model_dir / "vae_decoder" / "model.onnx")
    img_diffuser = session.load(model_dir / "unet" / "model.onnx")
    print("Models compiled.\n")

    # Tokenize inputs and run through text encoder.
    print("Processing input...")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        os.path.join(model_dir.path, "tokenizer")
    )
    max_length = Int(tokenizer.model_max_length)
    prompt_p = tokenizer(prompt, padding="max_length", max_length=max_length)
    prompt_n = tokenizer(
        negative_prompt, padding="max_length", max_length=max_length
    )

    prompt_p_tensor = python_utils.numpy_to_tensor[DType.int32](
        np.array(prompt_p["input_ids"], dtype=np.int32)
    )
    prompt_n_tensor = python_utils.numpy_to_tensor[DType.int32](
        np.array(prompt_n["input_ids"], dtype=np.int32)
    )

    input_ids = vstack[DType.int32](
        prompt_p_tensor.reshape(TensorShape(1, max_length)),
        prompt_n_tensor.reshape(TensorShape(1, max_length)),
    )

    encoder_output = txt_encoder.execute("input_ids", input_ids)
    encoder_hidden_states = encoder_output.get[DType.float32](
        "last_hidden_state"
    )
    print("Input processed.\n")

    # Initialize latent.
    print("Initializing latent...")

    schedule = Scheduler(num_steps)

    # Note: For onnx, shapes are given in NCHW format.
    latent = (
        Tensor[DType.float32].randn(
            TensorShape(1, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)
        )
        * Scheduler.init_noise_sigma
    )

    # Loop through diffusion model.
    for i in range(len(schedule.timesteps)):
        print("\rGenerating image:", i, "/", num_steps, end="")

        # Duplicate latent to create full sample.
        sample = vstack(latent, latent)

        # Execute the diffusion model with bs=2. Both batches have same primary input and
        # timestep, but the encoder_hidden_states (primary prompt vs negative) differs.
        timestep_tensor = Tensor[DType.int64](TensorSpec(DType.int64, 1))
        timestep_tensor[0] = Int(schedule.timesteps[i])

        diffuser_output = img_diffuser.execute(
            "sample",
            sample,
            "encoder_hidden_states",
            encoder_hidden_states,
            "timestep",
            timestep_tensor,
        )

        noise_pred = diffuser_output.get[DType.float32]("out_sample")

        # Merge conditioned & unconditioned outputs.
        noise_pred_text = split[DType.float32](noise_pred, 0)
        noise_pred_uncond = split[DType.float32](noise_pred, 1)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE_FACTOR * (
            noise_pred_text - noise_pred_uncond
        )

        # Merge latent with previous iteration.
        latent = schedule.step(noise_pred, schedule.timesteps[i], latent)

    # Decode finalized latent.
    print("\n\nDecoding image...")
    latent = latent * (1 / LATENT_SCALE_FACTOR)

    decoder_output = img_decoder.execute("latent_sample", latent)
    decoded = decoder_output.get[DType.float32]("sample")

    image = ((decoded / 2.0 + 0.5).clip(0, 1) * 255.0).astype[DType.uint8]()
    np_pixels = python_utils.tensor_to_numpy[DType.uint8](image, np)
    np_pixels = np_pixels.squeeze().transpose(1, 2, 0)
    Image.fromarray(np_pixels, "RGB").save(output)
    print("Image saved to " + output + ".")
