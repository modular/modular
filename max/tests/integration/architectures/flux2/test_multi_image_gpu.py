# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Asks each FLUX.2 executor for more than one image per request on a GPU."""

from __future__ import annotations

from typing import Literal

import huggingface_hub
import numpy as np
import numpy.typing as npt
import pytest
from max.driver import accelerator_api
from max.pipelines import AutoModel
from PIL import Image

KLEIN = "black-forest-labs/FLUX.2-klein-4B"
DEV = "black-forest-labs/FLUX.2-dev"
PROMPT = "a red fox in snow"
EDIT_PROMPT = "the same fox at night"
SIZE = 512
# The distilled Klein checkpoint runs exactly 4 steps. The others take any
# count, and a few keep the FLUX.2-dev ModuleV3 run short.
STEPS = 4
SEED = 0
# Batch size can change kernel rounding, so image i of an N-image request
# matches image i of a smaller request closely rather than exactly. Observed
# mean pixel differences on B200 and MI355 were 0.07 to 2.5 for the same
# image and 36 to 70 between different images.
SAME_IMAGE_MAX_MEAN_DIFF = 4.0
OTHER_IMAGE_MIN_MEAN_DIFF = 10.0

# The logit verification config runs FLUX.2 Klein on NVIDIA only.
nvidia_only = pytest.mark.skipif(
    accelerator_api() == "hip", reason="FLUX.2 Klein is verified on NVIDIA only"
)


def _load(
    model_path: str,
    prefer_module_v3: bool = False,
    cache: Literal["taylorseer", "first_block"] | None = None,
) -> AutoModel:
    if huggingface_hub.constants.HF_HUB_OFFLINE:
        # Not every GPU runner's Hugging Face cache holds every model yet.
        try:
            huggingface_hub.snapshot_download(model_path, local_files_only=True)
        except huggingface_hub.errors.LocalEntryNotFoundError:
            pytest.skip(f"{model_path} is not in this runner's cache.")
    if cache == "taylorseer":
        # Steps 0 and 1 run the transformer, step 2 skips it and step 3 runs
        # it again.
        return AutoModel(
            model_path,
            taylorseer=True,
            taylorseer_warmup_steps=2,
            taylorseer_cache_interval=2,
        )
    if cache == "first_block":
        return AutoModel(model_path, first_block_caching=True)
    return AutoModel(model_path, prefer_module_v3=prefer_module_v3)


def _generate(
    model: AutoModel,
    prompt: str,
    num_images: int,
    input_image: Image.Image | None = None,
) -> list[npt.NDArray[np.uint8]]:
    result = model.generate(
        prompt=[prompt],
        images=None if input_image is None else [[input_image]],
        height=SIZE,
        width=SIZE,
        num_inference_steps=STEPS,
        num_images_per_prompt=num_images,
        seed=SEED,
    )
    [row] = result.images
    assert len(row) == num_images
    for image in row:
        assert image.size == (SIZE, SIZE)
        assert image.mode == "RGB"
    return [np.asarray(image) for image in row]


def _mean_diff(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _check_two_images(
    model: AutoModel,
    prompt: str,
    input_image: Image.Image | None = None,
) -> list[npt.NDArray[np.uint8]]:
    """Checks a 2-image request against a 1-image one and returns both."""
    two = _generate(model, prompt, 2, input_image)
    [single] = _generate(model, prompt, 1, input_image)
    same = _mean_diff(two[0], single)
    other = _mean_diff(two[0], two[1])
    print(f"{prompt!r}: image 0 vs 1-image {same:.3f}, image 1 {other:.3f}")
    # numpy's RandomState draws the noise in order, so image 0 starts from
    # the 1-image request's noise.
    assert same <= SAME_IMAGE_MAX_MEAN_DIFF
    assert other >= OTHER_IMAGE_MIN_MEAN_DIFF
    return two


@nvidia_only
def test_klein_generates_two_images() -> None:
    with _load(KLEIN) as model:
        two = _check_two_images(model, PROMPT)
        # Image 1 starts from the same noise in a 3-image request.
        three = _generate(model, PROMPT, 3)
        same = _mean_diff(three[1], two[1])
        print(f"image 1 of 3 vs image 1 of 2 {same:.3f}")
        assert same <= SAME_IMAGE_MAX_MEAN_DIFF
        _check_two_images(
            model, EDIT_PROMPT, input_image=Image.fromarray(two[0])
        )


@nvidia_only
def test_klein_taylorseer_generates_two_images() -> None:
    with _load(KLEIN, cache="taylorseer") as model:
        _check_two_images(model, PROMPT)


def test_dev_generates_two_images() -> None:
    with _load(DEV) as model:
        [source, _] = _check_two_images(model, PROMPT)
        _check_two_images(
            model, EDIT_PROMPT, input_image=Image.fromarray(source)
        )


def test_dev_taylorseer_generates_two_images() -> None:
    with _load(DEV, cache="taylorseer") as model:
        _check_two_images(model, PROMPT)


def test_dev_first_block_cache_generates_two_images() -> None:
    with _load(DEV, cache="first_block") as model:
        _check_two_images(model, PROMPT)


def test_dev_module_v3_generates_two_images() -> None:
    with _load(DEV, prefer_module_v3=True) as model:
        _check_two_images(model, PROMPT)
