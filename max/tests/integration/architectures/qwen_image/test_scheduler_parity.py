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

"""Test that MAX scheduler sigma schedule matches diffusers logic exactly.

Computes the reference sigma schedule using the same math as diffusers
(without loading the full 20B model) and compares against MAX scheduler.
"""

import numpy as np
from max.pipelines.lib.diffusion_schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

PATCH_SIZE = 2
VAE_SCALE_FACTOR = 8

# QwenImage scheduler_config.json values
QWEN_BASE_IMAGE_SEQ_LEN = 256
QWEN_MAX_IMAGE_SEQ_LEN = 8192
QWEN_BASE_SHIFT = 0.5
QWEN_MAX_SHIFT = 0.9
QWEN_USE_DYNAMIC_SHIFTING = True
QWEN_SHIFT_TERMINAL = 0.02


def _compute_reference_sigmas(
    height: int,
    width: int,
    num_inference_steps: int,
) -> np.ndarray:
    """Compute sigma schedule using the same math as diffusers.

    This replicates FlowMatchEulerDiscreteScheduler.set_timesteps() from
    diffusers, including dynamic shifting and stretch_shift_to_terminal.
    """
    latent_h = height // VAE_SCALE_FACTOR
    latent_w = width // VAE_SCALE_FACTOR
    image_seq_len = (latent_h // PATCH_SIZE) * (latent_w // PATCH_SIZE)

    # Base sigmas: linearly spaced from 1.0 to 1/num_train_timesteps
    # (matches diffusers FlowMatchEulerDiscreteScheduler which uses
    # sigma_min = 1/num_train_timesteps, not 1/num_inference_steps)
    num_train_timesteps = 1000
    sigmas = np.linspace(1.0, 1.0 / num_train_timesteps, num_inference_steps)

    # Dynamic shifting: compute mu from linear interpolation
    slope = (QWEN_MAX_SHIFT - QWEN_BASE_SHIFT) / (
        QWEN_MAX_IMAGE_SEQ_LEN - QWEN_BASE_IMAGE_SEQ_LEN
    )
    mu = slope * image_seq_len + (
        QWEN_BASE_SHIFT - slope * QWEN_BASE_IMAGE_SEQ_LEN
    )

    # Exponential time shift: sigma(t) = exp(mu) / (exp(mu) + (1/t - 1))
    t_safe = np.clip(sigmas.astype(np.float64), 1e-7, 1.0)
    sigmas = (np.exp(mu) / (np.exp(mu) + (1.0 / t_safe - 1.0))).astype(
        np.float32
    )

    # Terminal stretching: stretch so last sigma = shift_terminal
    shift_terminal = QWEN_SHIFT_TERMINAL
    if shift_terminal is not None and shift_terminal > 0:
        one_minus_z = 1.0 - sigmas
        scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
        sigmas = (1.0 - (one_minus_z / scale_factor)).astype(np.float32)

    sigmas = np.append(sigmas, np.float32(0.0))
    return sigmas


def _get_max_sigmas(
    height: int,
    width: int,
    num_inference_steps: int,
) -> np.ndarray:
    """Get sigma schedule from MAX scheduler."""
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=QWEN_BASE_IMAGE_SEQ_LEN,
        max_image_seq_len=QWEN_MAX_IMAGE_SEQ_LEN,
        base_shift=QWEN_BASE_SHIFT,
        max_shift=QWEN_MAX_SHIFT,
        use_dynamic_shifting=QWEN_USE_DYNAMIC_SHIFTING,
        shift_terminal=QWEN_SHIFT_TERMINAL,
    )
    image_seq_len = (height // (VAE_SCALE_FACTOR * PATCH_SIZE)) * (
        width // (VAE_SCALE_FACTOR * PATCH_SIZE)
    )
    _, sigmas = scheduler.retrieve_timesteps_and_sigmas(
        image_seq_len=image_seq_len,
        num_inference_steps=num_inference_steps,
    )
    return sigmas


def test_sigma_schedule_matches_reference() -> None:
    """Verify MAX sigma schedule matches reference diffusers math (fp32)."""
    height, width, steps = 1024, 1024, 20

    ref_sigmas = _compute_reference_sigmas(height, width, steps)
    max_sigmas = _get_max_sigmas(height, width, steps)

    assert ref_sigmas.shape == max_sigmas.shape, (
        f"Shape mismatch: ref={ref_sigmas.shape} vs MAX={max_sigmas.shape}"
    )
    np.testing.assert_allclose(
        max_sigmas,
        ref_sigmas,
        atol=1e-6,
        rtol=1e-5,
        err_msg="Sigma schedules differ between MAX and reference",
    )


def test_sigma_schedule_50_steps() -> None:
    """Test sigma schedule with 50 steps at various resolutions."""
    for height, width in [(512, 512), (1024, 1024), (768, 1024)]:
        ref = _compute_reference_sigmas(height, width, 50)
        max_ = _get_max_sigmas(height, width, 50)
        np.testing.assert_allclose(
            max_,
            ref,
            atol=1e-6,
            rtol=1e-5,
            err_msg=f"Mismatch at {height}x{width}, 50 steps",
        )


def test_shift_terminal_effect() -> None:
    """Verify shift_terminal stretches the last sigma correctly."""
    shift_terminal = 0.02
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256,
        max_image_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
        use_dynamic_shifting=True,
        shift_terminal=shift_terminal,
    )
    _, sigmas = scheduler.retrieve_timesteps_and_sigmas(
        image_seq_len=4096,
        num_inference_steps=50,
    )
    # Last non-zero sigma should equal shift_terminal
    assert abs(sigmas[-2] - shift_terminal) < 1e-5, (
        f"Last non-zero sigma {sigmas[-2]} != shift_terminal {shift_terminal}"
    )
    assert sigmas[-1] == 0.0


def test_no_shift_terminal_preserves_behavior() -> None:
    """Without shift_terminal, scheduler should behave as before."""
    scheduler_with = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256,
        max_image_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
        use_dynamic_shifting=True,
        shift_terminal=None,
    )
    scheduler_without = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256,
        max_image_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
        use_dynamic_shifting=True,
    )
    _, sigmas_with = scheduler_with.retrieve_timesteps_and_sigmas(
        image_seq_len=4096, num_inference_steps=50
    )
    _, sigmas_without = scheduler_without.retrieve_timesteps_and_sigmas(
        image_seq_len=4096, num_inference_steps=50
    )
    np.testing.assert_array_equal(sigmas_with, sigmas_without)
