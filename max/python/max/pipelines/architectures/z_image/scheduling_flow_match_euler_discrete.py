# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025 Stability AI, Katherine Crowson, The HuggingFace Team, and Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from dataclasses import dataclass

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.tensor import Tensor

# Note: scipy.stats.beta.ppf is used for beta sigmas but requires scipy
# For now we keep this optional - use_beta_sigmas defaults to False
try:
    import scipy.stats

    _scipy_available = True
except ImportError:
    _scipy_available = False

logger = logging.getLogger("max.pipelines")


def linspace(
    start: int | float,
    stop: int | float,
    num: int,
    dtype: DType = DType.float32,
) -> Tensor:
    """Write from scratch via max"""
    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")
    div = (num - 1) if num > 1 else 1
    delta = stop - start

    step = delta / div
    y = Tensor.arange(0, num, dtype=dtype) * step + start
    if num > 1:
        y = F.concat(
            (y[:-1], Tensor.constant([stop], dtype=dtype, device=y.device))
        )
    return y


@dataclass(frozen=True)
class FlowMatchEulerDiscreteSchedulerOutput:
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: Tensor


class FlowMatchEulerDiscreteScheduler:
    """
    Euler scheduler.

    Native Modular implementation (ported from diffusers).

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to apply timestep shifting on-the-fly based on the image resolution.
        base_shift (`float`, defaults to 0.5):
            Value to stabilize image generation. Increasing `base_shift` reduces variation and image is more consistent
            with desired output.
        max_shift (`float`, defaults to 1.15):
            Value change allowed to latent vectors. Increasing `max_shift` encourages more variation and image may be
            more exaggerated or stylized.
        base_image_seq_len (`int`, defaults to 256):
            The base image sequence length.
        max_image_seq_len (`int`, defaults to 4096):
            The maximum image sequence length.
        invert_sigmas (`bool`, defaults to False):
            Whether to invert the sigmas.
        shift_terminal (`float`, defaults to None):
            The end value of the shifted timestep schedule.
        use_karras_sigmas (`bool`, defaults to False):
            Whether to use Karras sigmas for step sizes in the noise schedule during sampling.
        use_exponential_sigmas (`bool`, defaults to False):
            Whether to use exponential sigmas for step sizes in the noise schedule during sampling.
        use_beta_sigmas (`bool`, defaults to False):
            Whether to use beta sigmas for step sizes in the noise schedule during sampling.
        time_shift_type (`str`, defaults to "exponential"):
            The type of dynamic resolution-dependent timestep shifting to apply. Either "exponential" or "linear".
        stochastic_sampling (`bool`, defaults to False):
            Whether to use stochastic sampling.
    """

    _compatibles = []
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float | None = 0.5,
        max_shift: float | None = 1.15,
        base_image_seq_len: int | None = 256,
        max_image_seq_len: int | None = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool | None = False,
        use_exponential_sigmas: bool | None = False,
        use_beta_sigmas: bool | None = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
    ):
        # Store all config parameters as instance attributes
        self.num_train_timesteps = num_train_timesteps
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.invert_sigmas = invert_sigmas
        self.shift_terminal = shift_terminal
        self.use_karras_sigmas = use_karras_sigmas
        self.use_exponential_sigmas = use_exponential_sigmas
        self.use_beta_sigmas = use_beta_sigmas
        self.time_shift_type = time_shift_type
        self.stochastic_sampling = stochastic_sampling

        # Validations
        if use_beta_sigmas and not _scipy_available:
            raise ImportError(
                "Make sure to install scipy if you want to use beta sigmas."
            )
        if (
            sum([use_beta_sigmas, use_exponential_sigmas, use_karras_sigmas])
            > 1
        ):
            raise ValueError(
                "Only one of `use_beta_sigmas`, `use_exponential_sigmas`, `use_karras_sigmas` can be used."
            )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError(
                "`time_shift_type` must either be 'exponential' or 'linear'."
            )

        timesteps = linspace(
            1, num_train_timesteps, num_train_timesteps, DType.float32
        )
        # Reverse the tensor ([::-1] is not supported in Modular, use gather with reversed indices)
        n = int(timesteps.shape[0])
        reversed_indices = Tensor.arange(n - 1, -1, -1, dtype=DType.int64)
        timesteps = F.gather(timesteps, reversed_indices, axis=0)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self._shift = shift

        self.sigmas = sigmas.to(
            CPU()
        )  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def shift(self) -> float:
        """
        The value used for shifting.
        """
        return self._shift

    @property
    def step_index(self) -> int:
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self) -> int:
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`, defaults to `0`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        self._shift = shift

    def scale_noise(
        self,
        sample: Tensor,
        timestep: float | Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """
        Forward process in flow-matching

        Args:
            sample (`Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `Tensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(sample.device).cast(sample.dtype)

        schedule_timesteps = self.timesteps.to(sample.device)
        timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in timestep
            ]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma: Tensor) -> Tensor:
        return sigma * self.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: Tensor) -> Tensor:
        if self.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: Tensor) -> Tensor:
        r"""
        Stretches and shifts the timestep schedule to ensure it terminates at the configured `shift_terminal` config
        value.

        Reference:
        https://github.com/Lightricks/LTX-Video/blob/a01a171f8fe3d99dce2728d60a73fecf4d4238ae/ltx_video/schedulers/rf.py#L51

        Args:
            t (`Tensor`):
                A tensor of timesteps to be stretched and shifted.

        Returns:
            `Tensor`:
                A tensor of adjusted timesteps such that the final value equals `self.shift_terminal`.
        """
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | Device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ) -> None:
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `Device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`List[float]`, *optional*):
                Custom values for sigmas to be used for each diffusion step. If `None`, the sigmas are computed
                automatically.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting.
            timesteps (`List[float]`, *optional*):
                Custom values for timesteps to be used for each diffusion step. If `None`, the timesteps are computed
                automatically.
        """
        if self.use_dynamic_shifting and mu is None:
            raise ValueError(
                "`mu` must be passed when `use_dynamic_shifting` is set to be `True`"
            )

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length"
                )

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
                )
        else:
            num_inference_steps = (
                len(sigmas) if sigmas is not None else len(timesteps)
            )

        self.num_inference_steps = num_inference_steps

        # 1. Prepare default sigmas
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = Tensor.constant(timesteps).cast(DType.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = linspace(
                    self._sigma_to_t(self.sigma_max),
                    self._sigma_to_t(self.sigma_min),
                    num_inference_steps,
                )
            sigmas = timesteps / self.num_train_timesteps
        else:
            sigmas = Tensor.constant(sigmas).cast(DType.float32)
            num_inference_steps = len(sigmas)

        # 2. Perform timestep shifting. Either no shifting is applied, or resolution-dependent shifting of
        #    "exponential" or "linear" type is applied
        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # 3. If required, stretch the sigmas schedule to terminate at the configured `shift_terminal` value
        if self.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        # 4. If required, convert sigmas to one of karras, exponential, or beta sigma schedules
        if self.use_karras_sigmas:
            sigmas = self._convert_to_karras(
                in_sigmas=sigmas, num_inference_steps=num_inference_steps
            )
        elif self.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(
                in_sigmas=sigmas, num_inference_steps=num_inference_steps
            )
        elif self.use_beta_sigmas:
            sigmas = self._convert_to_beta(
                in_sigmas=sigmas, num_inference_steps=num_inference_steps
            )

        # 5. Convert sigmas and timesteps to tensors and move to specified device
        sigmas = sigmas.to(device).cast(DType.float32)
        if not is_timesteps_provided:
            timesteps = sigmas * self.num_train_timesteps
        else:
            timesteps = timesteps.to(device).cast(DType.float32)

        # 6. Append the terminal sigma value.
        #    If a model requires inverted sigma schedule for denoising but timesteps without inversion, the
        #    `invert_sigmas` flag can be set to `True`. This case is only required in Mochi
        if self.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.num_train_timesteps
            sigmas = F.concat(
                [
                    sigmas,
                    Tensor.ones([1], device=sigmas.device, dtype=sigmas.dtype),
                ]
            )
        else:
            sigmas = F.concat(
                [
                    sigmas,
                    Tensor.zeros([1], device=sigmas.device, dtype=sigmas.dtype),
                ]
            )

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(
        self, timestep: Tensor, schedule_timesteps: Tensor | None = None
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep: Tensor) -> None:
        if self.begin_index is None:
            if isinstance(timestep, Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: Tensor,
        timestep: float | Tensor,
        sample: Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        # generator: Generator | None = None,
        per_token_timesteps: Tensor | None = None,
        return_dict: bool = True,
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`Tensor`):
                The direct output from learned diffusion model.
            timestep (`float` or `Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`Tensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.cast(DType.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(axis=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]

            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma

        if self.stochastic_sampling:
            x0 = sample - current_sigma * model_output
            noise = random.normal(sample)
            prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            prev_sample = sample + dt * model_output

        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.cast(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def _convert_to_karras(
        self, in_sigmas: Tensor, num_inference_steps: int
    ) -> Tensor:
        """
        Construct the noise schedule as proposed in [Elucidating the Design Space of Diffusion-Based Generative
        Models](https://huggingface.co/papers/2206.00364).

        Args:
            in_sigmas (`Tensor`):
                The input sigma values to be converted.
            num_inference_steps (`int`):
                The number of inference steps to generate the noise schedule for.

        Returns:
            `Tensor`:
                The converted sigma values following the Karras noise schedule.
        """

        # Hack to make sure that other schedulers which copy this function don't break
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _convert_to_exponential(
        self, in_sigmas: Tensor, num_inference_steps: int
    ) -> Tensor:
        """
        Construct an exponential noise schedule.

        Args:
            in_sigmas (`Tensor`):
                The input sigma values to be converted.
            num_inference_steps (`int`):
                The number of inference steps to generate the noise schedule for.

        Returns:
            `Tensor`:
                The converted sigma values following an exponential schedule.
        """

        # Hack to make sure that other schedulers which copy this function don't break
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = F.exp(
            linspace(F.log(sigma_max), F.log(sigma_min), num_inference_steps)
        )
        return sigmas

    def _convert_to_beta(
        self,
        in_sigmas: Tensor,
        num_inference_steps: int,
        alpha: float = 0.6,
        beta: float = 0.6,
    ) -> Tensor:
        """
        Construct a beta noise schedule as proposed in [Beta Sampling is All You
        Need](https://huggingface.co/papers/2407.12173).

        Args:
            in_sigmas (`Tensor`):
                The input sigma values to be converted.
            num_inference_steps (`int`):
                The number of inference steps to generate the noise schedule for.
            alpha (`float`, *optional*, defaults to `0.6`):
                The alpha parameter for the beta distribution.
            beta (`float`, *optional*, defaults to `0.6`):
                The beta parameter for the beta distribution.

        Returns:
            `Tensor`:
                The converted sigma values following a beta distribution schedule.
        """

        # Hack to make sure that other schedulers which copy this function don't break
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = Tensor.constant(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(
                        timestep, alpha, beta
                    )  # TODO: natively in Modular?
                    for timestep in 1 - linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas

    def _time_shift_exponential(
        self, mu: Tensor, sigma: Tensor, t: Tensor
    ) -> Tensor:
        return F.exp(mu) / (F.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(
        self, mu: Tensor, sigma: Tensor, t: Tensor
    ) -> Tensor:
        return mu / (mu + (1 / t - 1) ** sigma)

    def __len__(self) -> int:
        return self.num_train_timesteps
