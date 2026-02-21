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
"""Flow Match Euler Discrete Scheduler for diffusion models."""

import numpy as np
import numpy.typing as npt


class FlowMatchEulerDiscreteScheduler:
    """Minimal Flow Match Euler Discrete Scheduler.

    This scheduler provides timestep and sigma scheduling for flow-matching
    diffusion models. The actual denoising step computation is handled by
    the pipeline (e.g., FluxPipeline._scheduler_step).
    """

    def __init__(
        self,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        use_flow_sigmas: bool = False,
        use_dynamic_shifting: bool = False,
        use_empirical_mu: bool = False,
        order: int = 1,
        **unused_kwargs,
    ) -> None:
        """Initialize the scheduler.

        Args:
            base_image_seq_len: Base image sequence length.
            max_image_seq_len: Maximum image sequence length.
            base_shift: Base shift.
            max_shift: Maximum shift.
            use_flow_sigmas: Whether to use flow sigmas.
            use_dynamic_shifting: Whether to use dynamic shifting.
            use_empirical_mu: Whether to use empirical mu.
            order: Order of the scheduler.
            **unused_kwargs: Unused keyword arguments.
        """
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.use_flow_sigmas = use_flow_sigmas
        self.use_dynamic_shifting = use_dynamic_shifting
        self.use_empirical_mu = use_empirical_mu
        self.order = order

        self._use_flow_sigmas = use_flow_sigmas
        self._shift_slope = (max_shift - base_shift) / (
            max_image_seq_len - base_image_seq_len
        )
        self._shift_intercept = (
            base_shift - self._shift_slope * base_image_seq_len
        )
        self._use_empirical_mu = use_empirical_mu
        self._use_dynamic_shifting = use_dynamic_shifting

    @staticmethod
    def _time_shift_exponential(
        mu: float, sigma_param: float, t: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Resolution-dependent timestep shift (diffusers FlowMatchEulerDiscreteScheduler)."""
        t_safe = np.clip(t.astype(np.float64), 1e-7, 1.0)
        out = np.exp(mu) / (np.exp(mu) + (1.0 / t_safe - 1.0) ** sigma_param)
        return out.astype(np.float32)

    @staticmethod
    def _compute_empirical_mu(
        image_seq_len: int, num_inference_steps: int
    ) -> float:
        """Compute empirical mu for Flux2 timestep scheduling.

        Taken from:
        https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/sampling.py#L251

        Args:
            image_seq_len: Length of image sequence (H*W after packing).
            num_inference_steps: Number of inference steps.

        Returns:
            Empirical mu value for scheduler.
        """
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_inference_steps + b

        return float(mu)

    def _calculate_mu(
        self, image_seq_len: int, num_inference_steps: int
    ) -> float:
        if self._use_empirical_mu:
            mu = self._compute_empirical_mu(image_seq_len, num_inference_steps)
        else:
            mu = self._shift_slope * image_seq_len + self._shift_intercept
            mu = float(mu)
        return mu

    def retrieve_timesteps_and_sigmas(
        self,
        image_seq_len: int,
        num_inference_steps: int,
        reverse: bool = False,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Retrieve timesteps and sigmas for the diffusion process.

        Args:
            image_seq_len: Length of image sequence (H*W after packing).
            num_inference_steps: Number of inference steps.
            reverse: Whether to reverse the timesteps and sigmas.

        Returns:
            Tuple of timesteps and sigmas.
        """
        if not self._use_flow_sigmas:
            sigmas = np.linspace(
                1.0,
                1.0 / num_inference_steps,
                num_inference_steps,
                dtype=np.float32,
            )
            if self._use_dynamic_shifting:
                mu = self._calculate_mu(image_seq_len, num_inference_steps)
                sigmas = self._time_shift_exponential(mu, 1.0, sigmas)
            timesteps = sigmas * 1000.0
            if reverse:
                timesteps = ((1000.0 - timesteps) / 1000.0).astype(np.float32)
            else:
                timesteps = (timesteps / 1000.0).astype(np.float32)
            # Append final sigma of 0.0 for the last scheduler step
            sigmas = np.append(sigmas, np.float32(0.0))
        else:
            # Generate default timesteps
            if reverse:
                timesteps = np.linspace(
                    1000, 0, num_inference_steps, dtype=np.float32
                )
            else:
                timesteps = np.linspace(
                    0, 1000, num_inference_steps, dtype=np.float32
                )
            sigmas = timesteps / 1000.0
        return timesteps, sigmas
