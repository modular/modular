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
"""Distribution classes for autoencoders."""

from max import functional as F
from max import random
from max.tensor import Tensor


class DiagonalGaussianDistribution:
    """A diagonal Gaussian distribution.

    Used by Variational Autoencoders (VAEs) to parameterize the latent space.
    """

    def __init__(self, parameters: Tensor, deterministic: bool = False) -> None:
        """Initialize parameters.

        Args:
            parameters: Tensor containing mean and log-variance.
            deterministic: If True, the distribution is treated as delta.
        """
        self.parameters = parameters
        self.mean, self.logvar = F.split(parameters, 2, axis=1)
        self.logvar = F.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = F.exp(0.5 * self.logvar)
        self.var = F.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = Tensor.constant(0.0, self.mean.shape)

    def sample(self, seed: int | None = None) -> Tensor:
        """Sample from the distribution.

        Args:
            seed: Optional random seed for sampling.

        Returns:
            A sample from the Gaussian distribution.
        """
        sample = random.gaussian(self.mean.shape, seed=seed)
        return self.mean + self.std * sample

    def mode(self) -> Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean
