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
"""Distance function definitions for use in testing infrastructure."""

import numpy as np
import numpy.typing as npt
from scipy.special import rel_entr, softmax


def is_euclidean_distance_close(
    result: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
    rtol: float = 0.01,
    atol: float = 1e-5,
) -> bool:
    """Computes whether the Euclidean distance between inputs is close."""
    diff_norm = np.linalg.norm(result - expected)
    return bool(
        diff_norm < atol
        or diff_norm / (np.linalg.norm(expected) + np.finfo(np.float32).eps)
        < rtol
    )


def _smooth_probs(
    probs: npt.NDArray[np.floating], eps: float = 1e-10
) -> npt.NDArray[np.floating]:
    """Smooths probabilities so no entry is below eps.

    Applies (1 - N*eps) * probs + eps, which preserves the simplex
    (probabilities still sum to 1) and guarantees a minimum value of eps.

    The choice of eps is a tradeoff between numerical stability and
    vocabulary size N. The (1 - N*eps) factor redistributes probability
    away from softmax values to fund the eps floor. For typical vocab
    sizes (~2e5), N*eps is ~2e-5, which is negligible for any
    significant probability.
    """
    n = probs.shape[-1]
    return (1 - n * eps) * probs + eps


def kl_divergence_from_logits(
    predicted: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
) -> float:
    """Computes the KL divergence between predicted and expected logits."""
    return rel_entr(
        _smooth_probs(softmax(expected, -1)),
        _smooth_probs(softmax(predicted, -1)),
    ).sum(-1)
