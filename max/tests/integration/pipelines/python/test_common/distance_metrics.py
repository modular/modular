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


def kl_divergence_from_logits(
    predicted: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
) -> float:
    """Computes the KL divergence between predicted and expected logits."""
    # Use the same formula as KLDivergenceValidator._compute_distance
    eps = 1e-9
    return rel_entr(softmax(expected, -1), softmax(predicted, -1) + eps).sum(-1)
