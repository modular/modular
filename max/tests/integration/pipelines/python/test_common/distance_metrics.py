# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Distance function definitions for use in testing infrastructure."""

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F


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
    logits_predicted = torch.from_numpy(predicted)
    logits_expected = torch.from_numpy(expected)

    # Convert logits to log-probabilities.
    log_probs_expected = F.log_softmax(logits_expected, dim=-1)
    log_probs_predicted = F.log_softmax(logits_predicted, dim=-1)

    # Compute the KL divergence.
    return F.kl_div(
        log_probs_predicted,
        log_probs_expected,
        reduction="sum",
        log_target=True,
    )


def kl_divergence_verifier(
    predicted: npt.NDArray[np.floating],
    expected: npt.NDArray[np.floating],
    description: str,
    threshold: float = 0.1,
) -> None:
    """Verifies predicted vs. expected vectors using KL divergence.

    This is preferable to elementwise comparison for vector distances, because
    elementwise comparison doesn't take into account the magnitude of the
    vectors.
    Furthermore since we are comparing logits, use the KL divergence as a
    distance between expected and predicted probability distributions (over
    next token classification).
    """
    if isinstance(predicted, np.int64):
        # Assert that token ids generated match PyTorch exactly.
        assert predicted == expected, description
        return

    eps_bf16 = torch.finfo(torch.bfloat16).eps
    if isinstance(predicted, np.float32):
        # Treat the "next logits" element specially until it is removed.
        assert np.isclose(
            predicted, expected, rtol=2 * eps_bf16, atol=2 * eps_bf16
        )
        return

    kl_divergence = kl_divergence_from_logits(predicted, expected)

    # Assert that the KL divergence between predicted and expected log
    # probabilities is below the threshold.
    assert kl_divergence < threshold, description
