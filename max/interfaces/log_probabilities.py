# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import msgspec


class LogProbabilities(msgspec.Struct, tag=True, omit_defaults=True):
    """Log probabilities for an individual output token.

    This is a data-only class that serves as a serializable data structure for
    transferring log probability information. It does not provide any functionality
    for calculating or manipulating log probabilities - it is purely for data storage
    and serialization purposes.
    """

    token_log_probabilities: list[float] = msgspec.field(default_factory=list)
    """Probabilities of each token."""
    top_log_probabilities: list[dict[int, float]] = msgspec.field(
        default_factory=list
    )
    """Top tokens and their corresponding probabilities."""
