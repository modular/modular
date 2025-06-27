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

    Attributes:
        token_log_probabilities (list[float]): Probabilities of each token.
        top_log_probabilities (list[dict[int, float]]): Top tokens and their corresponding probabilities.

    """

    token_log_probabilities: list[float] = msgspec.field(default_factory=list)
    top_log_probabilities: list[dict[int, float]] = msgspec.field(
        default_factory=list
    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogProbabilities):
            return False

        if len(self.token_log_probabilities) != len(
            other.token_log_probabilities
        ):
            return False

        if not all(
            a == b
            for a, b in zip(
                self.token_log_probabilities, other.token_log_probabilities
            )
        ):
            return False

        if len(self.top_log_probabilities) != len(other.top_log_probabilities):
            return False

        if not all(
            a == b
            for a, b in zip(
                self.top_log_probabilities, other.top_log_probabilities
            )
        ):
            return False

        return True
