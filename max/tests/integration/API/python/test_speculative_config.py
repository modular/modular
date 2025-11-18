# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Unit tests for SpeculativeConfig."""

from __future__ import annotations

from max.pipelines.lib.speculative_config import (
    SpeculativeConfig,
    SpeculativeMethod,
)


def test_is_standalone() -> None:
    """Verify is_standalone() returns correct boolean."""
    assert SpeculativeConfig(
        speculative_method=SpeculativeMethod.STANDALONE
    ).is_standalone()
    assert not SpeculativeConfig(speculative_method=None).is_standalone()


def test_num_speculative_tokens() -> None:
    """Verify num_speculative_tokens uses default and accepts custom values."""
    assert SpeculativeConfig().num_speculative_tokens == 5
    assert (
        SpeculativeConfig(num_speculative_tokens=10).num_speculative_tokens
        == 10
    )
    assert (
        SpeculativeConfig(num_speculative_tokens=1).num_speculative_tokens == 1
    )


def test_enum_values() -> None:
    """Verify expected SpeculativeMethod enum values exist."""
    assert SpeculativeMethod.STANDALONE == "standalone"
