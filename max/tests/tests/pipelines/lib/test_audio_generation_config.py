# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for AudioGenerationConfig."""

from max.pipelines.lib import AudioGenerationConfig


def test_audio_generation_config_missing_help_method() -> None:
    """Test that AudioGenerationConfig is missing a help() method and should have one."""

    # Check if AudioGenerationConfig has its own help method or inherits from PipelineConfig
    assert "help" in AudioGenerationConfig.__dict__, (
        "AudioGenerationConfig should have its own help() method"
    )
