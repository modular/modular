# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test utilities for MAXConfig testing."""

from max.pipelines.lib import MAXConfig


def assert_help_covers_all_public_fields(
    config: MAXConfig, config_name: str
) -> None:
    """Utility function to test that a config's help() method covers all public fields.

    This function validates that every public field (not starting with underscore)
    in a MAXConfig class has corresponding documentation in its help() method.

    Args:
        config: Instance of a MAXConfig class to test
        config_name: Name of the config class for error messages

    Raises:
        AssertionError: If the help() method doesn't document all public fields
    """
    help_dict = config.help()

    # Count only public fields (not starting with underscore).
    public_fields = {
        name for name in config.__dataclass_fields__ if not name.startswith("_")
    }

    # Test that help dict has the same number of fields as public fields.
    assert len(help_dict) == len(public_fields), (
        f"{config_name} help() should document all {len(public_fields)} public fields, "
        f"but only documents {len(help_dict)} fields. "
        f"Missing: {sorted(public_fields - set(help_dict.keys()))}, "
        f"Extra: {sorted(set(help_dict.keys()) - public_fields)}"
    )
