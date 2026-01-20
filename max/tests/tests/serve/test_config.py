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

"""Tests for MAX Serve configuration."""

import os

import pytest
from max.serve.config import Settings


def test_deprecated_dispatcher_config_fails() -> None:
    """Test that using deprecated MAX_SERVE_DISPATCHER_CONFIG fails loudly."""
    # Set the deprecated environment variable
    os.environ["MAX_SERVE_DISPATCHER_CONFIG"] = "tcp://localhost:5555"

    try:
        # Attempting to create Settings should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            Settings()

        # Verify the error message contains helpful information
        error_message = str(exc_info.value)
        assert "MAX_SERVE_DISPATCHER_CONFIG" in error_message
        assert "deprecated" in error_message.lower()
        assert "MAX_SERVE_DI_BIND_ADDRESS" in error_message
        assert "CLIN-608" in error_message
    finally:
        # Clean up the environment variable
        del os.environ["MAX_SERVE_DISPATCHER_CONFIG"]


def test_di_bind_address_works_without_deprecated_var() -> None:
    """Test that MAX_SERVE_DI_BIND_ADDRESS works correctly when deprecated var is not set."""
    # Ensure the deprecated variable is not set
    if "MAX_SERVE_DISPATCHER_CONFIG" in os.environ:
        del os.environ["MAX_SERVE_DISPATCHER_CONFIG"]

    # Test with default value
    settings = Settings()
    assert settings.di_bind_address == "tcp://127.0.0.1:5555"

    # Test with custom value via environment variable
    os.environ["MAX_SERVE_DI_BIND_ADDRESS"] = "tcp://0.0.0.0:6666"
    try:
        settings = Settings()
        assert settings.di_bind_address == "tcp://0.0.0.0:6666"
    finally:
        del os.environ["MAX_SERVE_DI_BIND_ADDRESS"]


def test_deprecated_dispatcher_config_fails_even_with_valid_di_bind_address() -> (
    None
):
    """Test that deprecated variable causes failure even if new variable is also set."""
    os.environ["MAX_SERVE_DISPATCHER_CONFIG"] = "tcp://old:5555"
    os.environ["MAX_SERVE_DI_BIND_ADDRESS"] = "tcp://new:6666"

    try:
        with pytest.raises(ValueError) as exc_info:
            Settings()

        error_message = str(exc_info.value)
        assert "MAX_SERVE_DISPATCHER_CONFIG" in error_message
        assert "deprecated" in error_message.lower()
    finally:
        if "MAX_SERVE_DISPATCHER_CONFIG" in os.environ:
            del os.environ["MAX_SERVE_DISPATCHER_CONFIG"]
        if "MAX_SERVE_DI_BIND_ADDRESS" in os.environ:
            del os.environ["MAX_SERVE_DI_BIND_ADDRESS"]
