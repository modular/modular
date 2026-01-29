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
"""Tests for provider options schemas."""

import json

import pytest
from max.interfaces.provider_options import (
    MaxProviderOptions,
    PixelModalityOptions,
    ProviderOptions,
)
from pydantic import ValidationError


def test_import_provider_options() -> None:
    """Test that all provider option types can be imported."""
    # If we get here, all imports succeeded
    assert MaxProviderOptions is not None
    assert PixelModalityOptions is not None
    assert ProviderOptions is not None


def test_max_provider_options_minimal() -> None:
    """Test creating MaxProviderOptions with no fields."""
    opts = MaxProviderOptions()
    assert opts.target_endpoint is None


def test_max_provider_options_with_target_endpoint() -> None:
    """Test creating MaxProviderOptions with target_endpoint."""
    opts = MaxProviderOptions(target_endpoint="instance-123")
    assert opts.target_endpoint == "instance-123"


def test_max_provider_options_frozen() -> None:
    """Test that MaxProviderOptions is frozen (immutable)."""
    opts = MaxProviderOptions(target_endpoint="instance-123")

    with pytest.raises(ValidationError):
        opts.target_endpoint = "instance-456"  # type: ignore[misc]


def test_pixel_modality_options_minimal() -> None:
    """Test creating PixelModalityOptions with no fields."""
    opts = PixelModalityOptions()
    assert opts.dummy_param is None


def test_pixel_modality_options_with_param() -> None:
    """Test creating PixelModalityOptions with dummy_param."""
    opts = PixelModalityOptions(dummy_param="test-value")
    assert opts.dummy_param == "test-value"


def test_pixel_modality_options_frozen() -> None:
    """Test that PixelModalityOptions is frozen (immutable)."""
    opts = PixelModalityOptions(dummy_param="test")

    with pytest.raises(ValidationError):
        opts.dummy_param = "new-value"  # type: ignore[misc]


def test_provider_options_empty() -> None:
    """Test creating ProviderOptions with no fields."""
    opts = ProviderOptions()
    assert opts.max is None
    assert opts.pixel is None


def test_provider_options_with_max_only() -> None:
    """Test creating ProviderOptions with only MAX options."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123")
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.pixel is None


def test_provider_options_with_pixel_only() -> None:
    """Test creating ProviderOptions with only pixel modality options."""
    opts = ProviderOptions(pixel=PixelModalityOptions(dummy_param="test"))
    assert opts.max is None
    assert opts.pixel is not None
    assert opts.pixel.dummy_param == "test"


def test_provider_options_with_all_fields() -> None:
    """Test creating ProviderOptions with both MAX and modality options."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123"),
        pixel=PixelModalityOptions(dummy_param="test"),
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.pixel is not None
    assert opts.pixel.dummy_param == "test"


def test_provider_options_frozen() -> None:
    """Test that ProviderOptions is frozen (immutable)."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123")
    )

    with pytest.raises(ValidationError):
        opts.max = MaxProviderOptions(target_endpoint="instance-456")  # type: ignore[misc]


def test_provider_options_json_serialization() -> None:
    """Test that ProviderOptions can be serialized to JSON."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123"),
        pixel=PixelModalityOptions(dummy_param="test"),
    )

    json_str = opts.model_dump_json()
    json_data = json.loads(json_str)

    assert json_data["max"]["target_endpoint"] == "instance-123"
    assert json_data["pixel"]["dummy_param"] == "test"


def test_provider_options_json_deserialization() -> None:
    """Test that ProviderOptions can be deserialized from JSON."""
    json_data = {
        "max": {"target_endpoint": "instance-123"},
        "pixel": {"dummy_param": "test"},
    }

    opts = ProviderOptions(**json_data)

    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.pixel is not None
    assert opts.pixel.dummy_param == "test"


def test_provider_options_json_deserialization_partial() -> None:
    """Test deserializing ProviderOptions with only some fields."""
    # Only MAX options
    json_data = {"max": {"target_endpoint": "instance-123"}}
    opts = ProviderOptions(**json_data)
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.pixel is None

    # Only pixel options
    json_data = {"pixel": {"dummy_param": "test"}}
    opts = ProviderOptions(**json_data)
    assert opts.max is None
    assert opts.pixel is not None
    assert opts.pixel.dummy_param == "test"


def test_provider_options_nested_validation() -> None:
    """Test that nested validation works correctly."""
    # Valid nested structure with explicit type
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123")
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"

    # Valid nested structure from dict (Pydantic auto-converts)
    opts = ProviderOptions(max={"target_endpoint": "instance-456"})
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-456"

    # Invalid nested structure should fail at creation
    with pytest.raises(ValidationError):
        ProviderOptions(max={"invalid_field": "value"})
