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
"""Tests for provider options schemas."""

import json

import pytest
from max.interfaces.provider_options import (
    ImageProviderOptions,
    MaxProviderOptions,
    ProviderOptions,
)
from pydantic import ValidationError


def test_import_provider_options() -> None:
    """Test that all provider option types can be imported."""
    # If we get here, all imports succeeded
    assert MaxProviderOptions is not None
    assert ImageProviderOptions is not None
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


def test_image_provider_options_minimal() -> None:
    """Test creating ImageProviderOptions with no fields."""
    opts = ImageProviderOptions()
    assert opts.negative_prompt is None
    assert opts.width is None
    assert opts.height is None


def test_image_provider_options_with_param() -> None:
    """Test creating ImageProviderOptions with parameters."""
    opts = ImageProviderOptions(width=1024, height=768, num_images=2)
    assert opts.width == 1024
    assert opts.height == 768
    assert opts.num_images == 2


def test_image_provider_options_frozen() -> None:
    """Test that ImageProviderOptions is frozen (immutable)."""
    opts = ImageProviderOptions(width=512, height=512)

    with pytest.raises(ValidationError):
        opts.width = 1024  # type: ignore[misc]


def test_provider_options_empty() -> None:
    """Test creating ProviderOptions with no fields."""
    opts = ProviderOptions()
    assert opts.max is None
    assert opts.image is None
    assert opts.video is None


def test_provider_options_with_max_only() -> None:
    """Test creating ProviderOptions with only MAX options."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123")
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.image is None
    assert opts.video is None


def test_provider_options_with_image_only() -> None:
    """Test creating ProviderOptions with only image modality options."""
    opts = ProviderOptions(image=ImageProviderOptions(width=512, height=512))
    assert opts.max is None
    assert opts.image is not None
    assert opts.image.width == 512
    assert opts.image.height == 512
    assert opts.video is None


def test_provider_options_with_all_fields() -> None:
    """Test creating ProviderOptions with both MAX and modality options."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123"),
        image=ImageProviderOptions(width=512, height=512),
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.image is not None
    assert opts.image.width == 512
    assert opts.image.height == 512


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
        image=ImageProviderOptions(width=1024, height=768),
    )

    json_str = opts.model_dump_json()
    json_data = json.loads(json_str)

    assert json_data["max"]["target_endpoint"] == "instance-123"
    assert json_data["image"]["width"] == 1024
    assert json_data["image"]["height"] == 768


def test_provider_options_json_deserialization() -> None:
    """Test that ProviderOptions can be deserialized from JSON."""
    json_data = {
        "max": {"target_endpoint": "instance-123"},
        "image": {"width": 512, "height": 512},
    }

    opts = ProviderOptions.model_validate(json_data)

    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.image is not None
    assert opts.image.width == 512
    assert opts.image.height == 512


def test_provider_options_json_deserialization_partial() -> None:
    """Test deserializing ProviderOptions with only some fields."""
    # Only MAX options
    max_json_data = {"max": {"target_endpoint": "instance-123"}}
    opts = ProviderOptions.model_validate(max_json_data)
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.image is None
    assert opts.video is None

    # Only image options
    image_json_data = {"image": {"width": 512, "height": 512}}
    opts = ProviderOptions.model_validate(image_json_data)
    assert opts.max is None
    assert opts.image is not None
    assert opts.image.width == 512
    assert opts.image.height == 512
    assert opts.video is None


def test_provider_options_nested_validation() -> None:
    """Test that nested validation works correctly."""
    # Valid nested structure with explicit type
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123")
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"

    # Valid nested structure from dict (Pydantic auto-converts)
    opts = ProviderOptions.model_validate(
        {"max": {"target_endpoint": "instance-456"}}
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-456"

    # Invalid nested structure should fail at creation
    with pytest.raises(ValidationError):
        ProviderOptions.model_validate({"max": {"invalid_field": "value"}})


class TestImageDimensionValidation:
    """Tests for height/width validation on ImageProviderOptions."""

    def test_dimensions_none_is_valid(self) -> None:
        opts = ImageProviderOptions()
        assert opts.width is None
        assert opts.height is None

    def test_valid_dimensions(self) -> None:
        opts = ImageProviderOptions(width=512, height=512)
        assert opts.width == 512
        assert opts.height == 512

    def test_minimum_dimensions(self) -> None:
        opts = ImageProviderOptions(width=128, height=128)
        assert opts.width == 128
        assert opts.height == 128

    def test_width_too_small(self) -> None:
        with pytest.raises(
            ValidationError, match="greater than or equal to 128"
        ):
            ImageProviderOptions(width=64, height=512)

    def test_height_too_small(self) -> None:
        with pytest.raises(
            ValidationError, match="greater than or equal to 128"
        ):
            ImageProviderOptions(width=512, height=64)

    def test_width_not_multiple_of_16(self) -> None:
        with pytest.raises(ValidationError, match="must be a multiple of 16"):
            ImageProviderOptions(width=130, height=512)

    def test_height_not_multiple_of_16(self) -> None:
        with pytest.raises(ValidationError, match="must be a multiple of 16"):
            ImageProviderOptions(width=512, height=130)

    def test_pixel_area_exceeds_max(self) -> None:
        with pytest.raises(
            ValidationError, match="exceeds the maximum pixel area"
        ):
            ImageProviderOptions(width=2048, height=1024)

    def test_pixel_area_at_max(self) -> None:
        opts = ImageProviderOptions(width=1024, height=1024)
        assert opts.width == 1024
        assert opts.height == 1024

    def test_only_width_set_skips_area_check(self) -> None:
        opts = ImageProviderOptions(width=2048)
        assert opts.width == 2048

    def test_only_height_set_skips_area_check(self) -> None:
        opts = ImageProviderOptions(height=2048)
        assert opts.height == 2048
