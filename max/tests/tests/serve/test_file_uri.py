# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test file URI support in MAX serve."""

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from max.serve.router.openai_routes import resolve_image_from_url
from PIL import Image
from pydantic import AnyUrl


def create_test_image_bytes() -> bytes:
    """Create a minimal test image and return it as JPEG bytes."""
    img = Image.new("RGB", (10, 10), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_file_uri_absolute_path(tmp_path):
    """Test resolving absolute file URIs."""
    test_image = tmp_path / "test.jpg"
    test_data = create_test_image_bytes()
    test_image.write_bytes(test_data)

    settings = MagicMock()
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    result = await resolve_image_from_url(
        AnyUrl(f"file://{test_image}"), settings
    )
    assert result == test_data


@pytest.mark.asyncio
async def test_file_uri_not_found(tmp_path):
    """Test error for missing files."""
    settings = MagicMock()
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="not found"):
        await resolve_image_from_url(
            AnyUrl(f"file://{tmp_path}/missing.jpg"), settings
        )


@pytest.mark.asyncio
async def test_file_uri_path_traversal_blocked(tmp_path):
    """Test path traversal protection."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    # Create a file outside the allowed directory
    secret_file = tmp_path / "secret.jpg"
    secret_file.write_bytes(create_test_image_bytes())

    settings = MagicMock()
    settings.allowed_image_roots = [str(allowed)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="forbidden"):
        await resolve_image_from_url(
            AnyUrl(f"file://{allowed}/../secret.jpg"), settings
        )


@pytest.mark.asyncio
async def test_file_uri_directory_blocked(tmp_path):
    """Test directories cannot be accessed."""
    settings = MagicMock()
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="directory"):
        await resolve_image_from_url(AnyUrl(f"file://{tmp_path}"), settings)


@pytest.mark.asyncio
async def test_file_uri_size_limit(tmp_path):
    """Test file size limit enforcement."""
    large_file = tmp_path / "large.jpg"
    large_file.write_bytes(b"x" * (21 * 1024 * 1024))  # 21MB

    settings = MagicMock()
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20 * 1024 * 1024  # 20MB

    with pytest.raises(ValueError, match="exceeds.*size limit"):
        await resolve_image_from_url(AnyUrl(f"file://{large_file}"), settings)


@pytest.mark.asyncio
async def test_file_uri_url_encoded_path(tmp_path):
    """Test URL-encoded file paths."""
    test_image = tmp_path / "test image.jpg"
    test_data = create_test_image_bytes()
    test_image.write_bytes(test_data)

    settings = MagicMock()
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    encoded_path = str(test_image).replace(" ", "%20")
    result = await resolve_image_from_url(
        AnyUrl(f"file://{encoded_path}"), settings
    )
    assert result == test_data


@pytest.mark.asyncio
async def test_file_uri_localhost_host(tmp_path):
    """Test file URI with localhost host."""
    test_image = tmp_path / "test.jpg"
    test_data = create_test_image_bytes()
    test_image.write_bytes(test_data)

    settings = MagicMock()
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    result = await resolve_image_from_url(
        AnyUrl(f"file://localhost{test_image}"), settings
    )
    assert result == test_data


@pytest.mark.asyncio
async def test_file_uri_remote_host_rejected(tmp_path):
    """Test remote hosts are rejected."""
    settings = MagicMock()
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="remote host"):
        await resolve_image_from_url(
            AnyUrl("file://example.com/image.jpg"), settings
        )


@pytest.mark.asyncio
async def test_file_uri_no_allowed_roots(tmp_path):
    """Test access blocked with no allowed roots."""
    test_file = tmp_path / "test.jpg"
    test_file.write_bytes(create_test_image_bytes())

    settings = MagicMock()
    settings.allowed_image_roots = []
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="no allowed roots"):
        await resolve_image_from_url(AnyUrl(f"file://{test_file}"), settings)


@pytest.mark.asyncio
async def test_file_uri_symlink_escape(tmp_path):
    """Test symlinks cannot escape allowed roots."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(create_test_image_bytes())

    symlink = allowed / "link.jpg"
    try:
        symlink.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks not supported")

    settings = MagicMock()
    settings.allowed_image_roots = [str(allowed)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="forbidden"):
        await resolve_image_from_url(AnyUrl(f"file://{symlink}"), settings)
